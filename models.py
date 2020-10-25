import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from torch.nn.utils import weight_norm
from math import floor
import numpy as np
import os
from dataloader import build_pretrain_embedding, load_embeddings


class WordRep(nn.Module):
    def __init__(self, args, Y, dicts):
        super(WordRep, self).__init__()

        self.gpu = args.gpu

        if args.embed_file:
            print("loading pretrained embeddings from {}".format(args.embed_file))
            if args.use_ext_emb:
                pretrain_word_embedding, pretrain_emb_dim = build_pretrain_embedding(args.embed_file, dicts['w2ind'], True)
                W = torch.from_numpy(pretrain_word_embedding)
            else:
                W = torch.Tensor(load_embeddings(args.embed_file))

            self.embed = nn.Embedding(W.size()[0], W.size()[1], padding_idx=0)
            self.embed.weight.data = W.clone()
        else:
            # add 2 to include UNK and PAD
            self.embed = nn.Embedding(len(dicts['w2ind']) + 2, args.embed_size, padding_idx=0)
        self.feature_size = self.embed.embedding_dim

        self.embed_drop = nn.Dropout(p=args.dropout)

    def forward(self, x, target, text_inputs):
        # x: [bs, seq_len]
        embed = self.embed(x)
        x = self.embed_drop(embed)
        return x


def label_smoothing(y, alpha, Y):
    return y*(1-alpha) + alpha/Y


class OutputLayer(nn.Module):
    def __init__(self, args, Y, dicts, input_size):
        super(OutputLayer, self).__init__()
        self.args = args
        self.Y = Y
        self.U = nn.Linear(input_size, Y)
        self.final = nn.Linear(input_size, Y)
        xavier_uniform_(self.U.weight)
        xavier_uniform_(self.final.weight)
        self.loss_func = nn.BCEWithLogitsLoss()


    def forward(self, x, target, text_inputs):
        att = self.U.weight.matmul(x.transpose(1, 2)) # [bs, Y, seq_len]
        alpha = F.softmax(att, dim=2)
        m = alpha.matmul(x)     # [bs, Y, dim]
        logits = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)
        if self.args.label_smoothing:
            target = label_smoothing(target, self.args.alpha, self.Y)
            yhat = torch.sigmoid(logits)
            loss = torch.mean(-target*torch.log(yhat) - (1-target)*torch.log(1-yhat))
        else:
            loss = self.loss_func(logits, target)
        return logits, loss


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        xavier_uniform_(self.conv1.weight)
        xavier_uniform_(self.conv2.weight)
        if self.downsample is not None:
            xavier_uniform_(self.downsample.weight)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class DCAN(nn.Module):
    def __init__(self, args, Y, dicts):
        super(DCAN, self).__init__()
        self.configs = args
        self.word_rep = WordRep(args, Y, dicts)
        num_chans = [args.nhid] * args.levels
        self.tcn = TemporalConvNet(self.word_rep.feature_size, num_chans, args.kernel_size, args.dropout)
        self.lin = nn.Linear(num_chans[-1], args.nproj)
        self.output_layer = OutputLayer(args, Y, dicts, args.nproj)

        xavier_uniform_(self.lin.weight)

    def forward(self, data, target, text_inputs=None):
        # data: [bs, len]
        bs, seq_len = data.size(0), data.size(1)
        x = self.word_rep(data, target, text_inputs)   # [bs, seq_len, dim_embed]
        hid_seq = self.tcn(x.transpose(1, 2)).transpose(1, 2)   # [bs, seq_len, nhid]
        hid_seq = F.relu(self.lin(hid_seq))
        logits, loss = self.output_layer(hid_seq, target, None)
        return logits, loss
    
    def freeze_net(self):
        for p in self.word_rep.embed.parameters():
            p.requires_grad = False


class CNN(nn.Module):
    def __init__(self, args, Y, dicts):
        super(CNN, self).__init__()

        self.word_rep = WordRep(args, Y, dicts)
        filter_size = int(args.filter_size)
        self.conv = nn.Conv1d(self.word_rep.feature_size, args.num_filter_maps, kernel_size=filter_size,
                                  padding=int(floor(filter_size / 2)))
        xavier_uniform_(self.conv.weight)
        self.output_layer = OutputLayer(args, Y, dicts, args.num_filter_maps)

    def forward(self, x, target, text_inputs):
        x = self.word_rep(x, target, text_inputs)
        x = x.transpose(1, 2)
        x = torch.tanh(self.conv(x).transpose(1, 2))
        y, loss, alpha, feat= self.output_layer(x, target, text_inputs)
        return y, loss, alpha, feat

    def freeze_net(self):
        for p in self.word_rep.embed.parameters():
            p.requires_grad = False


class MultiCNN(nn.Module):
    def __init__(self, args, Y, dicts):
        super(MultiCNN, self).__init__()

        self.word_rep = WordRep(args, Y, dicts)

        if args.filter_size.find(',') == -1:
            self.filter_num = 1
            filter_size = int(args.filter_size)
            self.conv = nn.Conv1d(self.word_rep.feature_size, args.num_filter_maps, kernel_size=filter_size,
                                  padding=int(floor(filter_size / 2)))
            xavier_uniform_(self.conv.weight)
        else:
            filter_sizes = args.filter_size.split(',')
            self.filter_num = len(filter_sizes)
            self.conv = nn.ModuleList()
            for filter_size in filter_sizes:
                filter_size = int(filter_size)
                tmp = nn.Conv1d(self.word_rep.feature_size, args.num_filter_maps, kernel_size=filter_size,
                                      padding=int(floor(filter_size / 2)))
                xavier_uniform_(tmp.weight)
                self.conv.add_module('conv-{}'.format(filter_size), tmp)

        self.output_layer = OutputLayer(args, Y, dicts, self.filter_num * args.num_filter_maps)


    def forward(self, x, target, text_inputs, x_mask=None):

        x = self.word_rep(x, target, text_inputs)
        x = x.transpose(1, 2)
        if self.filter_num == 1:
            x = torch.tanh(self.conv(x).transpose(1, 2))
        else:
            conv_result = []
            for tmp in self.conv:
                conv_result.append(torch.tanh(tmp(x).transpose(1, 2)))
            x = torch.cat(conv_result, dim=2)
        y, loss = self.output_layer(x, target, text_inputs)
        return y, loss

    def freeze_net(self):
        for p in self.word_rep.embed.parameters():
            p.requires_grad = False


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size, stride, use_res, dropout):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv1d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=int(floor(kernel_size / 2)), bias=False),
            nn.BatchNorm1d(outchannel),
            nn.Tanh(),
            nn.Conv1d(outchannel, outchannel, kernel_size=kernel_size, stride=1, padding=int(floor(kernel_size / 2)), bias=False),
            nn.BatchNorm1d(outchannel)
        )
        self.use_res = use_res
        if self.use_res:
            self.shortcut = nn.Sequential(
                        nn.Conv1d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm1d(outchannel)
                    )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        out = self.left(x)
        if self.use_res:
            out += self.shortcut(x)
        out = torch.tanh(out)
        out = self.dropout(out)
        return out

class ResCNN(nn.Module):

    def __init__(self, args, Y, dicts):
        super(ResCNN, self).__init__()

        self.word_rep = WordRep(args, Y, dicts)

        self.conv = nn.ModuleList()
        conv_dimension = self.word_rep.conv_dict[args.conv_layer]
        for idx in range(args.conv_layer):
            tmp = ResidualBlock(conv_dimension[idx], conv_dimension[idx + 1], int(args.filter_size), 1, True, args.dropout)
            self.conv.add_module('conv-{}'.format(idx), tmp)
        self.output_layer = OutputLayer(args, Y, dicts, args.num_filter_maps)

    def forward(self, x, target, text_inputs):
        x = self.word_rep(x, target, text_inputs)
        x = x.transpose(1, 2)
        for conv in self.conv:
            x = conv(x)
        x = x.transpose(1, 2)
        y, loss, alpha, feat= self.output_layer(x, target, text_inputs)
        return y, loss, alpha, feat

    def freeze_net(self):
        for p in self.word_rep.embed.parameters():
            p.requires_grad = False


class MultiResCNN(nn.Module):

    def __init__(self, args, Y, dicts):
        super(MultiResCNN, self).__init__()

        self.word_rep = WordRep(args, Y, dicts)
        self.conv = nn.ModuleList()
        filter_sizes = args.filter_size.split(',')
        self.filter_num = len(filter_sizes)
        for filter_size in filter_sizes:
            filter_size = int(filter_size)
            one_channel = nn.ModuleList()
            tmp = nn.Conv1d(self.word_rep.feature_size, self.word_rep.feature_size, kernel_size=filter_size,
                            padding=int(floor(filter_size / 2)))
            xavier_uniform_(tmp.weight)
            one_channel.add_module('baseconv', tmp)
            conv_dimension = self.word_rep.conv_dict[args.conv_layer]
            for idx in range(args.conv_layer):
                tmp = ResidualBlock(conv_dimension[idx], conv_dimension[idx + 1], filter_size, 1, True,
                                    args.dropout)
                one_channel.add_module('resconv-{}'.format(idx), tmp)
            self.conv.add_module('channel-{}'.format(filter_size), one_channel)
        self.output_layer = OutputLayer(args, Y, dicts, self.filter_num * args.num_filter_maps)

    def forward(self, x, target, text_inputs, x_mask=None):
        x = self.word_rep(x, target, text_inputs)   # [bs, len, dim_embed]
        x = x.transpose(1, 2)   #[bs, dim_embed, len]
        conv_result = []
        for conv in self.conv:
            tmp = x
            for idx, md in enumerate(conv):
                if idx == 0:
                    tmp = torch.tanh(md(tmp))
                else:
                    tmp = md(tmp)
            tmp = tmp.transpose(1, 2)
            conv_result.append(tmp)
        x = torch.cat(conv_result, dim=2)   # [bs, 25, 150
        y, loss = self.output_layer(x, target, text_inputs)
        return y, loss

    def freeze_net(self):
        for p in self.word_rep.embed.parameters():
            p.requires_grad = False


def pick_model(args, dicts):
    Y = len(dicts['ind2c'])
    if args.model == 'CNN':
        model = CNN(args, Y, dicts)
    elif args.model == 'MultiCNN':
        model = MultiCNN(args, Y, dicts)
    elif args.model == 'ResCNN':
        model = ResCNN(args, Y, dicts)
    elif args.model == 'MultiResCNN':
        model = MultiResCNN(args, Y, dicts)
    elif args.model == 'DCAN':
        model = DCAN(args, Y, dicts)
    else:
        raise RuntimeError("wrong model name")

    if args.test_model:
        sd = torch.load(args.test_model)
        model.load_state_dict(sd)
    if args.gpu >= 0:
        model.cuda(args.gpu)
    return model
