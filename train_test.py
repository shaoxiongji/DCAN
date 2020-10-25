import torch
import numpy as np
from utils import all_metrics, print_metrics
import json
import pickle


def train(args, model, optimizer, epoch, gpu, data_loader):
    print("EPOCH %d" % epoch)
    device = torch.device('cuda:{}'.format(args.gpu)) if args.gpu != -1 else torch.device('cpu')
    losses = []
    model.train()
    # loader
    data_iter = iter(data_loader)
    num_iter = len(data_loader)
    for i in range(num_iter):
        inputs_id, labels, text_inputs, inputs_mask = next(data_iter)
        inputs_id, labels = inputs_id.to(device), labels.to(device)
        output, loss = model(inputs_id, labels, None)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if i % args.print_every == 0:
            print("Train epoch: {:>2d} [batch #{:>4d}, max_seq_len {:>4d}]\tLoss: {:.6f}".format(epoch, i, inputs_id.size()[1], loss.item()))
    return losses


def test(args, model, data_path, fold, gpu, dicts, data_loader):
    filename = data_path.replace('train', fold)
    device = torch.device('cuda:{}'.format(args.gpu)) if args.gpu != -1 else torch.device('cpu')
    print('file for evaluation: %s' % filename)
    num_labels = len(dicts['ind2c'])
    y, yhat, yhat_raw, hids, losses = [], [], [], [], []

    model.eval()
    data_iter = iter(data_loader)
    num_iter = len(data_loader)

    for i in range(num_iter):
        with torch.no_grad():
            inputs_id, labels, text_inputs, inputs_mask = next(data_iter)
            inputs_id, labels = inputs_id.to(device), labels.to(device)
            output, loss = model(inputs_id, labels, None)
            output = torch.sigmoid(output)
            output = output.data.cpu().numpy()
            losses.append(loss.item())
            target_data = labels.data.cpu().numpy()
            yhat_raw.append(output)
            output = np.round(output)
            y.append(target_data)
            yhat.append(output)

    y = np.concatenate(y, axis=0)
    yhat = np.concatenate(yhat, axis=0)
    yhat_raw = np.concatenate(yhat_raw, axis=0)

    k = 5 if num_labels == 50 else [8,15]
    metrics = all_metrics(yhat, y, k=k, yhat_raw=yhat_raw)
    print_metrics(metrics)
    metrics['loss_%s' % fold] = np.mean(losses)
    return metrics