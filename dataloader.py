import csv
import torch
import numpy as np
from collections import defaultdict
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from elmo import elmo


def load_vocab_dict(args, vocab_file):
    """
    Load vocabulary dictionary from file: vocab_file
    """
    vocab = set()
    with open(vocab_file, 'r') as vocabfile:
        for i, line in enumerate(vocabfile):
            line = line.rstrip()
            if line != '':
                vocab.add(line.strip())
    ind2w = {i + 1: w for i, w in enumerate(sorted(vocab))}
    w2ind = {w: i for i, w in ind2w.items()}
    return ind2w, w2ind


def load_full_codes(train_path, mimic2_dir, version='mimic3'):
    """
    Load full set of ICD codes
    """
    if version == 'mimic2':
        ind2c = defaultdict(str)
        codes = set()
        with open(mimic2_dir, 'r') as f:
            r = csv.reader(f)
            next(r) # skip header
            for row in r:
                codes.update(set(row[-1].split(';')))
        codes = set([c for c in codes if c != ''])
        ind2c = defaultdict(str, {i:c for i,c in enumerate(sorted(codes))})
    else:
        codes = set()
        for split in ['train', 'dev', 'test']:
            with open(train_path.replace('train', split), 'r') as f:
                lr = csv.reader(f)
                next(lr)
                for row in lr:
                    for code in row[3].split(';'):
                        codes.add(code)
        codes = set([c for c in codes if c != ''])
        ind2c = defaultdict(str, {i:c for i,c in enumerate(sorted(codes))})
    return ind2c


def load_lookups(args):
    """
    Load lookup dictionaries: index2word, word2index, index2code, code2index 
    """
    ind2w, w2ind = load_vocab_dict(args, args.vocab)
    if args.Y == 'full':
        ind2c = load_full_codes(args.data_path, '%s/proc_dsums.csv' % args.MIMIC_2_DIR, version=args.version)
    else:
        codes = set()
        with open("%s/TOP_%s_CODES.csv" % (args.MIMIC_3_DIR, str(args.Y)), 'r') as labelfile:
            lr = csv.reader(labelfile)
            for i, row in enumerate(lr):
                codes.add(row[0])
        ind2c = {i:c for i,c in enumerate(sorted(codes))}
    c2ind = {c:i for i,c in ind2c.items()}
    dicts = {'ind2w': ind2w, 'w2ind': w2ind, 'ind2c': ind2c, 'c2ind': c2ind}
    return dicts


def prepare_instance(dicts, filename, args, max_length):
    # filename: data/mimic[2/3]/[train/dev/test]_[50/full].csv, e.g., data/mimic3/train_50.csv
    ind2w, w2ind, ind2c, c2ind = dicts['ind2w'], dicts['w2ind'], dicts['ind2c'], dicts['c2ind']
    instances = []
    num_labels = len(dicts['ind2c'])
    with open(filename, 'r') as infile:
        r = csv.reader(infile)
        next(r)     # skip header
        for row in r:
            text = row[2]
            labels_idx = np.zeros(num_labels)
            labelled = False
            for l in row[3].split(';'):
                if l in c2ind.keys():
                    code = int(c2ind[l])
                    labels_idx[code] = 1
                    labelled = True
            if not labelled:
                continue
            tokens_ = text.split()
            tokens = []
            tokens_id = []
            for token in tokens_:
                if token == '[CLS]' or token == '[SEP]':
                    continue
                tokens.append(token)
                token_id = w2ind[token] if token in w2ind else len(w2ind) + 1
                tokens_id.append(token_id)
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
                tokens_id = tokens_id[:max_length]
            dict_instance = {'label': labels_idx, 'tokens': tokens, "tokens_id": tokens_id}
            instances.append(dict_instance)
    return instances


def prepare_instance_bert(dicts, filename, args, max_length):
    ind2w, w2ind, ind2c, c2ind = dicts['ind2w'], dicts['w2ind'], dicts['ind2c'], dicts['c2ind']
    instances = []
    num_labels = len(dicts['ind2c'])
    wp_tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    with open(filename, 'r') as infile:
        r = csv.reader(infile)
        next(r)
        for row in r:
            text = row[2]
            labels_idx = np.zeros(num_labels)
            labelled = False
            for l in row[3].split(';'):
                if l in c2ind.keys():
                    code = int(c2ind[l])
                    labels_idx[code] = 1
                    labelled = True
            if not labelled:
                continue
            tokens_ = text.split()
            tokens = []
            for token in tokens_:
                if token == '[CLS]' or token == '[SEP]':
                    continue
                wps = wp_tokenizer.tokenize(token)
                tokens.extend(wps)
            tokens_max_len = max_length-2 # for CLS SEP
            if len(tokens) > tokens_max_len:
                tokens = tokens[:tokens_max_len]
            tokens.insert(0, '[CLS]')
            tokens.append('[SEP]')
            tokens_id = wp_tokenizer.convert_tokens_to_ids(tokens)
            masks = [1] * len(tokens)
            segments = [0] * len(tokens)
            dict_instance = {'label':labels_idx, 'tokens':tokens, "tokens_id":tokens_id, 
                             "segments":segments, "masks":masks}
            instances.append(dict_instance)
    return instances


class MyDataset(Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]


def pad_sequence(x, max_len, type=np.int):
    padded_x = np.zeros((len(x), max_len), dtype=type)
    for i, row in enumerate(x):
        padded_x[i][:len(row)] = row
    return padded_x


def my_collate(x):
    words = [x_['tokens_id'] for x_ in x]
    seq_len = [len(w) for w in words]
    masks = [[1]*len(w) for w in words]
    max_seq_len = max(seq_len) # TODO 
    # max_seq_len = args.MAX_LENGTH # TODO for capsule network

    inputs_idx = torch.LongTensor(pad_sequence(words, max_seq_len))
    inputs_mask = torch.LongTensor(pad_sequence(masks, max_seq_len))
    labels = torch.FloatTensor([x_['label'] for x_ in x])
    inputs_text = [x_['tokens'] for x_ in x]
    inputs_text = elmo.batch_to_ids(inputs_text)
    return inputs_idx, labels, inputs_text, inputs_mask


def my_collate_bert(x):
    words = [x_['tokens_id'] for x_ in x]
    segments = [x_['segments'] for x_ in x]
    masks = [x_['masks'] for x_ in x]
    seq_len = [len(w) for w in words]
    max_seq_len = max(seq_len)    # max of batch

    inputs_idx = torch.LongTensor(pad_sequence(words, max_seq_len))
    segments = torch.LongTensor(pad_sequence(segments, max_seq_len))
    masks = torch.LongTensor(pad_sequence(masks, max_seq_len))
    labels = torch.FloatTensor([x_['label'] for x_ in x])
    return inputs_idx, segments, masks, labels
    