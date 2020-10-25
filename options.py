import argparse
import sys

def args_parser():
    parser = argparse.ArgumentParser()
    path_root = '/your/project/path'
    parser.add_argument('--MODEL_DIR', type=str, default='{}/saved/models'.format(path_root))
    parser.add_argument('--DATA_DIR', type=str, default='{}/data'.format(path_root))
    parser.add_argument('--MIMIC_3_DIR', type=str, default='{}/data/mimic3'.format(path_root))

    parser.add_argument("--data_path", type=str, default='{}/data/mimic3/train_50.csv'.format(path_root))
    parser.add_argument("--vocab", type=str, default='{}/data/mimic3/vocab.csv'.format(path_root))
    parser.add_argument("--Y", type=str, default='50', choices=['full', '50'])
    parser.add_argument("--version", type=str, default='mimic3')
    parser.add_argument("--MAX_LENGTH", type=int, default=500)
    parser.add_argument("--resume", type=str, help="resume form saved model")
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--print_every", type=int, default=50)
    
    # model
    parser.add_argument("--model", type=str, default='DCAN')
    parser.add_argument("--embed_file", type=str, default='{}/data/mimic3/processed_full.embed'.format(path_root))
    parser.add_argument("--test_model", type=str, default=None)
    parser.add_argument("--use_ext_emb", action="store_const", const=True, default=False)
    parser.add_argument('--label_smoothing', action='store_true', help="label smoothing")
    parser.add_argument('--alpha', type=float, default=0.1, help='label smoothing')
    parser.add_argument('--batchnorm', action='store_true', help="batch normalization")
    parser.add_argument('--kernel_size', type=int, default=2, help='single kernel')
    parser.add_argument('--nhid', type=int, default=600, help='number of hidden units per layer')
    parser.add_argument('--nproj', type=int, default=300, help='linear projection dimension')
    parser.add_argument('--levels', type=int, default=1, help='number of levels')

    # training
    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--criterion", type=str, default='f1_macro', choices=['auc_macro', 'auc_micro', 'prec_at_8', 'prec_at_15', 'f1_macro', 'f1_micro', 'prec_at_5','loss_dev'])
    parser.add_argument("--gpu", type=int, default=0, help='-1 if not use gpu, >=0 if use gpu')
    parser.add_argument("--tune_wordemb", action="store_const", const=True, default=False)
    parser.add_argument('--random_seed', type=int, default=1, help='0 if randomly initialize the model, other if fix the seed')

    args = parser.parse_args()
    command = ' '.join(['python'] + sys.argv)
    args.command = command
    return args