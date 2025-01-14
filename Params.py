import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Model Params')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight_decay')
    parser.add_argument('--cr', default=0.8, type=float, help='contrastive loss rate')
    parser.add_argument('--ir', default=1, type=float, help='infomax loss rate')
    parser.add_argument('--t', default=0.05, type=float, help='temperature')
    parser.add_argument('--batch', default=16, type=int, help='batch size')
    parser.add_argument('--epoch', default=1, type=int, help='number of epochs')
    parser.add_argument('--latdim', default=16, type=int, help='embedding size')
    parser.add_argument('--temporalRange', default=30, type=int, help='number of hops for temporal features')
    parser.add_argument('--cateNum', default=4, type=int, help='number of categories')
    parser.add_argument('--data', default='NYC', type=str, help='name of dataset')
    parser.add_argument('--kernelSize', default=3, type=int, help='size of kernel')
    parser.add_argument('--border', default=0.5, type=float, help='border line for pos and neg predictions')
    parser.add_argument('--hyperNum', default=128, type=int, help='number of hyper edges')
    parser.add_argument('--dropRateL', default=0.2, type=float, help='drop rate for local encoder')
    parser.add_argument('--dropRateG', default=0.1, type=float, help='drop rate for global encoder')
    parser.add_argument('--device', type=str, default='cpu', help='cuda device')
    parser.add_argument('--tstEpoch', default=1, type=int, help='number of epoch to test while training')
    parser.add_argument('--save', type=str, default='./Save/', help='save path')
    parser.add_argument('--checkpoint', type=str, default='./Save/NYC/', help='test path')
    return parser.parse_args()
args = parse_args()