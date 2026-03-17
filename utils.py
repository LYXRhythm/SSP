import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_training_args():
    parser = argparse.ArgumentParser(description='SEHA')
    parser.add_argument("--dataset", type=str, default="wiki", help="Dataset to use (wiki, nus-wide, INRIA-Websearch, xmedianet)")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--partial_ratio", type=float, default=0.4)
    parser.add_argument("--MAX_EPOCH", type=int, default=180)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--output_dim", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)   # wiki: 1e-4, nus-wide: 1e-4, INRIA-Websearch: 1e-4, xmedianet: 2e-5
    parser.add_argument("--lamda", type=float, default=0.1) 
    parser.add_argument("--ema_decay", type=float, default=0.95)
    parser.add_argument('--linear', type=str2bool, default=True)
    parser.add_argument("--GPU", type=int, default=0)

    args = parser.parse_args()

    return args

# 0.01:{0.544,0.507,0.526}
# 0.1: {0.547,0.509,0.528} 0.2:{0.540,0.509,0.524} 0.3:{0.548,0.512,0.530}
# 1.0: {0.542,0.507,0.524}
# 10.0:{0.532,0.491,0.511}
