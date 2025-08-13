import argparse
# Training settings
parser = argparse.ArgumentParser(description='BRAIN implementation')

#########################
#### data parameters ####
#########################
parser.add_argument("--data_name", type=str, default="iapr", # flickr coco nus iapr
                    help="data name")
parser.add_argument('--root_dir', type=str, default='./')
parser.add_argument('--log_name', type=str, default='BRAIN')
parser.add_argument('--pretrain', action='store_true', default=False)
parser.add_argument('--pretrain_dir', type=str, default='EN')
parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg11', help='model architecture: ' + ' | '.join(['ResNet', 'VGG']) + ' (default: vgg11)')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--wd', type=float, default=1e-6)
parser.add_argument('--train_batch_size', type=int, default=256)
parser.add_argument('--eval_batch_size', type=int, default=256)
parser.add_argument('--max_epochs', type=int, default=100)
parser.add_argument('--log_interval', type=int, default=40)
parser.add_argument('--num_workers', type=int, default=5)
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--num_hiden_layers', default=[3, 2, 4, 2, 2, 1], nargs='+', help='<Required> Number of hiden lyaers')
parser.add_argument('--ls', type=str, default='linear', help='lr scheduler')
parser.add_argument('--bit', type=int, default=32, help='output shape')
parser.add_argument('--alpha', type=float, default=.9)
parser.add_argument('--threshold', type=float, default=0.65)
parser.add_argument('--margin', type=float, default=0.61)
parser.add_argument('--warmup_epoch', type=int, default=0)
parser.add_argument('--margin1', type=float, default=0.2)
parser.add_argument('--margin2', type=float, default=0.2)
parser.add_argument('--margin3', type=float, default=0.2)
parser.add_argument('--time_steps', default=[3, 3, 2, 2], nargs='+', help='<Required> Number of time steps')
parser.add_argument('--time_enc1', type=float, default=0.1)
parser.add_argument('--time_enc2', type=float, default=0.1)
parser.add_argument('--en1', type=int, default=256)
parser.add_argument('--en2', type=int, default=256)
parser.add_argument('--enk', type=int, default=16)
parser.add_argument('--dim1', type=float, default=1)
parser.add_argument('--dim2', type=float, default=1)
parser.add_argument('--using_UIC', action='store_true', default=False)
parser.add_argument('--using_EN', action='store_true', default=False)

args = parser.parse_args()
args.num_hiden_layers = [int(i) for i in args.num_hiden_layers]
print(args)
