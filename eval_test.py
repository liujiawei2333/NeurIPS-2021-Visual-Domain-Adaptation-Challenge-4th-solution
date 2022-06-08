import yaml
import easydict
import os
import torch
from utils.utils import log_set, load_model2
from utils.defaults import get_dataloaders, get_models
import argparse
from utils.amsoftmax_withm import AMSoftmax
from utils.loss_functions import AngularPenaltySMLoss
import warnings
from eval_pseudo import test2
warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser(description='Pytorch OVANet',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--config', type=str, default='./configs/image_to_objectnet_imagenet_c_r_o.yaml',
                    help='/path/to/config/file')
parser.add_argument('--source_data', type=str,
                    default='/home/storage/storage50/disk3/visda/ILSVRC/Data/CLS-LOC/train/',
                    help='path to source list')
parser.add_argument('--target_data', type=str,
                    default='./test/test_data_list.txt',
                    help='path to target list')
parser.add_argument('--network', type=str,
                    default='adveffi-b4',
                    help='network name')
parser.add_argument("--gpu_devices", type=int, nargs='+',
                    default=None, help="")
parser.add_argument("--save_model",
                    default=True, action='store_true')
parser.add_argument("--save_path", type=str,
                    default="record/B4",
                    help='/path/to/save/model')
parser.add_argument("--mode", type=str,
                    default="eval",
                    help='train or eval')
parser.add_argument("--softmax_mode", type=str,
                    default="normal",
                    help='normal or amsoftmax or arcface')
parser.add_argument('--advprop',
                    default=True,
                    action='store_true')
parser.add_argument('--change_size',
                    default=True,
                    action='store_true')
parser.add_argument('--pseudo_data', type=str,
                    default='./test/test_data_list.txt',
                    help='path to pesudo list')
parser.add_argument('--multi', type=float,
                    default=0.05,
                    help='weight factor for adaptation')
parser.add_argument('--exp_name', type=str,
                    default='ovanet',
                    help='/path/to/config/file')
parser.add_argument('--pseudo_times', type=int,
                    default=6,
                    help='pseduo times')

args = parser.parse_args()

config_file = args.config
conf = yaml.load(open(config_file),Loader=yaml.FullLoader)
save_config = yaml.load(open(config_file),Loader=yaml.FullLoader)
conf = easydict.EasyDict(conf)

if args.gpu_devices == None:
    gpu_devices = '0'
else:
    gpu_devices = ','.join([str(id) for id in args.gpu_devices])
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
args.cuda = torch.cuda.is_available()

torch.backends.cudnn.benchmark = True

if args.change_size:
    if "b0" in args.network:
        scale_size = 256
        crop_size = 224
    elif "b1" in args.network:
        scale_size = 256
        crop_size = 240
    elif "b2" in args.network:
        scale_size = 288
        crop_size = 260
    elif "b3" in args.network:
        scale_size = 332
        crop_size = 300
    elif "b4" in args.network:
        scale_size = 412
        crop_size = 380
    elif "b5" in args.network:
        scale_size = 488
        crop_size = 456
    elif "b6" in args.network:
        scale_size = 560
        crop_size = 528
    elif "b7" in args.network:
        scale_size = 732
        crop_size = 700
else:
    scale_size = 256
    crop_size = 224

target_data = args.target_data
evaluation_data = args.target_data
network = args.network
use_gpu = torch.cuda.is_available()
n_share = conf.data.dataset.n_share
n_source_private = conf.data.dataset.n_source_private
n_total = conf.data.dataset.n_total
opens = n_total - n_share - n_source_private > 0
num_class = n_share + n_source_private
script_name = os.path.basename(__file__)

inputs = vars(args)
inputs["evaluation_data"] = evaluation_data
inputs["conf"] = conf
inputs["script_name"] = script_name
inputs["num_class"] = num_class
inputs["config_file"] = config_file
inputs["scale_size"] = scale_size
inputs["crop_size"] = crop_size

source_loader, target_loader, \
test_loader, target_folder = get_dataloaders(inputs)

logname = log_set(inputs)

G, C1, C2, opt_g, opt_c, \
param_lr_g, param_lr_c = get_models(inputs)
ndata = target_folder.__len__()

target_loader.dataset.labels[target_loader.dataset.labels>1000] = 1000
test_loader.dataset.labels[test_loader.dataset.labels>1000] = 1000

if args.pseudo_times == 0:
    load_path = '%s/pre/model/model_pre.pth' % (args.save_path)
elif args.pseudo_times == 1:
    load_path = '%s/pre/model/model_pre.pth' % (args.save_path)
elif args.pseudo_times > 1:
    load_path = '%s/pseudo_times%d/model/model.pth' % (args.save_path,
                        args.pseudo_times)
G, C1, C2 = load_model2(G, C1, C2, load_path)
G.cuda()
C1.cuda()
C2.cuda()

if args.softmax_mode == "normal":
    print("use normal softmax")
    softmax_layer = None
elif args.softmax_mode == "amsoftmax":
    softmax_layer = AMSoftmax().cuda()
    print("use am-softmax")
elif args.softmax_mode == "arcface":
    softmax_layer = AngularPenaltySMLoss(s=32.0,m=0.2,in_features=1000,out_features=1000,loss_type='arcface').cuda()
    print("use arcface")

max_acc = 0.00001
step = "test"
test2(step, test_loader, logname, n_share,G,[C1, C2], 
softmax_layer, args.softmax_mode,max_acc, open=opens,
mode=args.mode,save_path=args.save_path,pseudo_times=args.pseudo_times)
