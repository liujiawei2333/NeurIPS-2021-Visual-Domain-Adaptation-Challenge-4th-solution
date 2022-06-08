'''training'''
import yaml
import easydict
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from apex import amp
from utils.utils import log_set, load_model2
from utils.loss import ova_loss, open_entropy
from utils.lr_schedule import inv_lr_scheduler
from utils.defaults import get_dataloaders, get_models
from eval_pseudo import test,test2
import argparse
from tensorboardX import SummaryWriter
from utils.amsoftmax_withm import AMSoftmax
from utils.loss_functions import AngularPenaltySMLoss
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser(description='Pytorch OVANet',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--config', type=str, default='./configs/image_to_objectnet_imagenet_c_r_o.yaml',
                    help='/path/to/config/file')
parser.add_argument('--source_data', type=str,
                    default='/home/storage/storage50/disk3/visda/ILSVRC/Data/CLS-LOC/train/',
                    help='path to source list')
parser.add_argument('--target_data', type=str,#pseudo label of target data or test data
                    default='record/B4/pre/model/adapt_pred.txt',
                    help='path to pesudo list')
parser.add_argument('--log-interval', type=int,
                    default=100,
                    help='how many batches before logging training status')
parser.add_argument('--exp_name', type=str,
                    default='ovanet',
                    help='/path/to/config/file')
parser.add_argument('--network', type=str,
                    default='adveffi-b4',
                    help='network name')
parser.add_argument("--gpu_devices", type=int, nargs='+',
                    default=None, help="")
parser.add_argument("--no_adapt",
                    default=False, action='store_true')
parser.add_argument("--save_model",
                    default=True, action='store_true')
parser.add_argument("--save_path", type=str,
                    default="record/B4",
                    help='/path/to/save/model')
parser.add_argument('--multi', type=float,
                    default=0.05,
                    help='weight factor for adaptation')
parser.add_argument("--mode", type=str,
                    default="train",
                    help='train or eval')
parser.add_argument("--softmax_mode", type=str,
                    default="normal",
                    help='normal or amsoftmax or arcface')
parser.add_argument('--resume',
                    default=True,
                    action='store_true')
parser.add_argument('--advprop',
                    default=True,
                    action='store_true')
parser.add_argument('--change_size',
                    default=True,
                    action='store_true')
parser.add_argument('--pseudo_times', type=int,
                    default=1,
                    help='pseduo times')
parser.add_argument('--t1', type=int,
                    default=200,
                    help='when setp bigger than t1,pseudo is used')
parser.add_argument('--t2', type=int,
                    default=1000,
                    help='when setp bigger than t2,pseudo weight is not change')
parser.add_argument('--af', type=float,
                    default=0.6,
                    help='weight factor for pseudo')

args = parser.parse_args()

def unlabeled_weight(step):
        alpha = 0.0
        if step > args.t1:
            alpha = (step - args.t1) / (args.t2 - args.t1) * args.af
            if step > args.t2:
                alpha = args.af
        return alpha

config_file = args.config
conf = yaml.load(open(config_file),Loader=yaml.FullLoader)
save_config = yaml.load(open(config_file),Loader=yaml.FullLoader)
conf = easydict.EasyDict(conf)

if not os.path.exists('%s/pseudo_times%d/tensorboard_log' % (args.save_path,args.pseudo_times)):
    os.makedirs('%s/pseudo_times%d/tensorboard_log' % (args.save_path,args.pseudo_times))
if not os.path.exists('%s/pseudo_times%d/model' % (args.save_path,args.pseudo_times)):
    os.makedirs('%s/pseudo_times%d/model' % (args.save_path,args.pseudo_times))

writer = SummaryWriter(log_dir="%s/pseudo_times%d/tensorboard_log" % (args.save_path,args.pseudo_times))

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

source_data = args.source_data
target_data = args.target_data
evaluation_data = args.target_data
network = args.network
use_gpu = torch.cuda.is_available()
n_share = conf.data.dataset.n_share
n_source_private = conf.data.dataset.n_source_private
n_total = conf.data.dataset.n_total
open = n_total - n_share - n_source_private > 0
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

if args.pseudo_times == 1:
    load_path = '%s/pre/model/model_pre.pth' % (args.save_path)
elif args.pseudo_times > 1:
    load_path = '%s/pseudo_times%d/model/model.pth' % (args.save_path,
                        args.pseudo_times-1)
G, C1, C2 = load_model2(G, C1, C2, load_path)
G.cuda()
C1.cuda()
C2.cuda()

param_G = sum(p.numel() for p in G.parameters())/1000000.0
param_C1 = sum(p.numel() for p in C1.parameters())/1000000.0
param_C2 = sum(p.numel() for p in C2.parameters())/1000000.0
print("params_G:%.4f" % param_G)
print("params_C1:%.4f" % param_C1)
print("params_C2:%.4f" % param_C2)

def train():
    global param_lr_g
    global param_lr_c
    max_acc = 0.00001
    criterion = nn.CrossEntropyLoss().cuda()
    print('train start!')
    data_iter_s = iter(source_loader)
    data_iter_t = iter(target_loader)
    len_train_source = len(source_loader)
    len_train_target = len(target_loader)
    if args.softmax_mode == "normal":
        print("use normal softmax")
        softmax_layer = None
    elif args.softmax_mode == "amsoftmax":
        softmax_layer = AMSoftmax().cuda()
        print("use am-softmax")
    elif args.softmax_mode == "arcface":
        softmax_layer = AngularPenaltySMLoss(in_features=1000,out_features=1000,loss_type='arcface').cuda()
        print("use arcface")

    start_step = 0

    for step in range(start_step,conf.train.min_step + 1):
        G.train()
        C1.train()
        C2.train()
        if step % len_train_target == 0:
            data_iter_t = iter(target_loader)
        if step % len_train_source == 0:
            data_iter_s = iter(source_loader)
        data_t = next(data_iter_t)
        data_s = next(data_iter_s)
        inv_lr_scheduler(param_lr_g, opt_g, step,
                         init_lr=conf.train.g_lr,
                         max_iter=conf.train.min_step)
        inv_lr_scheduler(param_lr_c, opt_c, step,
                         init_lr=conf.train.c_lr,
                         max_iter=conf.train.min_step)
        img_s = data_s[0]
        label_s = data_s[1]
        img_t = data_t[0]
        label_t = data_t[1]
        img_s, label_s = Variable(img_s.cuda()), \
                         Variable(label_s.cuda())
        img_t, label_t = Variable(img_t.cuda()), \
                         Variable(label_t.cuda())

        opt_g.zero_grad()
        opt_c.zero_grad()

        #Source loss calculation
        feat = G(img_s)
        out_s = C1(feat)
        if args.softmax_mode =="arcface":
            loss_s = softmax_layer(out_s, label_s, "train")
        elif args.softmax_mode == "amsoftmax":
            out_s = softmax_layer(out_s,label_s, "train")
            loss_s = criterion(out_s, label_s)
        elif args.softmax_mode == "normal":
            loss_s = criterion(out_s, label_s)

        #target pseudo loss calculation
        feat_t = G(img_t)
        out_t = C1(feat_t)
        if args.softmax_mode =="arcface":
            loss_t = softmax_layer(out_t, label_t, "train")
        elif args.softmax_mode == "amsoftmax":
            out_t = softmax_layer(out_t,label_t, "train")
            loss_t = criterion(out_t, label_t)
        elif args.softmax_mode == "normal":
            loss_t = criterion(out_t, label_t)

        loss_cls = loss_s + unlabeled_weight(step) * loss_t

        out_open = C2(feat)
        
        ## open set loss for source
        out_open = out_open.view(out_s.size(0), 2, -1)

        open_loss_pos, open_loss_neg = ova_loss(out_open, label_s)

        loss_open = 0.5 * (open_loss_pos + open_loss_neg)
        ## open set loss for target
        all = loss_cls + loss_open
        log_string = 'Train {}/{} \t ' \
                    'Loss Source: {:.4f} ' \
                    'Loss Pseudo: {:.4f} ' \
                    'Loss Open: {:.4f} ' \
                    'Loss Open Source Positive: {:.4f} ' \
                    'Loss Open Source Negative: {:.4f} '
        log_values = [step, conf.train.min_step,
                      loss_s.item(), loss_t.item(),  loss_open.item(),
                      open_loss_pos.item(), open_loss_neg.item()]
        if not args.no_adapt:
            feat_t = G(img_t)
            out_open_t = C2(feat_t)
            out_open_t = out_open_t.view(img_t.size(0), 2, -1)
            ent_open = open_entropy(out_open_t)
            all += args.multi * ent_open
            log_values.append(ent_open.item())
            log_string += "Loss Open Target: {:.6f}"

        with amp.scale_loss(all, [opt_g, opt_c]) as scaled_loss:
            scaled_loss.backward()

        opt_g.step()
        opt_c.step()
        opt_g.zero_grad()
        opt_c.zero_grad()

        if step % conf.train.log_interval == 0:
            print(log_string.format(*log_values))

            test(step, test_loader, logname,
            n_share,G,[C1, C2], softmax_layer, args.softmax_mode,max_acc, open=open,
            mode=args.mode,save_path=args.save_path,pseudo_times=args.pseudo_times)

            writer.add_scalars('loss_s', {'train': loss_s,'train_pseudo':unlabeled_weight(step) * loss_t}, step)
            writer.add_scalar('loss_open',loss_open, step)
            writer.add_scalar('open_loss_pos',open_loss_pos, step)
            writer.add_scalar('open_loss_neg',open_loss_neg, step)
            writer.add_scalar('ent_open',ent_open, step)
            writer.add_scalar('all', all,step)

            G.train()
            C1.train()

train()
