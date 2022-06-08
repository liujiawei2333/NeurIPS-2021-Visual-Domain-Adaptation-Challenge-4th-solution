import numpy as np
import torch
import torch.nn.functional as F
import logging
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
from utils.amsoftmax_withm import AMSoftmax
from utils.utils import save_model

'''test of test set'''
def test2(step, dataset_test, name, n_share, G, Cs, softmax_layer, softmax_mode, max_acc,
         open=False, mode="train", entropy=False, thr=None, save_path="record/",pseudo_times=1):
    G.eval()
    for c in Cs:
        c.eval()
    ## Known Score Calculation.
    size = 0
    per_class_num = np.zeros((n_share + 1))
    class_list = np.unique(dataset_test.dataset.labels)
    open_class = 1000

    for batch_idx, data in enumerate(dataset_test):
        with torch.no_grad():
            img_t, label_t = data[0].cuda(), data[1].cuda()
            feat = G(img_t)
            out_t = Cs[0](feat)
            if softmax_mode == "amsoftmax":
                out_t = softmax_layer(out_t,label_t,"eval")
            elif softmax_mode == "arcface":
                out_t = softmax_layer(out_t,label_t,"eval")

            pred = out_t.data.max(1)[1]
            out_t_softmax = F.softmax(out_t)
            if entropy:
                pred_unk = -torch.sum(out_t_softmax * torch.log(out_t_softmax), 1)
            else:
                out_open = Cs[1](feat)
                out_open_softmax = F.softmax(out_open.view(out_t_softmax.size(0), 2, -1),1)
                tmp_range = torch.range(0, out_t_softmax.size(0)-1).long().cuda()
                pred_unk = out_open_softmax[tmp_range, 0, pred]
            pred = pred.cpu().numpy()
            if batch_idx == 0:
                prediction = pred
            else:
                prediction = np.r_[prediction, pred]

            if open:
                label_t_numpy = label_t.data.cpu().numpy()
                if batch_idx == 0:
                    label_all = label_t_numpy
                    anomaly_score = pred_unk.data.cpu().numpy()
                else:
                    anomaly_score = np.r_[anomaly_score, pred_unk.data.cpu().numpy()]
                    label_all = np.r_[label_all, label_t_numpy]
    print('save model!')
    save_model(G, Cs[0], Cs[1], "%s/pseudo_times%d/model/model.pth" % (save_path,pseudo_times))
    print("save adapt_pred.txt!")
    gen_submission_files(('%s/pseudo_times%d/model/adapt_pred.txt' % (save_path,pseudo_times)),dataset_test.dataset.imgs, prediction, anomaly_score)

'''test of target set'''
def test(step, dataset_test, name, n_share, G, Cs, softmax_layer, softmax_mode, max_acc,
         open=False, mode="train", entropy=False, thr=None, save_path="record/",pseudo_times=1):
    global label_all
    G.eval()
    for c in Cs:
        c.eval()
    ## Known Score Calculation.
    correct = 0
    correct_close = 0
    size = 0
    per_class_num = np.zeros((n_share + 1))
    per_class_correct = np.zeros((n_share + 1)).astype(np.float32)
    class_list = np.unique(dataset_test.dataset.labels)
    open_class = 1000

    for batch_idx, data in enumerate(dataset_test):
        with torch.no_grad():
            img_t, label_t = data[0].cuda(), data[1].cuda()
            feat = G(img_t)
            out_t = Cs[0](feat)
            if softmax_mode == "amsoftmax":
                out_t = softmax_layer(out_t,label_t,"eval")
            elif softmax_mode == "arcface":
                out_t = softmax_layer(out_t,label_t,"eval")

            pred = out_t.data.max(1)[1]
            correct_close += pred.eq(label_t.data).cpu().sum()
            out_t_softmax = F.softmax(out_t)
            if entropy:
                pred_unk = -torch.sum(out_t_softmax * torch.log(out_t_softmax), 1)
            else:
                out_open = Cs[1](feat)
                out_open_softmax = F.softmax(out_open.view(out_t_softmax.size(0), 2, -1),1)
                tmp_range = torch.range(0, out_t_softmax.size(0)-1).long().cuda()
                pred_unk = out_open_softmax[tmp_range, 0, pred]
            correct += pred.eq(label_t.data).cpu().sum()
            pred = pred.cpu().numpy()
            k = label_t.data.size()[0]
            for i, t in enumerate(class_list):
                t_ind = np.where(label_t.data.cpu().numpy() == t)
                correct_ind = np.where(pred[t_ind[0]] == t)
                per_class_correct[i] += float(len(correct_ind[0]))
                per_class_num[i] += float(len(t_ind[0]))
            size += k
            if batch_idx == 0:
                prediction = pred
            else:
                prediction = np.r_[prediction, pred]

            if open:
                label_t_numpy = label_t.data.cpu().numpy()
                if batch_idx == 0:
                    label_all = label_t_numpy
                    anomaly_score = pred_unk.data.cpu().numpy()
                else:
                    anomaly_score = np.r_[anomaly_score, pred_unk.data.cpu().numpy()]
                    label_all = np.r_[label_all, label_t_numpy]

    if open:
        Y_test = label_binarize(label_all, classes=[i for i in class_list])
        roc = roc_auc_score(Y_test[:, -1], anomaly_score)
    else:
        roc = 0.0

    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=name, format="%(message)s")
    logger.setLevel(logging.INFO)
    close_count = float(per_class_num[:len(class_list) - 1].sum())
    acc_close_all = 100. *float(correct_close) / close_count
    output = ["step %s"%step,
              "acc close all %s" % float(acc_close_all),
              "roc %s"% float(roc)]
    logger.info(output)
    if mode == "eval":
        for i in range(len(dataset_test.dataset.imgs)):
            dataset_test.dataset.imgs[i] = dataset_test.dataset.imgs[i][36:]
    if acc_close_all > max_acc:
        max_acc = acc_close_all
        print('save model!')
        save_model(G, Cs[0], Cs[1], "%s/pseudo_times%d/model/model.pth" % (save_path,pseudo_times))
        print("acc:%.4f,auc:%.4f" % (acc_close_all,100. * roc))
        print("save adapt_pred.txt!")
        gen_submission_files(('%s/pseudo_times%d/model/adapt_pred.txt' % (save_path,pseudo_times)),dataset_test.dataset.imgs, prediction, anomaly_score)
    return acc_close_all, roc, out_t, out_open, label_t, max_acc

def test_pretrained(step, dataset_test, name, n_share, G,
         open=False, entropy=False, mode="eval", thr=None, prob=False, logit=False):
    G.eval()
    ## Known Score Calculation.
    correct = 0
    correct_close = 0
    size = 0
    per_class_num = np.zeros((n_share + 1))
    per_class_correct = np.zeros((n_share + 1)).astype(np.float32)
    class_list = np.unique(dataset_test.dataset.labels)

    for batch_idx, data in enumerate(dataset_test):
        with torch.no_grad():
            img_t, label_t = data[0].cuda(), data[1].cuda()
            out_t = G(img_t)
            pred = out_t.data.max(1)[1]
            correct_close += pred.eq(label_t.data).cpu().sum()
            logit_t = out_t
            out_t = F.softmax(out_t)

            if entropy:
                pred_unk = -torch.sum(out_t * torch.log(out_t), 1)

            elif prob:
                pred_unk = -torch.max(out_t, dim=-1)[0]

            else:
                pred_unk = -torch.max(logit_t, dim=-1)[0]

            correct += pred.eq(label_t.data).cpu().sum()
            pred = pred.cpu().numpy()
            k = label_t.data.size()[0]
            for i, t in enumerate(class_list):
                t_ind = np.where(label_t.data.cpu().numpy() == t)
                correct_ind = np.where(pred[t_ind[0]] == t)
                per_class_correct[i] += float(len(correct_ind[0]))
                per_class_num[i] += float(len(t_ind[0]))
            size += k

            if batch_idx == 0:
                prediction = pred
            else:
                prediction = np.r_[prediction, pred]

            if open:
                label_t = label_t.data.cpu().numpy()
                if batch_idx == 0:
                    label_all = label_t
                    anomaly_score = pred_unk.data.cpu().numpy()
                else:
                    anomaly_score = np.r_[anomaly_score, pred_unk.data.cpu().numpy()]
                    label_all = np.r_[label_all, label_t]
    if open:
        Y_test = label_binarize(label_all, classes=[i for i in class_list])
        roc = roc_auc_score(Y_test[:, -1], anomaly_score)


    else:
        roc = 0.0


    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=name, format="%(message)s")
    logger.setLevel(logging.INFO)
    close_count = float(per_class_num[:len(class_list) - 1].sum())
    acc_close_all = 100. *float(correct_close) / close_count
    output = ["step %s"%step,
              "acc close all %s" % float(acc_close_all),
              "roc %s"% float(roc)]
    logger.info(output)
    dataset_test_wanted = []
    for i in range(len(dataset_test.dataset.imgs)):
        dataset_test.dataset.imgs[i] = dataset_test.dataset.imgs[i][36:]
    gen_submission_files('./source_only_pred.txt', dataset_test.dataset.imgs, prediction, anomaly_score)

    return acc_close_all, roc

def gen_submission_files(outfile, image_names, prediction, anomaly_score):

    f = open(outfile, 'w')

    for img, pred, score in zip(image_names, prediction, anomaly_score):
        f.write('{} {} {}\n'.format(img, pred, score))

    f.close()

