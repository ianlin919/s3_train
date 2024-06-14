import pickle
import torch
import math
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from PIL import Image
import matplotlib.pyplot as plt
import os
import pydicom


def createTrainHistory(keywords):
    history = {"train": {}, "valid": {}}
    for words in keywords:
        history["train"][words] = list()
        history["valid"][words] = list()
    return history

def loadTxt(filename):
    f = open(filename)
    context = list()
    for line in f:
        context.append(line.replace("\n", ""))
    return context


def saveDict(filename, data):
    file = open(filename, 'wb')
    pickle.dump(data, file)
    file.close()


def loadDict(fileName):
    with open(fileName, 'rb') as handle:
        data = pickle.load(handle)
    return data


def saveTxt(filenamesList, saveName):
    fp = open(saveName, "a")
    for name in filenamesList:
        fp.write(name + "\n")
    fp.close()


def exp_rampup(rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""

    def warpper(epoch):
        if epoch < rampup_length:
            epoch = np.clip(epoch, 0.0, rampup_length)
            phase = 1.0 - epoch / rampup_length
            return float(np.exp(-5.0 * phase * phase))
        else:
            return 1.0

    return warpper


def decode_inputs(inputs):
    imgs_s, imgs_t, labels = inputs
    batch_size = labels.shape[0]
    imgs_s_s = list()
    imgs_t_s = list()
    label_s = list()
    imgs_s_u = list()
    imgs_t_u = list()

    for batch_idx in range(batch_size):
        if torch.sum(labels[batch_idx]).item() != 0:
            label_s.append(labels[batch_idx].unsqueeze(0))
            imgs_s_s.append(imgs_s[batch_idx].unsqueeze(0))
            imgs_t_s.append(imgs_t[batch_idx].unsqueeze(0))
        else:
            imgs_s_u.append(imgs_s[batch_idx].unsqueeze(0))
            imgs_t_u.append(imgs_t[batch_idx].unsqueeze(0))

    if len(imgs_s_u) == 0 and len(imgs_s_u) == 0:

        return (torch.cat([img for img in imgs_s_s], 0),
                torch.cat([img for img in imgs_t_s], 0),
                torch.cat([label for label in label_s], 0))

    elif len(imgs_s_s) == 0 and len(imgs_s_s) == 0:
        return (torch.cat([img for img in imgs_s_u], 0),
                torch.cat([img for img in imgs_t_u], 0))

    else:
        return (torch.cat([img for img in imgs_s_s], 0),
                torch.cat([img for img in imgs_t_s], 0),
                torch.cat([label for label in label_s], 0),
                torch.cat([img for img in imgs_s_u], 0),
                torch.cat([img for img in imgs_t_u], 0))


def get_SGD(net, name='SGD', lr=0.1, momentum=0.9, \
            weight_decay=5e-4, nesterov=True, bn_wd_skip=True):
    '''
    return optimizer (name) in torch.optim.
    If bn_wd_skip, the optimizer does not apply
    weight decay regularization on parameters in batch normalization.
    '''
    optim = getattr(torch.optim, name)

    decay = []
    no_decay = []
    for name, param in net.named_parameters():
        if ('bn' in name) and bn_wd_skip:
            no_decay.append(param)
        else:
            decay.append(param)

    per_param_args = [{'params': decay},
                      {'params': no_decay, 'weight_decay': 0.0}]

    optimizer = optim(per_param_args, lr=lr,
                      momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
    return optimizer



def update_ema(ema_model, model, alpha):
	# alpha = min(1 - 1 / (epoch + 1), alpha)
	for ema_param, param in zip(ema_model.parameters(), model.parameters()):
		ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
'''


def update_ema(ema_model, model, alpha):
    for params_train, params_eval in zip(model.parameters(), ema_model.parameters()):
        params_eval.copy_(params_eval * alpha + params_train.detach() * (1 - alpha))

    for buffer_train, buffer_eval in zip(model.buffers(), ema_model.buffers()):
        buffer_eval.copy_(buffer_train)
'''

def linear_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)

def update_bn(model, ema_model):
    for m2, m1 in zip(ema_model.named_modules(), model.named_modules()):
        if ('bn' in m2[0]) and ('bn' in m1[0]):
            bn2, bn1 = m2[1].state_dict(), m1[1].state_dict()
            bn2['running_mean'].data.copy_(bn1['running_mean'].data)
            bn2['running_var'].data.copy_(bn1['running_var'].data)
            bn2['num_batches_tracked'].data.copy_(bn1['num_batches_tracked'].data)



def get_cosine_schedule_with_warmup(optimizer,
                                    num_training_steps,
                                    num_cycles=7. / 16.,
                                    num_warmup_steps=0,
                                    last_epoch=-1):
    
    
    #Get cosine scheduler (LambdaLR).
    #if warmup is needed, set num_warmup_steps (int) > 0.
    

    def _lr_lambda(current_step):
        
        #_lr_lambda returns a multiplicative factor given an interger parameter epochs.
        #Decaying criteria: last_epoch
        

        if current_step < num_warmup_steps:
            _lr = float(current_step) / float(max(1, num_warmup_steps))
        else:
            num_cos_steps = float(current_step - num_warmup_steps)
            num_cos_steps = num_cos_steps / float(max(1, num_training_steps - num_warmup_steps))
            _lr = max(0.0, math.cos(math.pi * num_cycles * num_cos_steps))
        return _lr

    return LambdaLR(optimizer, _lr_lambda, last_epoch)
'''

        
def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_wait_steps=0,
                                    num_cycles=0.5,
                                    last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_wait_steps:
            return 0.0

        if current_step < num_warmup_steps + num_wait_steps:
            return float(current_step) / float(max(1, num_warmup_steps + num_wait_steps))

        progress = float(current_step - num_warmup_steps - num_wait_steps) / \
            float(max(1, num_training_steps - num_warmup_steps - num_wait_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)
'''

def space_augment_on_predicted(pred, work_q_list):
    batch = pred.shape[0]
    tmp = []
    for i in range(batch):
        op, val = work_q_list[i]
        pred1 = op(Image.fromarray(np.uint8(pred[i, 0, :, :])), val)
        tmp.append(np.array(pred1))
#         pred2 = Image.fromarray(np.array(pred1) * 255)
#         pred2.save("pred1.png")
#         input()
    pred = np.array(tmp)
    pred = np.expand_dims(pred, axis=1)

    return torch.tensor(pred, requires_grad=True, dtype=torch.float32)

def predicted_color_result(img, pred, label, threshold=0.5):
    img = img[0, 0, :, :]
    pred = pred[0, 0, :, :]
    label = label[0, 0, :, :]
    pred = torch.sigmoid(pred)
    pred = pred > threshold
    label = label == torch.max(label)

    TP_mask = (pred == 1) * (label == 1)
    FP_mask = (pred == 1) * (label == 0)
    FN_mask = (pred == 0) * (label == 1)

    img.float()
    img_r, img_g, img_b = img.clone(), img.clone(), img.clone()

    # TP
    img_r[TP_mask] = 1.0
    img_g[TP_mask] = 70 / 255
    img_b[TP_mask] = 70 / 255
    # FP
    img_r[FP_mask] = 0.7
    img_g[FP_mask] = 1.0
    img_b[FP_mask] = 0.4588
    # FN
    img_r[FN_mask] = 1.0
    img_g[FN_mask] = 0.96
    img_b[FN_mask] = 0.3922

    img = torch.stack([img_r, img_g, img_b], 0)
    img = img[None, :, :, :].float()

    pred = pred.float()
    pred_r, pred_g, pred_b = pred.clone(), pred.clone(), pred.clone()
    
    # FP is red color so set 0 to g and b color channel
    pred_r[FP_mask] = 0 # 0
    pred_g[FP_mask] = 1.0
    pred_b[FP_mask] = 0 # 0
    # FN is yellow color so set 0 to g and b color channel
    pred_r[FN_mask] = 1.0
    pred_g[FN_mask] = 0 # 0
    pred_b[FN_mask] = 0 # 0
    
    pred = torch.stack([pred_r, pred_g, pred_b], 0)
    pred = pred[None, :, :, :].float()

    return pred # img, pred

def make_grid(img, labels, pred, data, save_dir):
    img, labels, pred = img.cpu(), labels.cpu(), pred.cpu()
    pred = pred.permute(1, 2, 0)
    
    fig, ax = plt.subplots(1, 3, figsize=(14, 5))
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    fig.suptitle('Recall: {:.3f}   Precision: {:.3f}   F1: {:.3f}'.format(float(data["SE"]),
        float(data["PR"]),
        float(data["F1"])),
        fontsize=20, x=0.5, y=0.1, horizontalalignment='center')
    ax[0].set_axis_off()
    ax[0].imshow(img, cmap="gray", vmin=0, vmax=1)
    ax[0].set_title("Image", fontsize=18)

    ax[1].set_axis_off()
    ax[1].imshow(labels, cmap="gray", vmin=0, vmax=1)
    ax[1].set_title("Label", fontsize=18)

    ax[2].set_axis_off()
    ax[2].imshow(pred, vmin=0, vmax=1)
    ax[2].set_title("Predicted", fontsize=18)
    
    file_name = data["file_name"][:-4] + "_compare" + data["file_name"][-4:]
    fig.savefig(os.path.join(save_dir, file_name), pad_inches = 0)
    plt.close()

def tau_statistics(img):
    """
    0 <= img <= 1
    """

    # ci = cofidence interval
    ci = 0.1
    statistic = {}

    statistic[0.1] = torch.sum(img == 0).item()
    for i in range(1, 11):
        i /= 10
        if i == 0.1 :
            statistic[i] += torch.sum(torch.logical_and(img <= i, img > (i - ci))).item()
        else:
            statistic[i] = torch.sum(torch.logical_and(img <= i, img > (i - ci))).item()
    # print(statistic)

    # s = 0
    # for m, n in statistic.items():
    #     s += n
    # print("sum:", s)

    return statistic

def dicom_to_png(dcm):
    dcm = pydicom.dcmread(dcm)
    dcm = dcm.pixel_array.astype(np.float32) / 255
    return dcm
    
def get_dicom_fps(dcm):
    dcm = pydicom.dcmread(dcm)
    return round(1 / float(dcm[0x0018, 0x1063].value) * 1000)

def moving_average(data, period=3):
    """
    計算移動平均
    :param data: 原始變化量
    :param period: 平均週期
    :return: 平均後結果
    """
    if period > 0:
        ret = np.cumsum(data, dtype=float)
        ret[period:] = ret[period:] - ret[:-period]
        return np.append([np.nan] * (period - 1), ret[period - 1:] / period)
    return np.array(data)

def normal_centroid(x, y, period=3):
    """
    計算重心
    :param x: x 座標
    :param y: y 座標
    :param period: 平均週期
    :return:
    """
    ret = np.cumsum(x, dtype=float)
    ret[period:] = ret[period:] - ret[:-period]
    output_x = np.append([np.nan] * (period - 1), ret[period - 1:] / period)
    ret = np.cumsum(y, dtype=float)
    ret[period:] = ret[period:] - ret[:-period]
    output_y = np.append([np.nan] * (period - 1), ret[period - 1:] / period)
    return output_x, output_y

def gray_centroid(image, prev_x, prev_y, threshold=0.5):
    """
    灰度重心法
    參考網站: https://blog.csdn.net/moses1213/article/details/44679603
    :param image: 單張影像
    :param prev_x: 前一重心的 x 座標
    :param prev_y: 前一重心的 y 座標
    :param threshold: 定義門檻值以下的
    :return: (x, y) 重心座標
    """
    tmp = image.copy()
    tmp[tmp <= threshold] = 0

    count = np.sum(tmp)
    if not count:
        return prev_x, prev_y

    index_x = np.tile(np.arange(0, 512), (512, 1))
    index_y = index_x.T
    sum_x = np.sum(tmp * index_x)
    sum_y = np.sum(tmp * index_y)

    return sum_x / count, sum_y / count

def step_2_start_n_end(diff_num, centroid_distance, one_heartbeat_distance, moving_end):
    """
    重心影格差異法
    :param diff_num: 每個 predicted images frame 的變化量
    :param centroid_distance: 重心移動距離
    :param one_heartbeat_distance: 定義一次心跳的重心移動距離
    :param moving_end: 定義變化量趨緩的截止點
    :return: 開始與結束影格
    """
    # 從重心移動最大的點開始
    step_2_start = centroid_distance[1:int(centroid_distance.shape[0] / 2)].argmax() + 1
    while centroid_distance[step_2_start + 1] >= one_heartbeat_distance or \
            centroid_distance[step_2_start + 2] >= one_heartbeat_distance:
        step_2_start += 1
    # 如果 step_2_start 是 nan 或 <0，則繼續加 1
    while np.isnan(diff_num[step_2_start]) or diff_num[step_2_start] < 0 or diff_num[step_2_start + 1] < 0:
        step_2_start += 1
    # 下界門檻值，RCA normal 用 15，RCA problem 用 10
    step_2_end = step_2_start + 15
    while diff_num[step_2_end] >= moving_end and diff_num[step_2_end + 1] >= moving_end:
        step_2_end += 1
    step_2_start += 2  # index from 2
    step_2_end += 2  # index from 2

    return step_2_start, step_2_end
