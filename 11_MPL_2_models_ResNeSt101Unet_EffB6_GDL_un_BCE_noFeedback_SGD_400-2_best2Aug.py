from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader
from datetime import datetime
from dataset import ImageDataSet, ImageDataSet2
from metric import get_F1, get_precision, get_specificity, get_sensitivity
from apex import amp
from GDiceLoss import pGeneralizedDiceLoss, GeneralizedDiceLoss
from utils import createTrainHistory, saveDict, loadDict, exp_rampup, get_SGD, predicted_color_result, make_grid
from utils import get_cosine_schedule_with_warmup
from argparse import ArgumentParser
from unet_model_semi import UNet
from tqdm import tqdm
from RandAugment import *
from itertools import cycle
import os
import torch
from torchvision.utils import save_image
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import ssl
import numpy as np
import random
import csv
import pandas as pd

ssl._create_default_https_context = ssl._create_unverified_context
seed = 1234
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)



class MPL:
    def __init__(self,
                 teacher_model,
                 student_model,
                 learning_rate_t,
                 learning_rate_s,
                 img_dir,
                 label_dir,
                 save_dir,
                 l2_decay,
                 batch_size_l,
                 batch_size_u,
                 epochs,
                 finetune_epochs,
                 cpu_core,
                 device,
                 train_txt_path,
                 train_un_txt_path,
                 valid_txt_path,
                 train_iter_s,
                 train_iter_t,
                 metric_item,
                 su_loss1,
                 un_loss,
                 threshold,
                 cons_weight,
                 model_path,
                 apex=True,
                 seed=1234):

        self.device = device
        self.cons_weight = cons_weight
        self.threshold = threshold
        torch.manual_seed(seed)
        self.student_model = student_model.to(device)
        self.model_path = model_path
        torch.manual_seed(seed)
        self.teacher_model = teacher_model.to(device)

        # Count model's parameters
        print("=" * 50)
        print("student model parameters: {:.2f}M".format(sum(p.numel() for p in self.student_model.parameters())/1e6))
        print("teacher model parameters: {:.2f}M".format(sum(p.numel() for p in self.teacher_model.parameters())/1e6))
        print("=" * 50)

        self.su_loss1 = su_loss1.to(device)
        self.un_loss = un_loss.to(device)

        self.optim_s = get_SGD(net=self.student_model,
                               lr=learning_rate_s,
                               weight_decay=l2_decay)

        self.optim_t = get_SGD(net=self.teacher_model,
                               lr=learning_rate_t,
                               weight_decay=l2_decay)

        self.apex = apex
        # nvidia apex initialize model and optimizer
        if self.apex:
            self.student_model, self.optim_s = amp.initialize(self.student_model,
                                                      self.optim_s,
                                                      opt_level="O1")

            self.teacher_model, self.optim_t = amp.initialize(self.teacher_model,
                                                          self.optim_t,
                                                          opt_level="O1")

        self.scheduler_s = get_cosine_schedule_with_warmup(optimizer=self.optim_s, num_warmup_steps=0,
                                                           num_training_steps=train_iter_s)
        self.scheduler_t = get_cosine_schedule_with_warmup(optimizer=self.optim_t, num_warmup_steps=0,
                                                           num_training_steps=train_iter_t)

        self.save_dir = save_dir
        if not os.path.isdir(str(self.save_dir)):
            os.makedirs(str(self.save_dir))

        self.epochs = epochs
        self.finetune_epochs = finetune_epochs
        # dataset & dataloader

        self.train_dataset_l = ImageDataSet(txt_path=train_txt_path,
                                            img_dir=img_dir,
                                            label_dir=label_dir)

        self.train_dataset_u = ImageDataSet2(labeled_txt_path=train_txt_path,
                                             un_txtPath=train_un_txt_path,
                                             img_dir=img_dir,
                                             transform=RandAugment_best_2aug(1, 20))

        self.valid_dataset = ImageDataSet(txt_path=valid_txt_path,
                                          img_dir=img_dir,
                                          label_dir=label_dir,
                                          sort=True)

        self.train_loader_l = DataLoader(self.train_dataset_l,
                                         batch_size=batch_size_l,
                                         shuffle=True,
                                         num_workers=cpu_core,
                                         pin_memory=True,
                                         drop_last=True)

        self.train_loader_u = DataLoader(self.train_dataset_u,
                                         batch_size=batch_size_u,
                                         shuffle=True,
                                         num_workers=cpu_core,
                                         pin_memory=True,
                                         drop_last=True)

        self.valid_loader = DataLoader(self.valid_dataset,
                                       batch_size=1,
                                       shuffle=False,
                                       num_workers=cpu_core,
                                       pin_memory=True)


        self.moving_dot_product = torch.empty(1).to(self.device)
        limit = 3.0 ** (0.5)  # 3 = 6 / (f_in + f_out)
        nn.init.uniform_(self.moving_dot_product, -limit, limit)

        self.train_history = createTrainHistory(metric_item)
        self.best_target = 0.0
        self.best_target_epoch = 0

    def train(self, epoch):
        self.student_model.train()
        self.teacher_model.train()

        su_loss = 0.0
        un_loss = 0.0
        mpl = 0.0
        # try:
        with tqdm(total=len(self.train_dataset_u), desc="train ", unit="img",
                  bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}') as pbar:
            # get imgs, labels from train_loader_l
            # un_w, un_s from train_loader_u
            for (imgs, labels), (un_w, un_s) in zip(cycle(self.train_loader_l), self.train_loader_u):
                # cast imgs, labels, un_w, un_s to device(gpu) and transform type into torch.float32
                imgs, labels = imgs.to(self.device, dtype=torch.float32), labels.to(self.device,
                                                                                    dtype=torch.float32)
                un_w, un_s = un_w.to(self.device, dtype=torch.float32), un_s.to(self.device, dtype=torch.float32)
                un_batch_size = un_s.shape[0]
                # teacher part1 (UDA) : supervised and unsupervised
                # get each prediction from teacher model
                pred1 = self.teacher_model(imgs)
                # calculate supervised loss for pred1 & labels

                # loss1 = self.su_loss1(torch.sigmoid(pred1), labels)
                loss1 = self.su_loss1(pred1, labels)
                # loss1 += self.dice_loss(torch.sigmoid(pred1), labels, k=0.75)

                su_loss += loss1.item() / imgs.shape[0]
                del pred1, imgs, labels

                # get mask torch.sigmoid(pred_w) > threshold
                pred_w, pred_s = self.teacher_model(un_w), self.teacher_model(un_s)
                mask = (torch.sigmoid(pred_w).ge(self.threshold)).to(dtype=torch.float32)
                # calculate consistency loss
                cons_loss = self.cons_weight * self.un_loss(pred_s, torch.sigmoid(pred_w.detach()) * mask) # pred_w.detach() * mask

                un_loss += cons_loss.item() / un_batch_size
                del pred_w, un_s, pred_s
                # clear all optimizer grad before this epoch loss backpropagation
                self.optim_s.zero_grad()
                self.optim_t.zero_grad()
                # Student
                # get prediction from student model
                #     student_l, student_uw = self.student_model(imgs), self.student_model(un_w)
                student_uw = self.student_model(un_w)
                # calculate loss before student model parameter update for validate student model's performance
                #     s_loss_l_old = self.su_loss1(student_l.detach(), labels)
                # calculate supervised loss for student model to do backpropagation
                s_loss = self.su_loss1(student_uw, mask)

                # s_loss += self.dice_loss(torch.sigmoid(student_uw), mask, k=0.75)


                del student_uw, mask, un_w

                # student loss backpropagation
                if self.apex:
                    with amp.scale_loss(s_loss, self.optim_s) as scaled_loss:
                        scaled_loss.backward()
                else:
                    s_loss.backward()
                # update student model's parameter
                self.optim_s.step()
                self.scheduler_s.step()

                # MPL
                # with torch.no_grad():
                #     student_l = self.student_model(imgs)
                # # calculate loss after student model parameter update for validate student model's performance
                # s_loss_l_new = self.su_loss1(student_l.detach(), labels)
                # del student_l, imgs, labels
                # # signal calculation
                # dot_product = s_loss_l_new.item() - s_loss_l_old.item()
                # self.moving_dot_product = self.moving_dot_product * 0.99 + dot_product * 0.01
                # dot_product = dot_product - self.moving_dot_product
                # dot_product.requires_grad = True
                # mask = (torch.sigmoid(pred_s).ge(self.threshold)).to(dtype=torch.float32)
                # # MPL loss
                # loss_mpl = dot_product * self.su_loss1(pred_s, mask)
                # mpl += loss_mpl.item() / pred_s.shape[0]
                # # sum teacher model's loss: supervised loss + consistency loss + MPL loss
                # t_loss = loss1 + cons_loss + loss_mpl
                t_loss = loss1 + cons_loss

                # teacher loss backpropagation
                if self.apex:
                    with amp.scale_loss(t_loss, self.optim_t) as scaled_loss:
                        scaled_loss.backward()
                else:
                    t_loss.backward()

                # update teacher model's parameter
                self.optim_t.step()
                self.scheduler_t.step()

                pbar.update(un_batch_size)

        # except Exception as e:
        # print(e)

        # finally:
        self.train_history["train"]["loss"].append(mpl + un_loss + su_loss)


    def valid(self):
        self.student_model.eval()
        v_SE = 0.0
        v_SP = 0.0
        v_PR = 0.0
        v_F1 = 0.0

        try:
            with torch.no_grad():
                with tqdm(total=len(self.valid_dataset), desc="valid ", unit="img",
                          bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}') as pbar:
                    for imgs, labels in self.valid_loader:
                        imgs, labels = imgs.to(self.device, dtype=torch.float32), labels.to(self.device,
                                                                                            dtype=torch.float32)

                        pred = self.student_model(imgs)
                        SP = get_specificity(pred, labels)
                        SE = get_sensitivity(pred, labels)
                        PR = get_precision(pred, labels)
                        F1 = get_F1(pred, labels)

                        v_SE += SE
                        v_SP += SP
                        v_PR += PR
                        v_F1 += F1
                        pbar.update(imgs.shape[0])

        except Exception as e:
            print(e)

        finally:
            self.train_history["valid"]["SE"].append(v_SE / len(self.valid_loader))
            self.train_history["valid"]["SP"].append(v_SP / len(self.valid_loader))
            self.train_history["valid"]["PR"].append(v_PR / len(self.valid_loader))
            self.train_history["valid"]["F1"].append(v_F1 / len(self.valid_loader))

    def finetune(self, epoch):
        checkpoint = torch.load(self.model_path, self.device)
        self.student_model.load_state_dict(checkpoint)
        self.student_model.train()
        su_loss = 0.0
        mpl = 0.0
        with tqdm(total=len(self.train_dataset_l), desc="finetune ", unit="img",
                  bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}') as pbar:
            # get imgs, labels from train_loader_l
            for imgs, labels in self.train_loader_l:
                # cast imgs, labels, un_w, un_s to device(gpu) and transform type into torch.float32
                imgs, labels = imgs.to(self.device, dtype=torch.float32), labels.to(self.device,
                                                                                    dtype=torch.float32)
                pred = self.student_model(imgs)
                loss1 = self.su_loss1(pred, labels)
                print(loss1)
                # loss1 += self.dice_loss(torch.sigmoid(pred1), labels, k=0.75)

                su_loss += loss1.item() / imgs.shape[0]

                # clear all optimizer grad before this epoch loss backpropagation
                self.optim_s.zero_grad()

                # student loss backpropagation
                if self.apex:
                    with amp.scale_loss(loss1, self.optim_s) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss1.backward()
                # update student model's parameter
                self.optim_s.step()
                self.scheduler_s.step()

                pbar.update(imgs.shape[0])
                self.train_history["train"]["loss"].append(su_loss)

    def load_model_and_save_csv(self):
        checkpoint = torch.load(self.model_path, self.device)
        self.student_model.load_state_dict(checkpoint)
        current_fold = str(self.save_dir)[-1]
        csv_file_path = str(self.save_dir) + "/../" + "{}.csv".format(str(self.save_dir).split("/")[-2])

        self.student_model.eval()
        v_loss = 0.0
        v_SE = 0.0
        v_SP = 0.0
        v_PR = 0.0
        v_F1 = 0.0
        with torch.no_grad():
            with tqdm(total=len(self.valid_dataset), desc="valid ", unit="img",
                      bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}') as pbar:

                if current_fold == "1" or True:
                    f = open(csv_file_path, "w")
                    load_model_writer = csv.writer(f)
                    load_model_writer.writerow(["Fold", "filename", "loss", "SP", "SE", "PR", "F1"])
                else:
                    f = open(csv_file_path, "a")
                    load_model_writer = csv.writer(f)
                file_index = 0
                for imgs, labels in self.valid_loader:
                    imgs, labels = imgs.to(self.device, dtype=torch.float32), labels.to(self.device,
                                                                                        dtype=torch.float32)
                    file_name = self.valid_dataset.fileNames[file_index]
                    pred = self.student_model(imgs)
                    # loss = self.su_loss1(torch.sigmoid(pred), labels)
                    loss = self.su_loss1(pred, labels)
                    SP = get_specificity(pred, labels)
                    SE = get_sensitivity(pred, labels)
                    PR = get_precision(pred, labels)
                    F1 = get_F1(pred, labels)
                    pred = predicted_color_result(imgs, pred, labels)
                    save_image(pred, self.save_dir/file_name)
                    make_grid(imgs[0, 0, :, :], labels[0, 0, :, :], pred[0, :, :, :],
                        {"SE": SE, "PR": PR, "F1": F1, "file_name": file_name}, str(self.save_dir))

                    load_model_writer.writerow([str(self.save_dir)[-1], file_name, loss.item(), SP, SE, PR, F1])
                    v_loss += loss.item()
                    v_SE += SE
                    v_SP += SP
                    v_PR += PR
                    v_F1 += F1

                    file_index += 1

                    pbar.update(imgs.shape[0])
                f.close()
                if current_fold == "5" or True:
                    total_5_csv = pd.read_csv(csv_file_path)
                    avg_result = [total_5_csv['loss'].mean(),
                        total_5_csv['SP'].mean(),
                        total_5_csv['SE'].mean(),
                        total_5_csv['PR'].mean(),
                        total_5_csv['F1'].mean()]
                    std_result = [total_5_csv['loss'].std(),
                        total_5_csv['SP'].std(),
                        total_5_csv['SE'].std(),
                        total_5_csv['PR'].std(),
                        total_5_csv['F1'].std()]
                    fold_F1_avg = [total_5_csv['F1'].loc[:99].mean(),
                        total_5_csv['F1'].loc[100:199].mean(),
                        total_5_csv['F1'].loc[200:299].mean(),
                        total_5_csv['F1'].loc[300:399].mean(),
                        total_5_csv['F1'].loc[400:499].mean()]
                    fold_F1_avg = [float(i) for i in fold_F1_avg]
                    fold_F1_std = [total_5_csv['F1'].loc[:99].std(),
                        total_5_csv['F1'].loc[100:199].std(),
                        total_5_csv['F1'].loc[200:299].std(),
                        total_5_csv['F1'].loc[300:399].std(),
                        total_5_csv['F1'].loc[400:499].std()]
                    fold_F1_std = [float(i) for i in fold_F1_std]
                    with open(csv_file_path, "a") as f:
                        load_model_writer = csv.writer(f)
                        load_model_writer.writerow(["", "", "loss_avg", "SP_avg", "SE_avg", "PR_avg", "F1_avg"])
                        load_model_writer.writerow(["", ""] + [round(i, 5) for i in avg_result])
                        load_model_writer.writerow(["", ""] + [round(i, 4) for i in std_result])
                        load_model_writer.writerow(["", "", "fold1_F1_avg", "fold2_F1_avg", "fold3_F1_avg", "fold4_F1_avg", "fold5_F1_avg"])
                        load_model_writer.writerow(["", ""] + [round(i, 5) for i in fold_F1_avg])
                        load_model_writer.writerow(["", ""] + [round(i, 4) for i in fold_F1_std])
                        load_model_writer.writerow(["", "", "fold_F1_max", "", "", "", ""])
                        load_model_writer.writerow(["", ""] + [round(max(fold_F1_avg),5)] + [fold_F1_avg.index(max(fold_F1_avg))+1])
                        load_model_writer.writerow(["", ""] + [round(fold_F1_std[fold_F1_avg.index(max(fold_F1_avg))],4)])

    def all_thresholds(self):
        checkpoint = torch.load(self.model_path, self.device)
        self.student_model.load_state_dict(checkpoint)
        current_fold = str(self.save_dir)[-1]
        csv_PR_curve_plot = str(self.save_dir) + "/../" + "pr_curve_0.1_0.9.csv"
        csv_PR_curve_best_th = str(self.save_dir) + "/../" + "pr_curve_0.01_0.99.csv"
        csv_PR_curve_best_th_100_valid = str(self.save_dir) + "/../" + "best_th_valid_data.csv"


        self.student_model.eval()
        with torch.no_grad():
            th_start = 1
            th_end_plot = 10
            th_end_best_th = 100
            step_plot = 10
            step_best_th = 100
            with tqdm(total=len(self.valid_dataset) * (th_end_best_th - th_start), desc="PR Curve ", unit="img",
                      bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}') as pbar:
                pr_curve_plot = {current_fold: {"SE": [], "PR":[],"F1":[]}}  # 放目前的 fold 的 SE 與 PR 0.1-0.9
                pr_curve_best_th = {current_fold: {"SE": [], "PR":[],"F1":[]}}  # 放目前的 fold 的 SE 與 PR 0.01-0.99
                best_v_F1 = 0
                for th in range(th_start, th_end_best_th):
                    v_loss = 0.0
                    v_SE = 0.0
                    v_SP = 0.0
                    v_PR = 0.0
                    v_F1 = 0.0
                    file_index = 0
                    th /= step_best_th  # 0.01 ~ 0.99
                    best_v_F1_csv = {
                        "Fold": [],
                        "filename": [],
                        "loss": [],
                        "SP": [],
                        "SE": [],
                        "PR": [],
                        "F1": []
                    }
                    for imgs, labels in self.valid_loader:
                        imgs, labels = imgs.to(self.device, dtype=torch.float32), labels.to(self.device,
                                                                                            dtype=torch.float32)
                        file_name = self.valid_dataset.fileNames[file_index]
                        pred = self.student_model(imgs)
                        # loss = self.su_loss1(torch.sigmoid(pred), labels)
                        loss = self.su_loss1(pred, labels)
                        SP = get_specificity(pred, labels, threshold=th)
                        SE = get_sensitivity(pred, labels, threshold=th)
                        PR = get_precision(pred, labels, threshold=th)
                        F1 = get_F1(pred, labels, threshold=th)
                        best_v_F1_csv["Fold"].append(current_fold)
                        best_v_F1_csv["filename"].append(file_name)
                        best_v_F1_csv["loss"].append(loss.item())
                        best_v_F1_csv["SP"].append(SP)
                        best_v_F1_csv["SE"].append(SE)
                        best_v_F1_csv["PR"].append(PR)
                        best_v_F1_csv["F1"].append(F1)

                        v_loss += loss.item()
                        v_SE += SE
                        v_SP += SP
                        v_PR += PR
                        v_F1 += F1
                        file_index += 1

                        pbar.update(imgs.shape[0])
                    if v_F1 > best_v_F1:
                        best_v_F1 = v_F1
                        with open(csv_PR_curve_best_th_100_valid, "w") as f:
                            pickle_writer = csv.writer(f)
                            pickle_writer.writerow((["Fold", "filename", "loss", "SP", "SE", "PR", "F1"]))
                            for i in range(len(best_v_F1_csv["Fold"])):
                                pickle_writer.writerow([
                                    best_v_F1_csv["Fold"][i],
                                    best_v_F1_csv["filename"][i],
                                    best_v_F1_csv["loss"][i],
                                    best_v_F1_csv["SP"][i],
                                    best_v_F1_csv["SE"][i],
                                    best_v_F1_csv["PR"][i],
                                    best_v_F1_csv["F1"][i],
                                ])
                    if th * 10 in list(range(1,10)):
                        pr_curve_plot[current_fold]["SE"].append(v_SE/len(self.valid_loader))  # 新增 SE
                        pr_curve_plot[current_fold]["PR"].append(v_PR/len(self.valid_loader))  # 新增 PR
                        pr_curve_plot[current_fold]["F1"].append(v_F1/len(self.valid_loader))  # 新增 F1
                    pr_curve_best_th[current_fold]["SE"].append(v_SE/len(self.valid_loader))  # 新增 SE
                    pr_curve_best_th[current_fold]["PR"].append(v_PR/len(self.valid_loader))  # 新增 PR
                    pr_curve_best_th[current_fold]["F1"].append(v_F1/len(self.valid_loader))  # 新增 F1

                # print(pr_curve)
                saveDict("{}/../PR_curve_fold_{}_plot.pickle".format(str(self.save_dir), current_fold), pr_curve_plot)
                saveDict("{}/../PR_curve_fold_{}_best_th.pickle".format(str(self.save_dir), current_fold), pr_curve_best_th)

                if current_fold == "5" or True:
                    # combine pickles
                    FiveFold_SE_plot = np.zeros((1, (th_end_plot - th_start)), dtype=np.float32)[0]
                    FiveFold_PR_plot = np.zeros((1, (th_end_plot - th_start)), dtype=np.float32)[0]
                    FiveFold_F1_plot = np.zeros((1, (th_end_plot - th_start)), dtype=np.float32)[0]
                    FiveFold_SE_best_th = np.zeros((1, (th_end_best_th - th_start)), dtype=np.float32)[0]
                    FiveFold_PR_best_th = np.zeros((1, (th_end_best_th - th_start)), dtype=np.float32)[0]
                    FiveFold_F1_best_th = np.zeros((1, (th_end_best_th - th_start)), dtype=np.float32)[0]
                    for fold_index in range(2, 3):
                        fold_index = str(fold_index)
                        pickle_data_plot = loadDict("{}/../PR_curve_fold_{}_plot.pickle".format(str(self.save_dir), fold_index))
                        pickle_data_best_th = loadDict("{}/../PR_curve_fold_{}_best_th.pickle".format(str(self.save_dir), fold_index))
                        print(pickle_data_plot)
                        print(pickle_data_best_th)
                        FiveFold_SE_plot += np.array(pickle_data_plot[fold_index]["SE"])
                        FiveFold_PR_plot += np.array(pickle_data_plot[fold_index]["PR"])
                        FiveFold_F1_plot += np.array(pickle_data_plot[fold_index]["F1"])
                        FiveFold_SE_best_th += np.array(pickle_data_best_th[fold_index]["SE"])
                        FiveFold_PR_best_th += np.array(pickle_data_best_th[fold_index]["PR"])
                        FiveFold_F1_best_th += np.array(pickle_data_best_th[fold_index]["F1"])
                    # FiveFold_SE = FiveFold_SE / 5
                    # FiveFold_PR = FiveFold_PR / 5
                    # FiveFold_F1 = FiveFold_F1 / 5
                    with open(csv_PR_curve_plot, "w") as f:
                        pickle_writer = csv.writer(f)
                        pickle_writer.writerow(["TH", ] + [i/step_plot for i in range(th_start, th_end_plot)])
                        pickle_writer.writerow(["SE", ] + FiveFold_SE_plot.tolist())
                        pickle_writer.writerow(["PR", ] + FiveFold_PR_plot.tolist())
                        pickle_writer.writerow(["F1", ] + FiveFold_F1_plot.tolist())
                    # os.system("rm {}/../PR_curve_fold_*.pickle".format(str(self.save_dir)))
                    with open(csv_PR_curve_best_th, "w") as f:
                        pickle_writer = csv.writer(f)
                        pickle_writer.writerow(["TH", ] + [i/step_best_th for i in range(th_start, th_end_best_th)])
                        pickle_writer.writerow(["SE", ] + FiveFold_SE_best_th.tolist())
                        pickle_writer.writerow(["PR", ] + FiveFold_PR_best_th.tolist())
                        pickle_writer.writerow(["F1", ] + FiveFold_F1_best_th.tolist())

    def start(self):
        begin_time = datetime.now()
        print("*" * 150)
        print(f"training start at {begin_time}")
        print(f"Torch Version : {torch.__version__}")
        print(f"Device : {self.device}")
        print("*" * 150)

#         try:
        for epoch in range(self.epochs):
            print("*" * 150)
            print(f"Epoch {epoch + 1}/{self.epochs} \n")
            start_time = datetime.now()
            self.train(epoch)
            self.valid()
            end_time = datetime.now()
            epoch_total_time = end_time - start_time
            self.show_epoch_results(epoch, epoch_total_time)
            self.save_best_target(epoch)
            saveDict("%s/train_history.pickle" % (str(self.save_dir)), self.train_history)
            print(f"Epoch {epoch + 1} end")
        finish_time = datetime.now()

#         except Exception as e:
#             print(e)

#         finally:
        print("*" * 150)
        print(f"training end at {finish_time}")
        print(f"Total Training Time : {finish_time - begin_time}")
        print("*" * 150)

    def finetune_start(self):
        begin_time = datetime.now()
        print("*" * 150)
        print(f"training start at {begin_time}")
        print(f"Torch Version : {torch.__version__}")
        print(f"Device : {self.device}")
        print("*" * 150)

#         try:
        for epoch in range(self.finetune_epochs):
            print("*" * 150)
            print(f"Epoch {epoch + 1}/{self.finetune_epochs} \n")
            start_time = datetime.now()
            self.finetune(epoch)
            self.valid()
            end_time = datetime.now()
            epoch_total_time = end_time - start_time
            self.show_epoch_results(epoch, epoch_total_time)
            self.save_best_target(epoch)
            saveDict("%s/train_history_finetune.pickle" % (str(self.save_dir)), self.train_history)
            print(f"Epoch {epoch + 1} end")
        finish_time = datetime.now()

#         except Exception as e:
#             print(e)

#         finally:
        print("*" * 150)
        print(f"training end at {finish_time}")
        print(f"Total Training Time : {finish_time - begin_time}")
        print("*" * 150)

    def save_best_target(self, epoch):
        try:
            if (self.train_history["valid"]["F1"][epoch]) > self.best_target:
                self.best_target_epoch = epoch + 1
                last_best = self.best_target
                self.best_target = self.train_history["valid"]["F1"][epoch]
                print(f"F1 improves from {last_best:2.5f} to {self.best_target:2.5f}")
                torch.save(self.student_model.state_dict(), self.save_dir / "bestF1.pth")
                print(f"save model to {str(self.save_dir / 'bestF1.pth')}")
            else:
                print(f"valid_F1 did not improve from {self.best_target:2.5f} "
                      f"since Epoch {self.best_target_epoch:d}")
        except Exception as e:
            print(e)

    def show_epoch_results(self, epoch, epoch_total_time):
        print(f"Epoch {epoch + 1} time : {epoch_total_time.seconds} secs, "
              f"loss : {self.train_history['train']['loss'][epoch]:2.5f}, "
              f"valid_PR : {self.train_history['valid']['PR'][epoch]:2.5f}, "
              f"valid_SE : {self.train_history['valid']['SE'][epoch]: 2.5f}, "
              f"valid_SP : {self.train_history['valid']['SP'][epoch]:2.5f}, "
              f"valid_F1 : {self.train_history['valid']['F1'][epoch]:2.5f}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--bl", "--batch_size_labeled", default=20)
    parser.add_argument("--bu", "--batch_size_un", default=80)
    parser.add_argument("--e", "--Epoch", default=200)
    parser.add_argument("--fe", "--finetune_epoch", default=125)
    parser.add_argument("--c", "--cpu_core", default=10)
    parser.add_argument("--d", "--device",
                        default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--su_loss1", default=GeneralizedDiceLoss()) # nn.BCEWithLogitsLoss()
    parser.add_argument("--un_loss", default=nn.BCEWithLogitsLoss()) # nn.MSELoss()
    parser.add_argument("--i_dir", "--img_dir", default=Path("/mnt/workspace/CAG/imgs"))
    parser.add_argument("--l_dir", "--label_dir", default=Path("/mnt/workspace/CAG/labels2"))
    parser.add_argument("--s_dir", "--save_dir", default=Path("./record/11_MPL_2_models_ResNeSt101Unet_EffB6_GDL_un_BCE_noFeedback_SGD_400-2_best2Aug/5F_2"))
    parser.add_argument("--t_txt_path", "--train_txt_path", default="/mnt/workspace/s3_train/1229/labeled_400_2.txt")
    parser.add_argument("--t_un_txt_path", "--train_un_txt_path", default="/mnt/workspace/s3_train/1229/unlabeled_all.txt")
    parser.add_argument("--v_txt_path", "--valid_txt_path", default="/mnt/workspace/s3_train/1229/valid_2.txt")
    parser.add_argument("--ms", "--student_model", default=smp.Unet("efficientnet-b6", in_channels=1, classes=1)) # timm-resnest101e
    parser.add_argument("--mt", "--teacher_model", default=smp.Unet("timm-resnest101e", in_channels=1, classes=1))
    parser.add_argument("--lr_t", "--learning_rate_t", default=4e-1)
    parser.add_argument("--lr_s", "--learning_rate_s", default=8e-1)
    parser.add_argument("--l2", "--weight_decay", default=1e-4)
    parser.add_argument("--k", "--metric", default=["loss", "SE", "SP", "PR", "F1"])
    parser.add_argument("--cons_w", "--consistency_weight", default=1)
    parser.add_argument("--th", "--threshold", default=0.9)
    parser.add_argument("--iter_s", "--train_iter_unlabeled", default=600)
    parser.add_argument("--iter_t", "--train_iter", default=600)
    parser.add_argument("--model_path", default="./record/11_MPL_2_models_ResNeSt101Unet_EffB6_GDL_un_BCE_noFeedback_SGD_400-2_best2Aug/5F_2/bestF1.pth")
    args = parser.parse_args()



    train = MPL(teacher_model=args.mt,
                student_model=args.ms,
                batch_size_l=args.bl,
                batch_size_u=args.bu,
                su_loss1=args.su_loss1,
                un_loss=args.un_loss,
                device=args.d,
                learning_rate_t=args.lr_t,
                learning_rate_s=args.lr_s,
                l2_decay=args.l2,
                img_dir=args.i_dir,
                label_dir=args.l_dir,
                save_dir=args.s_dir,
                cpu_core=args.c,
                metric_item=args.k,
                epochs=args.e,
                finetune_epochs=args.fe,
                train_txt_path=args.t_txt_path,
                train_un_txt_path=args.t_un_txt_path,
                valid_txt_path=args.v_txt_path,
                threshold=args.th,
                train_iter_s=args.iter_s,
                train_iter_t=args.iter_t,
                cons_weight=args.cons_w,
                model_path=args.model_path)
    train.start()
    # train.finetune_start()
    train.load_model_and_save_csv()
    # train.all_thresholds()
