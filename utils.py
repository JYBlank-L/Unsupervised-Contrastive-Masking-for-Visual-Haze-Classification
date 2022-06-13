import io
import os
import ipdb
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as utils
import math
from math import log10
from torch.autograd import Variable
from skimage import measure


# def draw(epoch, psnr_list, ssim_list, result_path, mode='Train', loss_list=None):
#     epochs = [i for i in range(epoch)]
#
#     if loss_list is not None:
#         plt.plot(epochs, loss_list, '-')
#         plt.title(mode + ' Loss')
#         plt.xlabel('Epochs')
#         plt.legend()
#         plt.savefig(result_path + mode + " Loss.png")
#
#         plt.clf()
#
#     plt.plot(epochs, psnr_list, '-')
#     plt.title(mode + ' PSNR')
#     plt.xlabel('Epochs')
#     plt.legend()
#     plt.savefig(result_path + mode + " PSNR.png")
#
#     plt.clf()
#
#     plt.plot(epochs, ssim_list, '-')
#     plt.title(mode + ' SSIM')
#     plt.xlabel('Epochs')
#     plt.legend()
#     plt.savefig(result_path + mode + " SSIM.png")


def getPath(root):
    root = root
    photo_path = root + 'images/'
    train_path = root + 'train.txt'
    test_path = root + 'test.txt'
    val_path = root + 'val.txt'
    running_path = [train_path, val_path, test_path]

    haze_images = [[], [], []]  # train,val,test
    haze_labels = [[], [], []]  # train,val,test

    for x in range(0, 3):
        # read labels
        label_path = running_path[x]
        with open(label_path, "r") as f:
            for line in f.readlines():
                line = line.strip('\n')
                per_image_path, per_image_label = line.split(' ')
                haze_images[x].append(photo_path + per_image_path)
                haze_labels[x].append(int(per_image_label))

    return haze_images, haze_labels


def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay=4, decay_rate=0.1):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (decay_rate ** (epoch // lr_decay))

    if epoch % lr_decay == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def val(opt, epoch, dataLoader, modelList, flag, num_classes, result_path, bestValPred):
    total = 0
    correct = 0
    test_predictions_list = np.empty(0)
    test_labels = np.empty(0)
    confusion_matrix = np.zeros((num_classes, num_classes))
    isFE = False
    if len(modelList) == 1:
        model = modelList[0]
    else:
        isFE = True
        model_x, model_y, model_xy = modelList
    with torch.no_grad():
        for images, labels in dataLoader:
            if opt.CUDA:
                images, labels = images.cuda(), labels.cuda()

            test = Variable(images)
            if not isFE:
                outputs = model(test)
                model.eval()
            else:
                seg_img = images[:, 0:3, :, :]
                org_img = images[:, 3:, :, :]
                org_outputs = model_x(org_img)
                seg_outputs = model_y(seg_img)
                xy = torch.cat([org_outputs, seg_outputs], dim=1)
                outputs = model_xy(xy)
                model_x.eval()
                model_y.eval()
                model_xy.eval()
            test_predictions = torch.max(outputs, 1)[1]
            correct += (test_predictions == labels).sum()
            total += len(labels)

            test_predictions = np.array(test_predictions.cpu())
            labels = np.array(labels.cpu())
            test_predictions_list = np.append(test_predictions_list, test_predictions)
            test_labels = np.append(test_labels, labels)

    if flag == 1:
        test_predictions_list = np.array(test_predictions_list)
        labels = np.array(test_labels)
        for i in range(num_classes):
            position = test_predictions_list[np.where(labels == i)]
            for j in position:
                confusion_matrix[i][int(j)] += 1
        np.savetxt(result_path + 'val_confusion_matrix.txt', confusion_matrix, fmt="%d", delimiter=" ")
        flag = 0

    prediction = torch.true_divide(correct * 100, total)
    with io.open(result_path + 'val.txt', 'a', encoding='utf-8') as file:
        file.write(
            "Iteration: {}, Correct: {}, Total: {}, Accuracy: {}%\n".format(epoch, correct, total, prediction.item()))

    return prediction.item(), flag
