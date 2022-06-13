from __future__ import print_function
import torch.optim as optim
import argparse
import itertools
import os
from os.path import join
import torch
import ipdb
from torch.utils.data import DataLoader
from importlib import import_module
import random
import re
import time
import statistics
import torch.nn.functional as F
from datasets.Level import Haze_Level
from datasets.Wild import Haze_Wild
from utils import *

# Training settings
parser = argparse.ArgumentParser(description="PyTorch Train")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate, default=1e-4")
parser.add_argument("--batchSize", type=int, default=128, help="Training batch size")
parser.add_argument("--testBatchSize", type=int, default=1, help="Test batch size")
parser.add_argument("--epochs", type=int, default=46, help="Train epoch")
parser.add_argument("--decay", type=int, default=15, help="Change the learning rate for every N epochs")
parser.add_argument("--decay_rate", type=float, default=0.1, help="Decay scale of learning rate, default=0.5")
parser.add_argument('--dataset', default="wild", type=str, help='hazel_level or haze_wild')
parser.add_argument('--model', default='ResNet', type=str, help='select ResNet or LeNet as backbone')
parser.add_argument('--name', default='base', type=str, help='Filename of the training models')
parser.add_argument('--crop_size', type=int, help='Set the crop_size', default=[64, 64], nargs='+')
parser.add_argument('--filters', type=int, help='Set the size of filters of ucm', default=[40, 40, 70], nargs='+')
parser.add_argument("--method", type=int, default=0, help="0->base, 1->IA, 2->NC, 3->FE")
parser.add_argument("--CUDA", type=int, default=1, help="1 for using CUDA, 0 for not")
parser.add_argument("--use_lr_decay", type=bool, default=True, help="use decay")
parser.add_argument("--pretrained", type=bool, default=False, help="use pretrained model")
parser.add_argument("--data_name", type=str, default='original_64', help="data_name: original or seg and original or mask and original")

opt = parser.parse_args()
# opt.seed = random.randint(1, 10000)
opt.seed = 1
result_path = './result/backbone:' + str(opt.model) + '_name:' + str(opt.name) + '_SEED:' + str(
    opt.seed) + '_lr:' + str(opt.lr) + '_bs:' + str(opt.batchSize) + '_epochs:' + str(opt.epochs) + \
              '_decay:' + str(opt.decay) + '_decayRate' + str(opt.decay_rate) + '/'
if not os.path.exists(result_path):
    os.makedirs(result_path)
Net = import_module(opt.model + '.' + opt.name)

torch.manual_seed(opt.seed)
if opt.CUDA:
    torch.cuda.manual_seed(opt.seed)

if opt.dataset is 'level':
    num_classes = 9
    data_path = './data/'
    train_datasets = Haze_Level(data_path, opt.crop_size, opt.filters, method=opt.method, data_name=opt.data_name, mode='train')
    trainLoader = DataLoader(dataset=train_datasets, batch_size=opt.batchSize, shuffle=True, num_workers=4)
    val_datasets = Haze_Level(data_path, opt.crop_size, opt.filters, method=opt.method, data_name=opt.data_name, mode='val')
    valLoader = DataLoader(dataset=val_datasets, batch_size=opt.testBatchSize, shuffle=False, num_workers=4)
else:
    num_classes = 10
    data_path = './data/'
    # images_path, labels = getPath(data_path)
    train_datasets = Haze_Wild(data_path, opt.crop_size, opt.filters, data_name=opt.data_name, mode='train')
    trainLoader = DataLoader(dataset=train_datasets, batch_size=opt.batchSize, shuffle=True, num_workers=4)
    val_datasets = Haze_Wild(data_path, opt.crop_size, opt.filters, method=opt.method, mode='val')
    valLoader = DataLoader(dataset=val_datasets, batch_size=opt.testBatchSize, shuffle=False, num_workers=4)
    test_datasets = Haze_Wild(data_path, opt.crop_size, opt.filters, method=opt.method, mode='test')
    testLoader = DataLoader(dataset=test_datasets, batch_size=opt.testBatchSize, shuffle=False, num_workers=4)

if opt.method != 3:
    model = Net.createModel(num_classes)
    # if opt.pretrained:
    #     pretrained_model = result_path + 'bestVal.pt'
    #     model.load_state_dict(torch.load(pretrained_model))
    if opt.CUDA:
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
else:
    model_x, model_y, model_xy = Net.createModel(num_classes)
    if opt.CUDA:
        model_x, model_y, model_xy = model_x.cuda(), model_y.cuda(), model_xy.cuda()
    optimizer = torch.optim.Adam(itertools.chain(model_x.parameters(), model_y.parameters(), model_xy.parameters()),
                                 lr=opt.lr)

error = nn.CrossEntropyLoss()
if opt.CUDA:
    error = error.cuda()

bestLoss = 100
bestValPred = 0
flag = 0

# -------------------------- train ------------------------------
for epoch in range(1, opt.epochs + 1):
    loss_list = []
    precision_list = []
    modelList = []
    if opt.use_lr_decay:
        optimizer = exp_lr_scheduler(optimizer, epoch, opt.lr, opt.decay)
    start_time = time.time()
    for batch_id, [images, labels] in enumerate(trainLoader):
        batch_time = time.time()
        if opt.CUDA:
            images = images.cuda()
            labels = labels.cuda()

        if opt.method == 3:
            model_x.train()
            model_y.train()
            model_xy.train()
            seg_img = images[:, 0:3, :, :]
            org_img = images[:, 3:, :, :]
            org_outputs = model_x(org_img)
            seg_outputs = model_y(seg_img)
            xy = torch.cat([org_outputs, seg_outputs], dim=1)
            outputs = model_xy(xy)
            loss = error(outputs, labels.long())
        elif opt.method == 2:
            model.train()
            outputs, similarity = model(images)
            similarity = similarity.mean()
            loss = error(outputs, labels.long()) + similarity * 0.001
        else:
            model.train()
            outputs = model(images)
            loss = error(outputs, labels.long())

        loss_list.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predictions = torch.max(outputs, 1)[1]
        hit = (predictions == labels).sum().item()
        precision = hit / len(labels)
        precision_list.append(precision)

        with io.open(result_path + 'train_log.txt', 'a', encoding='utf-8') as file:
            file.write(
                'Train Epoch: {} [{}/{} ({:.0f}%)] | Loss: {:.4f} |  Precision: {:.4f} | Time:{} | Total_Time:{}\n'.format(
                    epoch, (batch_id + 1), len(trainLoader), 100. * (batch_id + 1) / len(trainLoader),
                    loss,
                    precision,
                    round((time.time() - batch_time), 4),
                    round((time.time() - start_time), 4)))

    if np.array(loss_list).mean() < bestLoss:
        bestLoss = np.array(loss_list).mean()
        flag = 1
        if opt.method == 3:
            torch.save(model_x.state_dict(), result_path + 'seg_bestLoss.pt')
            torch.save(model_y.state_dict(), result_path + 'org_bestLoss.pt')
            torch.save(model_xy.state_dict(), result_path + 'org_seg_bestLoss.pt')
        else:
            torch.save(model.state_dict(), result_path + 'bestLoss.pt')

    with io.open(result_path + 'train_epoch_log.txt', 'a', encoding='utf-8') as file:
        file.write('Epoch {}: Avg_loss: {:.4f} | Avg_precision: {:.4f}\n'.format(epoch, np.array(loss_list).mean(),
                                                                                 np.array(precision_list).mean()))

    # ------------------------------------ val -----------------------------------
    if opt.method == 3:
        modelList = [model_x, model_y, model_xy]
    else:
        modelList = [model]
    prediction, flag = val(opt, epoch, valLoader, modelList, flag, num_classes, result_path, bestValPred)
    if prediction > bestValPred:
        bestValPred = prediction
        if opt.method == 3:
            torch.save(model_x.state_dict(), result_path + 'org_bestVal.pt')
            torch.save(model_y.state_dict(), result_path + 'seg_bestVal.pt')
            torch.save(model_xy.state_dict(), result_path + 'org_seg_bestVal.pt')
#             pretrained_model_x = result_path + 'org_bestVal.pt'
#             pretrained_model_y = result_path + 'seg_bestVal.pt'
#             pretrained_model_xy = result_path + 'org_seg_bestVal.pt'
#             model_x1, model_y1, model_xy1 = Net.createModel(num_classes)
#             if opt.CUDA:
#                 model_x1, model_y1, model_xy1 = model_x1.cuda(), model_y1.cuda(), model_xy1.cuda()
#                 model_x1.load_state_dict(torch.load(pretrained_model_x))
#                 model_y1.load_state_dict(torch.load(pretrained_model_y))
#                 model_xy1.load_state_dict(torch.load(pretrained_model_xy))
#                 modelList = [model_x1, model_y1, model_xy1]
#                 val(opt, epoch, valLoader, modelList, flag, num_classes, result_path, bestValPred)[0]
        else:
            torch.save(model.state_dict(), result_path + 'bestVal.pt')
            

model_1 = Net.createModel(num_classes)
model_1.load_state_dict(torch.load(result_path + 'bestVal.pt'))
if opt.CUDA:
    model_1 = model_1.cuda()
modelList = [model_1]
val(opt, 1000, testLoader, modelList, flag, num_classes, result_path, bestValPred)

