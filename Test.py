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

from datasets.Level import Haze_Level
from datasets.Wild import Haze_Wild
from utils import *

# Training settings
parser = argparse.ArgumentParser(description="PyTorch Train")
parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate, default=1e-4")
parser.add_argument("--batchSize", type=int, default=16, help="Training batch size")
parser.add_argument("--testBatchSize", type=int, default=1, help="Test batch size")
parser.add_argument("--epochs", type=int, default=151, help="Train epoch")
parser.add_argument("--decay", type=int, default=50, help="Change the learning rate for every N epochs")
parser.add_argument("--decay_rate", type=float, default=0.5, help="Decay scale of learning rate, default=0.5")
parser.add_argument('--dataset', default="wild", type=str, help='hazel_level or haze_wild')
parser.add_argument('--model', default='ResNet', type=str, help='select ResNet or LeNet as backbone')
parser.add_argument('--name', default='base', type=str, help='Filename of the training models')
parser.add_argument('--crop_size', type=int, help='Set the crop_size', default=[64, 64], nargs='+')
parser.add_argument('--filters', type=int, help='Set the size of filters of ucm', default=[40, 40, 70], nargs='+')
parser.add_argument("--method", type=int, default=0, help="0->base, 1->IA, 2->NC, 3->FE")
parser.add_argument("--CUDA", type=int, default=1, help="1 for using CUDA, 0 for not")
parser.add_argument("--use_lr_decay", type=bool, default=True, help="use decay")
parser.add_argument("--pretrained", type=bool, default=True, help="use pretrained model")
parser.add_argument("--dataType", type=int, default=0, help="0->original_64, 1->seg_original_64, 2->mask_original_64")

opt = parser.parse_args()
# opt.seed = 1010
opt.data_name = ['original_64', 'seg_original_64', 'mask_original_64']
methods = ['Base', 'IA', 'NC', 'FE']
data_type = ['', 'S', 'F']
model_path = './model/' + methods[opt.method] + '_' + data_type[opt.dataType] + '/' + opt.dataset + '/' + opt.model + '/'
result_path = './result/' + methods[opt.method] + '_' + data_type[opt.dataType] + '_' + opt.dataset + '_' + opt.model + '/'
if not os.path.exists(result_path):
    os.makedirs(result_path)
Net = import_module(opt.model + '.' + opt.name)

# torch.manual_seed(opt.seed)
# if opt.CUDA:
#     torch.cuda.manual_seed(opt.seed)

if opt.dataset is 'level':
    num_classes = 9
    data_path = './data/'
    test_datasets = Haze_Level(data_path, opt.crop_size, opt.filters, method=opt.method, data_name=opt.data_name[opt.dataType], mode='val')
    testLoader = DataLoader(dataset=test_datasets, batch_size=opt.testBatchSize, shuffle=False, num_workers=4)
else:
    num_classes = 10
    data_path = './data/'
    # images_path, labels = getPath(data_path)
    test_datasets = Haze_Wild(data_path, opt.crop_size, opt.filters, method=opt.method, data_name=opt.data_name[opt.dataType], mode='test')
    testLoader = DataLoader(dataset=test_datasets, batch_size=opt.testBatchSize, shuffle=False, num_workers=4)

if opt.method != 3:
    model = Net.createModel(num_classes)
    if opt.pretrained:
        pretrained_model = model_path + 'bestVal.pt'
        model.load_state_dict(torch.load(pretrained_model))
    if opt.CUDA:
        model = model.cuda()
else:
    model_x, model_y, model_xy = Net.createModel(num_classes)
    if opt.pretrained:
        pretrained_model_x = model_path + 'org_bestVal.pt'
        pretrained_model_y = model_path + 'seg_bestVal.pt'
        pretrained_model_xy = model_path + 'org_seg_bestVal.pt'
        model_x.load_state_dict(torch.load(pretrained_model_x))
        model_y.load_state_dict(torch.load(pretrained_model_y))
        model_xy.load_state_dict(torch.load(pretrained_model_xy))
    if opt.CUDA:
        model_x, model_y, model_xy = model_x.cuda(), model_y.cuda(), model_xy.cuda()

error = nn.CrossEntropyLoss()
if opt.CUDA:
    error = error.cuda()

bestLoss = 100
bestValPred = 0
flag = 0
precision_list = []

# -------------------------- test ------------------------------
total = 0
correct = 0
test_predictions_list = np.empty(0)
test_labels = np.empty(0)
confusion_matrix = np.zeros((num_classes, num_classes))
if opt.method != 3:
    model.eval()
else:
    model_x.eval()
    model_y.eval()
    model_xy.eval()

for images, labels in testLoader:
    if opt.CUDA:
        images, labels = images.cuda(), labels.cuda()

    test = Variable(images)
    if opt.method == 3:
        seg_img = images[:, 0:3, :, :]
        org_img = images[:, 3:, :, :]
        org_outputs = model_x(org_img)
        seg_outputs = model_y(seg_img)
        xy = torch.cat([org_outputs, seg_outputs], dim=1)
        outputs = model_xy(xy)
    elif opt.method == 2:
        outputs, _ = model(test)
    else:
        outputs = model(test)
    test_predictions = torch.max(outputs, 1)[1]
    correct += (test_predictions == labels).sum()
    total += len(labels)
    test_predictions = np.array(test_predictions.cpu())
    labels = np.array(labels.cpu())
    test_predictions_list = np.append(test_predictions_list, test_predictions)
    test_labels = np.append(test_labels, labels)

test_predictions_list = np.array(test_predictions_list)
labels = np.array(test_labels)
for i in range(num_classes):
    position = test_predictions_list[np.where(labels == i)]
    for j in position:
        confusion_matrix[i][int(j)] += 1
np.savetxt(result_path + 'test_confusion_matrix.txt', confusion_matrix, fmt="%d", delimiter=" ")

prediction = torch.true_divide(correct * 100, total)
with io.open(result_path + 'test.txt', 'a', encoding='utf-8') as file:
    file.write(
        "Correct: {}, Total: {}, Accuracy: {}%\n".format(correct, total, prediction))

