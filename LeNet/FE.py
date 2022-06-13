import torch
import torch.nn as nn
from collections import OrderedDict


def FE_LeNet_1(number_of_class):
    # define net
    class C1(nn.Module):
        def __init__(self):
            super(C1, self).__init__()

            self.c1 = nn.Sequential(OrderedDict([
                ('c1', nn.Conv2d(3, 6, kernel_size=(5, 5))),
                ('relu1', nn.ReLU()),
                ('s1', nn.MaxPool2d(kernel_size=(2, 2), stride=2))
            ]))

        def forward(self, img):
            output = self.c1(img)
            return output  # torch.Size([32, 6, 30, 30])

    class C2(nn.Module):
        def __init__(self):
            super(C2, self).__init__()

            self.c2 = nn.Sequential(OrderedDict([
                ('c2', nn.Conv2d(6, 16, kernel_size=(5, 5))),
                ('relu2', nn.ReLU()),
                ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2))
            ]))

        def forward(self, img):
            output = self.c2(img)
            return output  # torch.Size([32, 16, 13, 13])

    class C3(nn.Module):
        def __init__(self):
            super(C3, self).__init__()

            self.c3 = nn.Sequential(OrderedDict([
                ('c3', nn.Conv2d(16, 120, kernel_size=(5, 5))),
                ('relu3', nn.ReLU())
            ]))

        def forward(self, img):
            output = self.c3(img)
            return output  # torch.Size([32, 120, 9, 9])

    class LeNet5(nn.Module):
        """
        Input - 3x64x64
        Output - 9
        """

        def __init__(self):
            super(LeNet5, self).__init__()

            self.c1 = C1()
            self.c2_1 = C2()
            self.c2_2 = C2()
            self.c3 = C3()

        def forward(self, img):
            output = self.c1(img)

            x = self.c2_1(output)
            output = self.c2_2(output)

            output += x

            output = self.c3(output)
            output = output.view(img.size(0), -1)
            # output = self.f6(output)
            # output = self.f4(output)
            # output = self.f5(output)
            return output

    model = LeNet5()
    return model


def FE_LeNet_2(number_of_class):
    # define net

    class F4(nn.Module):
        def __init__(self):
            super(F4, self).__init__()

            self.f4 = nn.Sequential(OrderedDict([
                ('f4', nn.Linear(120, 84)),
                ('relu4', nn.ReLU())
            ]))

        def forward(self, img):
            output = self.f4(img)
            return output

    class F5(nn.Module):
        def __init__(self):
            super(F5, self).__init__()

            self.f5 = nn.Sequential(OrderedDict([
                ('f5', nn.Linear(84, number_of_class)),
                ('sig5', nn.LogSoftmax(dim=-1))
            ]))

        def forward(self, img):
            output = self.f5(img)
            return output

    class F6(nn.Module):
        def __init__(self):
            super(F6, self).__init__()

            self.f6 = nn.Sequential(OrderedDict([
                ('f6', nn.Linear(120 * 9 * 9, 120)),
                ('relu6', nn.ReLU())
            ]))

        def forward(self, img):
            output = self.f6(img)
            return output

    class F7(nn.Module):
        def __init__(self):
            super(F7, self).__init__()

            self.f7 = nn.Sequential(OrderedDict([
                ('f7', nn.Linear(120 * 9 * 9 * 2, 120 * 9 * 9)),
                ('relu7', nn.ReLU())
            ]))

        def forward(self, img):
            output = self.f7(img)
            return output

    class LeNet5(nn.Module):
        """
        Input - 3x64x64
        Output - 9
        """

        def __init__(self):
            super(LeNet5, self).__init__()

            self.f7 = F7()
            self.f6 = F6()
            self.f4 = F4()
            self.f5 = F5()

        def forward(self, img):
            output = self.f7(img)
            output = self.f6(output)
            output = self.f4(output)
            output = self.f5(output)
            return output

    model = LeNet5()
    return model


def createModel(number_of_class):
    model_x = FE_LeNet_1(number_of_class)
    model_y = FE_LeNet_1(number_of_class)
    model_xy = FE_LeNet_2(number_of_class)
    return model_x, model_y, model_xy
