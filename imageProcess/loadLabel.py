import scipy.io as sio
import numpy as np

def loadLabel(label_class, label_path):
    '''
    :param

    label_class: 0 for realphotos , 1 for haze_level

    :return

    label: store labels (shape:[1,number_of_photos])
    '''
    type = label_class
    if type == 0:
        label = sio.loadmat(label_path)
        label = label['label']
    elif type == 1:
        label_list = []
        with open(label_path, "r") as f:
            label_data = f.readlines()
            for line in label_data:
                line = line.strip('\n')
                level = int(line[-2:]) // 5 - 1
                label_list.append(level)
        label = np.array(label_list)
        label = label[np.newaxis, :]

    return label