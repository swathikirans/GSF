import os
import torch
import torchvision
import torchvision.datasets as datasets


ROOT_DATASET = './data'


def return_something_v1(split):
    root_data = 'something-v1/20bn-something-something-v1'
    filename_imglist_train = 'something-v1/train_videofolder.txt'
    filename_imglist_val = 'something-v1/val_videofolder.txt'
    prefix = '{:05d}.jpg'

    return filename_imglist_train, filename_imglist_val, root_data, prefix

def return_something_v2(split):
    root_data = 'something-v2/20bn-something-something-v2'
    filename_imglist_train = 'something-v2/train_videofolder.txt'
    filename_imglist_val = 'something-v2/val_videofolder.txt'
    prefix = '{:06d}.jpg'

    return filename_imglist_train, filename_imglist_val, root_data, prefix
    
def return_diving48(split):
    root_data = 'Diving48/frames'
    filename_imglist_train = 'Diving48/train_videofolder.txt'
    filename_imglist_val = 'Diving48/val_videofolder.txt'
    prefix = 'frame{:06d}.jpg'

    return filename_imglist_train, filename_imglist_val, root_data, prefix
    
def return_kinetics400(split):
    root_data = 'kinetics400'
    filename_imglist_train = 'kinetics400/train.txt'
    filename_imglist_val = 'kinetics400/val.txt'
    prefix = 'img_{:05d}.jpg'

    return filename_imglist_train, filename_imglist_val, root_data, prefix

def return_dataset(dataset, split):
    dict_single = {'something-v1': return_something_v1, 'something-v2': return_something_v2, 
                   'diving48':return_diving48, 'kinetics400': return_kinetics400,}
    if dataset in dict_single:
            file_imglist_train, file_imglist_val, root_data, prefix = dict_single[dataset](split)
    else:
        raise ValueError('Unknown dataset '+dataset)
    file_imglist_train = os.path.join(ROOT_DATASET, file_imglist_train)
    file_imglist_val = os.path.join(ROOT_DATASET, file_imglist_val)
    root_data = os.path.join(ROOT_DATASET, root_data)

    return file_imglist_train, file_imglist_val, root_data, prefix
