from fastai.vision import *
from fastai.metrics import error_rate
from pathlib import Path

import numpy as np
import os
import shutil
import re

def split_indices(folder, seed1, seed2):
    n = len(os.listdir(folder))
    full_set = list(range(1, n + 1))

    random.seed(seed1)
    train = random.sample(list(range(1, n + 1)), int(.5 * n))

    remain = list(set(full_set) - set(train))

    random.seed(seed2)
    valid = random.sample(remain, int(.5 * len(remain)))
    test = list(set(remain) - set(valid))

    return (train, valid, test)

def get_names(waste_type, indices):
    file_names = [waste_type + str(i) + ".jpg" for i in indices]
    return (file_names)

def move_files(source_files, destination_folder):
    for file in source_files:
        shutil.move(file, destination_folder)

if __name__ == '__main__':

    subsets = ['train', 'valid']
    waste_types = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

    path = Path(os.getcwd())/"data"
    print(path)

    tfms = get_transforms(do_flip=True, flip_vert=True)
    data = ImageDataBunch.from_folder(path, test="test", ds_tfms=tfms, bs=16)

    print(data)
    print(data.classes)

    data.show_batch(rows=4,figsize=(10,8))
    learn = create_cnn(data,models.resnet34,metrics=error_rate)

    print(learn.model)

    learn.lr_find(start_lr=1e-6,end_lr=1e1)
    learn.recorder.plot()

    learn.fit_one_cycle(1,max_lr=5.13e-03)

    interp = ClassificationInterpretation.from_learner(learn)
    losses,idxs = interp.top_losses()

    interp.plot_top_losses(9, figsize=(15,11))

    interp.most_confused(min_val=2)

    preds = learn.get_preds(ds_type=DatasetType.Test)

    print(preds[0].shape)
    print(preds[0])

    print(data.classes)

    max_idxs = np.asarray(np.argmax(preds[0],axis=1))

    yhat = []
    for max_idx in max_idxs:
        yhat.append(data.classes[max_idx])

    print(yhat)

    y = []

    for label_path in data.test_ds.items:
        y.append(str(label_path))

    pattern = re.compile("([a-z]+)[0-9]+")
    for i in range(len(y)):
        y[i] = pattern.search(y[i]).group(1)

    print(yhat[0:5])
    print(y[0:5])

    learn.export(file='D:\PyCharm\PyCharm Projects\Wastes\model\export.pkl')

