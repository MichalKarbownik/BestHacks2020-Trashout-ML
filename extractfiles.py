## paths will be train/cardboard, train/glass, etc...
import os

from main import split_indices, get_names, move_files

subsets = ['train', 'valid']
waste_types = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

## create destination folders for data subset and waste type
for subset in subsets:
    for waste_type in waste_types:
        folder = os.path.join('data', subset, waste_type)
        if not os.path.exists(folder):
            os.makedirs(folder)

if not os.path.exists(os.path.join('data', 'test')):
    os.makedirs(os.path.join('data', 'test'))

## move files to destination folders for each waste type
for waste_type in waste_types:
    source_folder = os.path.join('dataset-resized', waste_type)
    train_ind, valid_ind, test_ind = split_indices(source_folder, 1, 1)

    ## move source files to train
    train_names = get_names(waste_type, train_ind)
    train_source_files = [os.path.join(source_folder, name) for name in train_names]
    train_dest = "data\\train\\" + waste_type
    move_files(train_source_files, train_dest)

    ## move source files to valid
    valid_names = get_names(waste_type, valid_ind)
    valid_source_files = [os.path.join(source_folder, name) for name in valid_names]
    valid_dest = "data\\valid\\" + waste_type
    move_files(valid_source_files, valid_dest)

    ## move source files to test
    test_names = get_names(waste_type, test_ind)
    test_source_files = [os.path.join(source_folder, name) for name in test_names]
    ## I use data/test here because the images can be mixed up
    move_files(test_source_files, "data\\test")