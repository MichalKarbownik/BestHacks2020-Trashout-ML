from fastai.vision import *

if __name__ == '__main__':
    path = os.getcwd()

    test = ImageList.from_folder(Path(path + '\data\\testing'))

    learner = load_learner(Path(path), file='new_model.pkl', test=test)

    preds = learner.get_preds(ds_type=DatasetType.Test)

    wastes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

    print(preds)
    print(wastes)
