import sys
sys.path.insert(0,'..')
from logger import logger
from constants import clinical_dir, args
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
import configs.config

def Kfold(dtype, conf):
    cur_data = conf['cur_data']
    data = pd.read_csv(clinical_dir, sep=',')
    id_list = [i for i in range(len(data))]
    y = data['ckdp'].tolist()
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.kflod_seed)
    for id, item in enumerate(kf.split(id_list, y)):
        if id == cur_data:
            train = item[0]
            test = item[1]
    y_train = []
    for i in train:
        y_train.append(y[i])
    train, val = train_test_split(train, random_state=args.kflod_seed, stratify=y_train, test_size=0.2)

    res = {
        'train': train,
        'val': val,
        'test': test
        }
    return res[dtype]

def Direct(dtype, conf):
    data = pd.read_csv(clinical_dir, sep=',')
    id_list = [i for i in range(len(data))]
    return id_list
