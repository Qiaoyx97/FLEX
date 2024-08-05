import os
os.environ["CUDA_VISIBLE_DEVICES"]="4"

import sys
sys.path.insert(0,'../')
sys.path.insert(0,'./')
#修改dataset和distribution的constants_test
from constants import ConfName, args, seed
from logger import logger
from configs.config import config_dic
from utils import distribution,  hash_funcs
from utils.metrics import Calcmetrics
import dataset.dataset
import models.models
from models.models import models_dic

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import trange, tqdm
import sklearn

logger.Log('args: {}'.format(args))

conf = config_dic[ConfName].config
conf['seed'] = seed
feature_store = conf['feature_store']
device = conf['device']

loss_fn = conf['loss'].to(device)

def ModelEpoch(model, data_loader, optimizer=None):
    all_pred, all_label, all_meta = [], [], []
    loss_main, loss_meta = [], []
    for batch, item in tqdm(enumerate(data_loader)):
        y = item['label']
        pred = model(item)  
        print(pred)
        loss = loss_fn(pred['pred'], y.long())
        loss_main.append(loss.item())
            
        for i in y.detach().cpu().numpy():
            all_label.append(i)
        for i in torch.nn.Softmax(-1)(pred['pred']).detach().cpu().numpy():
            all_pred.append(i)

        if conf['meta']:
            loss = 0.45*(loss_fn(pred['pred'], y.long()))+0.55*(loss_fn(pred['meta'], y.long()))
            loss_meta.append(loss_fn(pred['meta'], y.long()).item())
            for i in torch.nn.Softmax(-1)(pred['meta']).detach().cpu().numpy():
                all_meta.append(i)

        if optimizer is not None:
            if conf['hash_train']:
                feature_store.ZeroGrad()
            optimizer.zero_grad()
            loss.backward()
                
            if conf['hash_train']:
                feature_store.Step()
            optimizer.step()
    return all_pred, all_label, all_meta, loss_main, loss_meta

def TestStep(test_loader, model):
    model_CKPT = torch.load('model_CKPT/best.pth2')
    model.load_state_dict(model_CKPT['model_state_dict'])
    model.eval()

    if conf['hash']:
        feature_store.embedding = model_CKPT['feature_embedding']
    meta_auc = 0

    all_pred, all_label, all_meta, loss_main, loss_meta = ModelEpoch(model, test_loader)
    acc, auc, f1, recall, precision, specificity, sensitivity = Calcmetrics(all_pred, all_label)
    logger.Log('test acc: {}, auc: {}, f1: {}, recall: {}, precision: {}, sensitivity: {}, specificity: {}'.format(acc, auc, f1, recall, precision, sensitivity, specificity))

    if conf['meta']:
        meta_acc, meta_auc, meta_f1, meta_recall, meta_precision, meta_specificity, meta_sensitivity = Calcmetrics(all_meta, all_label)
        logger.Log('meta_test acc: {}, auc: {}, f1: {}, recall: {}, precision: {}, sensitivity: {}, specificity: {}'.format(meta_acc, meta_auc, meta_f1, meta_recall, meta_precision, meta_sensitivity, meta_specificity))
    return auc, meta_auc


def main():
    model = models_dic[conf['model_name']](conf)
    optimizer = conf['optimizer'](model.parameters(), lr=conf['learning_rate'], weight_decay=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-8, verbose=True)
    
    Dataset = conf['dataset']
    BatchSize = conf['batch_size']
    _ = Dataset('train', conf)
    test_set = Dataset('test', conf)
    test_loader = DataLoader(test_set, batch_size=BatchSize, shuffle=False, num_workers=0)
    model = model.to(device)

    if device == 'cuda':
        model = torch.nn.DataParallel(model)
    
    test_auc, meta_auc = TestStep(test_loader, model)
    print(test_auc, meta_auc)

if __name__ == '__main__':
    main()