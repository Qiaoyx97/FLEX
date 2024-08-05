import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import sys
sys.path.insert(0,'../')
sys.path.insert(0,'./')

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

def TrainStep(epoch, train_loader, model, optimizer, scheduler):
    model.train()
    all_pred, all_label, all_meta, loss_main, loss_meta = ModelEpoch(model, train_loader, optimizer)
    scheduler.step()
    acc, auc, f1, recall, precision, specificity, sensitivity = Calcmetrics(all_pred, all_label)
    logger.Log('epoch: {} ---- train acc: {}, auc: {}, f1: {}, recall: {}, precision: {}, sensitivity: {}, specificity: {}, loss_main: {}'.format(epoch, acc, auc, f1, recall, precision, sensitivity, specificity, np.mean(loss_main)))

    if conf['meta']:
        meta_acc, meta_auc, meta_f1, meta_recall, meta_precision, meta_specificity, meta_sensitivity = Calcmetrics(all_meta, all_label)
        logger.Log('epoch: {} ---- meta_train acc: {}, auc: {}, f1: {}, recall: {}, precision: {}, sensitivity: {}, specificity: {}, loss_meta: {}'.format(epoch, meta_acc, meta_auc, meta_f1, meta_recall, meta_precision, meta_sensitivity, meta_specificity, np.mean(loss_meta)))


def ValStep(best_auc, patience, epoch, valid_loader, model, optimizer, cur_data, loss_epo):
    model.eval()
    all_pred, all_label, all_meta, loss_main, loss_meta = ModelEpoch(model, valid_loader)
    acc, auc, f1, recall, precision, specificity, sensitivity = Calcmetrics(all_pred, all_label)
    logger.Log('epoch: {} ---- valid acc: {}, auc: {}, f1: {}, recall: {}, precision: {}, sensitivity: {}, specificity: {}, loss_main: {}'.format(epoch, acc, auc, f1, recall, precision, sensitivity, specificity, np.mean(loss_main)))

    if conf['meta']:
        meta_acc, meta_auc, meta_f1, meta_recall, meta_precision, meta_specificity, meta_sensitivity = Calcmetrics(all_meta, all_label)
        logger.Log('epoch: {} ---- meta_val acc: {}, auc: {}, f1: {}, recall: {}, precision: {}, sensitivity: {}, specificity: {}, loss_meta: {}'.format(epoch, meta_acc, meta_auc, meta_f1, meta_recall, meta_precision, meta_sensitivity, meta_specificity, np.mean(loss_meta)))
        auc = meta_auc

    if conf['meta']:
        loss_mean = np.mean(loss_meta)
    else:
        loss_mean = np.mean(loss_main)

    if loss_mean < loss_epo:
        best_auc = auc
        loss_epo = loss_mean
        patience = 0
        if conf['hash']:
            logger.DumpModel(model, feature_store, optimizer, cur_data)
        else:
            logger.DumpModel_a(model, optimizer, cur_data)
    else:
        patience += 1  
    return  best_auc, patience, loss_epo 


def TestStep(test_loader, model, cur_data):
    model_CKPT = torch.load(logger.log_dir+'/best.pth'+str(cur_data))
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
    auc_cur, meta_cur = [], []
    for i in range(5):
        logger.Log('cur_data {}'.format(i))
        cur_data = i
        conf['cur_data'] = i
        model = models_dic[conf['model_name']](conf)
        optimizer = conf['optimizer'](model.parameters(), lr=conf['learning_rate'], weight_decay=1e-8)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-8, verbose=True)
        
        Dataset = conf['dataset']
        BatchSize = conf['batch_size']
        train_set = Dataset('train', conf)
        train_loader = DataLoader(train_set, batch_size=BatchSize, shuffle=True, num_workers=0)
        valid_set = Dataset('val', conf)
        valid_loader = DataLoader(valid_set, batch_size=BatchSize, shuffle=False, num_workers=0)
        test_set = Dataset('test', conf)
        test_loader = DataLoader(test_set, batch_size=BatchSize, shuffle=False, num_workers=0)
        model = model.to(device)

        if device == 'cuda':
            model = torch.nn.DataParallel(model)
        best_auc, patience, loss_epo = 0, 0, 100
        for e in trange(conf['epochs']):
            TrainStep(e, train_loader, model, optimizer, scheduler)
            best_auc, patience, loss_epo = ValStep(best_auc, patience, e, valid_loader, model, optimizer, cur_data, loss_epo)
            if patience >= 10:
                break
        logger.Log('cur data {}, best auc {}'.format(cur_data, best_auc))
        test_auc, meta_auc = TestStep(test_loader, model, cur_data)
        auc_cur.append(test_auc.item())
        if conf['meta']:
            meta_cur.append(meta_auc.item())
        if conf['hash']:
            feature_store.Reset()
    logger.LogFinal('final auc: {}, meta auc: {}'.format(np.mean(auc_cur),np.mean(meta_cur)))
    return np.mean(auc_cur), np.mean(meta_cur)

if __name__ == '__main__':
    main()