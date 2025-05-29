import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from logger import logger
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from glob import glob
import torch
from utils.transformer import GetTransformer
import configs.config
from constants import image_dir, proteomics_dir, metabolomics_dir, clinical_dir
from utils.distribution import *
from utils.hash_funcs import *
from tqdm import tqdm

class OmicDataset():
    def __init__(self, dtype, conf):
        self.conf = conf
        self.dtype = dtype
        self.conf['dtype'] = dtype
        self.device = conf['device']
        self.transformer = GetTransformer(dtype)
        self.feature_dims = conf['feature_dims']
        self.feature_store = conf['feature_store']
        self.proteomics_data = pd.read_csv(proteomics_dir, sep=',')
        self.feature_list_proteomics = self.proteomics_data.columns.tolist()[2:]
        self.metabolomics_data = pd.read_csv(metabolomics_dir, sep=',')
        self.feature_list_metabolomics = self.metabolomics_data.columns.tolist()[2:]
        self.clin_data = pd.read_csv(clinical_dir, sep=',')
        self.feature_list_clin = self.clin_data.columns.tolist()[2:]
        self.idx_list = self.GetIdList(dtype)
        self.pid_list = self.GetPidList()
        self.imgdir_list = self.GetImgDirList(dtype)
        if self.conf['hash']:
            if dtype == 'train' or  dtype == 'all':
                self.PrepareFeature()
    
    def GetPidList(self):
        res = []
        pid_list = self.clin_data['ID'].values.tolist()
        for i in self.idx_list:
            res.append(pid_list[i])
        return res

    def GetIdList(self,dtype):
        return self.conf['distribution'](dtype, self.conf)

    def GetImgDirList(self, dtype):
        imgdir_list = []
        for i in self.pid_list:
            imgdir_list.append(glob(image_dir+str(i)+'*')[0])
        return imgdir_list
    
    def GetImage(self, idx):
        assert self.transformer is not None, 'transform is not None'
        img = np.load(self.imgdir_list[idx])
        aug = self.transformer(image=img)
        img = torch.from_numpy(aug['image'])
        img = img.to(self.device)
        return img
    
    def GetLabel(self, idx):
        label = self.clin_data['ckdp'].loc[self.idx_list[idx]]
        label = torch.from_numpy(np.asarray(label).astype(float)).to(self.device)
        return label
    
    def GetPID(self, idx):
        PID = self.clin_data['ID'].loc[self.idx_list[idx]]
        return PID

    def PrepareFeature(self):
        Hash = self.conf['features_hash']
        for fea in self.feature_list_clin:
            self.feature_store.AddFeture(fea, self.feature_dims, Hash)
        for fea in self.feature_list_proteomics:
            self.feature_store.AddFeture(fea, self.feature_dims, Hash)
        for fea in self.feature_list_metabolomics:
            self.feature_store.AddFeture(fea, self.feature_dims, Hash)

    def GetClin(self, idx):
        res = []
        for fea in self.feature_list_clin:
            value = self.clin_data[fea].loc[self.idx_list[idx]]
            if self.conf['hash']:
                hash_value = self.feature_store.Get(fea, value)
                hash_value = hash_value.to(self.device)
                res.append(hash_value)
            else:
                res.append(torch.FloatTensor([value]).to(self.device))
        return res

    def GetProteomics(self, idx):
        res = []
        for fea in self.feature_list_proteomics:
            value = self.proteomics_data[fea].loc[self.idx_list[idx]]
            if self.conf['hash']:
                hash_value = self.feature_store.Get(fea, value)
                hash_value = hash_value.to(self.device)
                res.append(hash_value)
            else:
                res.append(torch.FloatTensor([value]).to(self.device))
        return res

    def GetMetabolomics(self, idx):
        res = []
        for fea in self.feature_list_metabolomics:
            value = self.metabolomics_data[fea].loc[self.idx_list[idx]]
            if self.conf['hash']:
                hash_value = self.feature_store.Get(fea, value)
                hash_value = hash_value.to(self.device)
                res.append(hash_value)
            else:
                res.append(torch.FloatTensor([value]).to(self.device))
        return res

    def __len__(self):
        return len(self.imgdir_list)

    def __getitem__(self, idx):
        return {'image':self.GetImage(idx), 'label':self.GetLabel(idx), 'ID':self.GetPID(idx), 'C': self.GetClin(idx), 'P':self.GetProteomics(idx), 'M':self.GetMetabolomics(idx)}