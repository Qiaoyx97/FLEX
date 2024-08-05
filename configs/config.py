from logger import logger
import torch
from dataset.dataset import *
from utils.hash_funcs import *
from feature_store.feature_store import feature_store

config_dic = {}
class Config:
    check_list = ['dataset']
    def __init__(self, name, config):
        for item in self.check_list:
            assert item in config, 'invaild, {} not in config'.format(item)
        self.config = config
        self.name = name
        config_dic[name] = self
    def dump(self):
        logger.LogConf(self.name, self.config)

config = {
    'dataset': OmicDataset,
    'model_name': 'Merge',
    'batch_size': 8,
    'classes': 2,
    'epochs': 100,
    'learning_rate': 1e-3,
    'loss': torch.nn.CrossEntropyLoss(weight = torch.FloatTensor([1, 1.44])),
    'device': 'cuda',
    'optimizer': torch.optim.Adam,
    'distribution': Direct,
    'feature_dims': 10,
    'feature_nums_clin': 30,
    'feature_nums_proteomics': 30,
    'feature_nums_metabolomics': 30,
    'features_hash': LOG,
    'feature_store': feature_store,
    'hash': True,
    'hash_train': True,
    'meta': False,
    'modal':[
        'image',
        'clin',
        'proteomics',
        'metabolomics'
    ],
    'modal_meta':[
        # 'meta',
        # 'image',
        # 'clin',
        # 'proteomics',
        # 'metabolomics',
    ]
}
Config('Merge', config).dump()

