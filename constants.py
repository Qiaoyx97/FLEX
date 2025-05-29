import torch
import numpy as np
import random
import argparse
parser = argparse.ArgumentParser(description='cmd args')
parser.add_argument('--seed', default=363144, type=int, help ='seed')
parser.add_argument('--kflod_seed', default=925677, type=int, help ='kflod seed')
parser.add_argument('--log_dir', default=None, type=str, help ='log日志的根目录')
parser.add_argument('--embeding_fc_dims', default=1, type=int, help ='embeding单独fc的输出维度')
parser.add_argument('-ic', '--independent_fc', action="store_true", help ='embeding是否过单独fc')

args = parser.parse_args()

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
     
seed = args.seed
setup_seed(seed)

image_dir = 'sample/images/'
proteomics_dir = 'sample/proteomics.csv'
metabolomics_dir = 'sample/metabolomics.csv'
clinical_dir = 'sample/clinical.csv'


ConfName = 'Merge'
