import os
import torch
import shutil
from datetime import datetime
from constants import ConfName, args

class Logger:
    def __init__(self, name):
        if args.log_dir is None:
            self.log_dir = os.path.join('logs', name + '-' + datetime.now().strftime('%m-%d_%H:%M:%S'))
        else:
            self.log_dir = os.path.join(args.log_dir, name + '-' + datetime.now().strftime('%m-%d_%H:%M:%S'))
        os.makedirs(self.log_dir)
        self.log_file = open(os.path.join(self.log_dir, 'log.txt'), 'w')
        self.conf_file = open(os.path.join(self.log_dir, 'conf.txt'), 'w')
        self.res_file = open(os.path.join(self.log_dir, 'res.txt'), 'w')
        
        self.Log(['cur log dir ', self.log_dir])
        
        self.code_dir = self.log_dir + '/code'
        os.makedirs(self.code_dir)
        shutil.copy2('./models/models.py', self.code_dir + '/models.py')
        shutil.copy2('./configs/config.py', self.code_dir + '/config.py')
        shutil.copy2('./feature_store/feature_store.py', self.code_dir + '/feature_store.py')
        shutil.copy2('./main.py', self.code_dir + '/main.py')
        shutil.copy2('./constants.py', self.code_dir + '/constants.py')
        shutil.copy2('./dataset/dataset.py', self.code_dir + '/dataset.py')
    
    def Log(self, x):
        print(x)
        print(x, file=self.log_file)
    
    def LogPred(self, x, idx):
        print(x, file=open(os.path.join(self.log_dir, 'pred.txt' + str(idx)), 'w'))
    
    def LogConf(self, name, x):
        print('Conf Name:', name, file=self.conf_file)
        print(x, file=self.conf_file)
    
    def LogFinal(self, x):
        print(x)
        print(x, file=self.res_file)

    def DumpModel(self, model, feature, optimizer, idx):
        info = {'model_state_dict': model.state_dict(), 
            'optimizer': optimizer.state_dict(),
            'feature_optimizer': feature.optimizer.state_dict(),
            'feature_embedding': feature.embedding
            }
        torch.save(info, os.path.join(self.log_dir, 'best.pth' + str(idx)))
    
    def DumpModel_a(self, model, optimizer, idx):
        info = {'model_state_dict': model.state_dict(), 
            'optimizer': optimizer.state_dict()
            }
        torch.save(info, os.path.join(self.log_dir, 'best.pth' + str(idx)))

logger = Logger(ConfName)