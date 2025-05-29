from logger import logger
import torch 
import torch.nn as nn
import torchvision.models as models
from constants import ConfName, args
from configs.config import config_dic
conf = config_dic[ConfName].config

models_dic = {}

class Image(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.model = models.resnet18(pretrained = True)
        fc_features = self.model.fc.in_features
        self.fc0 = nn.Linear(fc_features, 2)
        self.fc_img = nn.Linear(fc_features, 20)
        self.relu = torch.nn.ReLU(inplace=True)

    def Getlayer(self,x):
        x = x['image']
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        image = torch.flatten(x, 1)
        image = self.relu(self.fc_img(image))
        return image 


class Clin(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        struct_dims = conf['feature_nums_clin'] * args.embeding_fc_dims if args.independent_fc else conf['feature_nums_clin'] * conf['feature_dims']
        struct_dims_1 = 300
        self.emb_fcs = []
        for i in range(conf['feature_nums_clin']):
            exec("self.emb_fc_{} = nn.Linear({}, {})".format(i, conf['feature_dims'], args.embeding_fc_dims))
            exec("self.emb_fcs.append(self.emb_fc_{})".format(i))
        self.fc = nn.Linear(struct_dims, struct_dims_1)
        self.fc_struct = nn.Linear(struct_dims_1, 20)
    
    def Getlayer(self,x):
        embedding = x['C']
        if args.independent_fc:
            for i in range(len(embedding)):
                embedding[i] = self.emb_fcs[i](embedding[i])
        embedding = torch.cat(embedding, dim=1)
        struct = self.relu(self.fc(embedding))
        struct = self.relu(self.fc_struct(struct))
        return struct

class Prote(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        struct_dims = conf['feature_nums_proteomics'] * args.embeding_fc_dims if args.independent_fc else conf['feature_nums_proteomics'] * conf['feature_dims']
        struct_dims_1 = 300
        self.emb_fcs = []
        for i in range(conf['feature_nums_proteomics']):
            exec("self.emb_fc_{} = nn.Linear({}, {})".format(i, conf['feature_dims'], args.embeding_fc_dims))
            exec("self.emb_fcs.append(self.emb_fc_{})".format(i))
        self.fc = nn.Linear(struct_dims, struct_dims_1)
        self.fc_struct = nn.Linear(struct_dims_1, 20)
    
    def Getlayer(self,x):
        embedding = x['P']
        if args.independent_fc:
            for i in range(len(embedding)):
                embedding[i] = self.emb_fcs[i](embedding[i])
        embedding = torch.cat(embedding, dim=1)
        struct = self.relu(self.fc(embedding))
        struct = self.relu(self.fc_struct(struct))
        return struct


class Metabol(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        struct_dims = conf['feature_nums_metabolomics'] * args.embeding_fc_dims if args.independent_fc else conf['feature_nums_metabolomics'] * conf['feature_dims']
        struct_dims_1 = 300
        self.emb_fcs = []
        for i in range(conf['feature_nums_metabolomics']):
            exec("self.emb_fc_{} = nn.Linear({}, {})".format(i, conf['feature_dims'], args.embeding_fc_dims))
            exec("self.emb_fcs.append(self.emb_fc_{})".format(i))
        self.fc = nn.Linear(struct_dims, struct_dims_1)
        self.fc_struct = nn.Linear(struct_dims_1, 20)
    
    def Getlayer(self,x):
        embedding = x['M']
        if args.independent_fc:
            for i in range(len(embedding)):
                embedding[i] = self.emb_fcs[i](embedding[i])
        embedding = torch.cat(embedding, dim=1)
        struct = self.relu(self.fc(embedding))
        struct = self.relu(self.fc_struct(struct))
        return struct


class Meta(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        struct_dims = conf['feature_nums_clin']*conf['feature_dims']+conf['feature_nums_proteomics']*conf['feature_dims']+conf['feature_nums_metabolomics']*conf['feature_dims']
        # struct_dims = 20+conf['feature_nums_proteomics']*conf['feature_dims']+conf['feature_nums_metabolomics']*conf['feature_dims']
        self.fc_struct = nn.Linear(struct_dims, 20)
    
    def Getlayer(self,x):
        # img = x['image_emb']
        clin = torch.cat(x['C'], dim=1)
        proteomics = torch.cat(x['P'], dim=1)
        metabolomics = torch.cat(x['M'], dim=1)
        x = torch.cat([clin,proteomics,metabolomics], dim=1)
        struct = self.relu(self.fc_struct(x))
        return struct


class Merge(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.image_model = Image(conf).to('cuda')
        self.clin_model = Clin(conf).to('cuda')
        self.proteomics_model = Prote(conf).to('cuda')
        self.metabolomics_model = Metabol(conf).to('cuda')
        self.meta_model = Meta(conf).to('cuda')
        self.model_dic = {
            'image': self.image_model,
            'clin': self.clin_model,
            'proteomics': self.proteomics_model,
            'metabolomics': self.metabolomics_model,
            'meta': self.meta_model
        }
        self.modal_confs = conf['modal']
        self.meta_confs = conf['modal_meta']
        self.meta = conf['meta']
        self.fc_final_0 = nn.Linear(80, 20)
        self.fc_final = nn.Linear(20, 2)
        self.relu = torch.nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        meta_out = None
        if self.meta:
            meta_out = self.forward_meta(input)
        res = []
        for i in self.modal_confs:
            x = self.model_dic[i].Getlayer(input)
            res.append(x)
        res = self.relu(self.fc_final_0(torch.cat(res, dim=1)))
        res = self.fc_final(res)
        return {'pred': res, 'meta': meta_out}

    def forward_meta(self, input):
        res = []
        input['image_emb'] = self.model_dic['image'].Getlayer(input)
        for i in self.meta_confs:
            x = self.model_dic[i].Getlayer(input)
            res.append(x)
        res = self.relu(self.fc_final_0(torch.cat(res, dim=1)))
        res = self.fc_final(res)
        return res
models_dic['Merge'] = Merge