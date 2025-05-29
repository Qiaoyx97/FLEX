import torch

class FeatureStore:
    def __init__(self):
        self.embedding = {}
        self.dims = {}
        self.hash_funcs = {}
        self.optimizer = None
    
    def Reset(self):
        self.embedding = {}
        self.dims = {}
        self.hash_funcs = {}
        self.optimizer = None
    
    def AddFeture(self, feature, dim, hash_func):
        assert feature not in self.embedding, "feature already added"
        self.embedding[feature] = {}
        self.dims[feature] = dim
        self.hash_funcs[feature] = hash_func
    
    def Add(self, feature, value):
        assert feature in self.embedding, "feature not found"
        value = self.hash_funcs[feature](value)
        if value not in self.embedding[feature]:
            self.embedding[feature][value] = torch.rand(self.dims[feature])
            self.embedding[feature][value].requires_grad = True
            if self.optimizer is None:
                self.optimizer = torch.optim.Adam([self.embedding[feature][value]], lr=1e-3)
            else:
                self.optimizer.add_param_group({'params':self.embedding[feature][value], 'lr': 1e-3})
    
    def Get(self, feature, value):
        assert feature in self.embedding, "feature not found"
        if self.hash_funcs[feature](value) not in self.embedding[feature]:
            self.Add(feature, value)
        value = self.hash_funcs[feature](value)
        return self.embedding[feature][value]

    def ZeroGrad(self):
        if self.optimizer is not None:
            self.optimizer.zero_grad()

    def Step(self):
        if self.optimizer is not None:
            self.optimizer.step()

    def Debug(self):
        print(self.embedding)
        for item in self.embedding:
            for it in self.embedding[item]:
                print(self.embedding[item][it].grad, self.embedding[item][it].grad_fn)

feature_store = FeatureStore()