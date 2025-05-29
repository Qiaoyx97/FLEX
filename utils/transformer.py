import albumentations as albu
import numpy as np

def to_rbg(x, **kwargs):
    x = x[:,:,:3]
    x = x / 255.0
    x = x.astype(np.float32)
    return x

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def GetTransformer(dtype):
    transform = {'train':[
        albu.Lambda(image=to_rbg),
        albu.Lambda(image=to_tensor)
    ],
    'val':[
        albu.Lambda(image=to_rbg),
        albu.Lambda(image=to_tensor)
    ],
    'test':[
        albu.Lambda(image=to_rbg),
        albu.Lambda(image=to_tensor)
    ],
    'all':[
        albu.Lambda(image=to_rbg),
        albu.Lambda(image=to_tensor)
    ]}
    return albu.Compose(transform[dtype])