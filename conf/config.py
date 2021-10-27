# config.py
import os.path

# gets home dir cross platform
HOME = os.path.expanduser("~")

# different DATA set configs
CIFAR10 = {
    'num_classes':10,
    'max_epoch':100,
    'LR':1e-1,
    'lr_steps':(0,30,60,90),
    'AvgPool_InputSize':8,
    'PEDCC_Type':'center_pedcc/10_256_s.pkl',
    'feature_size':256,
    'batch_size':200,
    'name':'CIFAR10'
}
CIFAR100 = {
    'num_classes':100,
    'max_epoch':100,
    'LR':1e-1,
    'lr_steps':(0,30,60,90),
    'AvgPool_InputSize':8,
    'PEDCC_Type':'center_pedcc/100_512_s.pkl',
    'feature_size':512,
    'batch_size':128, #128
    'name':'CIFAR100'
}
TinyImageNet = {
    'num_classes':200,
    'max_epoch':100,
    'LR':1e-1,
    'lr_steps':(0,30,60,90),
    'AvgPool_InputSize':8,
    'PEDCC_Type':'center_pedcc/200_512_s.pkl',
    'feature_size':512,
    'batch_size':128,
    'name':'TinyImageNet'
}
Facescrubs = {
    'num_classes':100,
    'max_epoch':100,
    'LR':1e-1,
    'lr_steps':(0,30,60,90),
    'AvgPool_InputSize':8,
    'PEDCC_Type':'center_pedcc/100_512_s.pkl',
    'feature_size':512,
    'batch_size':128,
    'name':'Facescrubs'
}
Imagenet = {
    'num_classes':100,
    'max_epoch':100,
    'LR':1e-1,
    'lr_steps':(0,30,60,90),
    'AvgPool_InputSize':8,
    'PEDCC_Type':'center_pedcc/100_512_s.pkl',
    'feature_size':512,
    'batch_size':96,
    'name':'Imagenet'
}