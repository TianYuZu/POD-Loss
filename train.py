import torch.utils.data as Data
from Resnet import ResNet, Bottleneck
import conf.config as conf
import argparse
import pickle
import os
import torchvision
import torchvision.transforms as transforms
import torch
import torch.optim as optim
from datetime import datetime
from tqdm import tqdm
from matplotlib import pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
device_ids = [0, 1]
def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")
parser = argparse.ArgumentParser(
    description='WideResnet Training With Pytorch')
parser.add_argument('--dataset', default='CIFAR10', choices=['CIFAR100', 'CIFAR100', 'TinyImageNet', 'Facescrubs', 'Imagenet'],
                    type=str, help='CIFAR10, CIFAR100, TinyImageNet, Facescrubs or Imagenet')
parser.add_argument('--dataset_root', default='',
                    help='Dataset root directory path')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--save_folder', default='./models/CIFAR10/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

CIFAR10_Train_ROOT = '/home/data/zhengguohui/Data/CIFAR10/train'
CIFAR10_Test_ROOT = '/home/data/zhengguohui/Data/CIFAR10/test'
CIFAR100_Train_ROOT = '/home/data/zhengguohui/Data/CIFAR100/train'
CIFAR100_Test_ROOT = '/home/data/zhengguohui/Data/CIFAR100/test'
TinyImageNet_Train_ROOT = '/home/data/zhengguohui/Data/TinyImageNet/train'
TinyImageNet_Test_ROOT = '/home/data/zhengguohui/Data/TinyImageNet/test'
Facescrubs_Train_ROOT = '/home/data/ZXW/Data/FaceScrubs/train_flip_64'
Facescrubs_Test_ROOT = '/home/data/ZXW/Data/FaceScrubs/test_flip_64'
Imagenet_Train_ROOT = '/home/data/Imagenet1000/100dir/train'
Imagenet_Test_ROOT = '/home/data/Imagenet1000/100dir/val'
# CIFAR10 data set
CIFAR10_train_data = torchvision.datasets.ImageFolder(CIFAR10_Train_ROOT,
    transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
)
CIFAR10_test_data = torchvision.datasets.ImageFolder(CIFAR10_Test_ROOT,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
)
# CIFAR100 data set
CIFAR100_train_data = torchvision.datasets.ImageFolder(CIFAR100_Train_ROOT,
    transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
    ])
)
CIFAR100_test_data = torchvision.datasets.ImageFolder(CIFAR100_Test_ROOT,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
    ])
)
# TinyImageNet data set
TinyImageNet_train_data = torchvision.datasets.ImageFolder(TinyImageNet_Train_ROOT,
    transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=64, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
    ])
)
TinyImageNet_test_data = torchvision.datasets.ImageFolder(TinyImageNet_Test_ROOT,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
    ])
)
# Facescrubs data set
Facescrubs_train_data = torchvision.datasets.ImageFolder(Facescrubs_Train_ROOT,
    transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=64, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
    ])
)
Facescrubs_test_data = torchvision.datasets.ImageFolder(Facescrubs_Test_ROOT,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
    ])
)
# Imagenet data set
Imagenet_train_data = torchvision.datasets.ImageFolder(Imagenet_Train_ROOT,
    transform=transforms.Compose([
        transforms.RandomResizedCrop(size=224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
)
Imagenet_test_data = torchvision.datasets.ImageFolder(Imagenet_Test_ROOT,
    transform=transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
)


def read_pkl(PEDCC_PATH):
    #pedcc_path = os.path.join(conf.HOME, PEDCC_PATH)
    f = open(PEDCC_PATH, 'rb')
    a = pickle.load(f)
    f.close()
    return a


def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().item()
    return num_correct / total


def draw(history):
    epochs = range(1, len(history['loss_train']) + 1)
    plt.plot(epochs, history['loss_train'], 'blue', label='Training loss')
    plt.plot(epochs, history['loss_val'], 'r', label='Validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./Training and Validation loss.jpg')
    plt.figure()
    epochs = range(1, len(history['acc_train']) + 1)
    plt.plot(epochs, history['acc_train'], 'b', label='Training acc')
    plt.plot(epochs, history['acc_val'], 'r', label='Validation acc')
    plt.title('Training and validation acc')
    plt.xlabel('Epochs')
    plt.ylabel('acc')
    plt.legend()
    plt.savefig('./Training and validation acc.jpg')
    plt.show()

    
def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


#####NaCLoss##################
def NaCLoss(input, target, delta):
    ret_before = input * target
    ret_before = torch.sum(ret_before, dim=1).view(-1, 1)

    add_feature = delta * torch.ones((input.shape[0], 1)).cuda()
    input_after = torch.cat((input, add_feature), dim=1)
    input_after_norm = torch.norm(input_after, p=2, dim=1, keepdim=True)

    ret = ret_before / input_after_norm
    ret = 1 - ret
    ret = ret.pow(2)
    ret = torch.mean(ret)
    
    return ret


#####SCLoss#########################
def SCLoss(map_PEDCC, label, feature):
    average_feature = map_PEDCC[label.long().data].float().cuda()
    feature_norm = l2_norm(feature)
    feature_norm = feature_norm - average_feature
    covariance100 = 1 / (feature_norm.shape[0] - 1) * torch.mm(feature_norm.T, feature_norm).float()
    covariance100_loss = torch.sum(pow(covariance100, 2)) - torch.sum(pow(torch.diagonal(covariance100), 2))
    covariance100_loss = covariance100_loss / (covariance100.shape[0] - 1)
    return covariance100_loss


def train_start(net, train_data, valid_data, cfg, save_folder, classes_num):
    LR = cfg['LR']
    if torch.cuda.is_available():
        net = torch.nn.DataParallel(net, device_ids=device_ids)
        net = net.cuda()
    prev_time = datetime.now()
    map_dict = read_pkl(cfg['PEDCC_Type'])

    history = dict()
    loss_train = []
    loss_val = []
    acc_train = []
    acc_val = []

    map_PEDCC = torch.Tensor([])
    for i in range(classes_num):
        map_PEDCC = torch.cat((map_PEDCC, map_dict[i].float()), 0)
    map_PEDCC = map_PEDCC.view(classes_num, -1)  # (class_num, dimension)

    delta = 0.05
    alpha = 0.01
    for epoch in range(cfg['max_epoch']):
        if epoch in cfg['lr_steps']:
            if epoch != 0:
                LR *= 0.1
            optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        train_loss = 0
        train_acc = 0
        length, num = 0, 0
        length_test = 0
        net = net.train()
        
        l2_norm_trainSample = torch.Tensor([]).cuda()
        for im, label in tqdm(train_data):
            if torch.cuda.is_available():
                im = im.cuda()  # (bs, 3, h, w)
                label = label.cuda()  # (bs, h, w)
                tensor_empty1 = map_PEDCC[label].float().cuda()
                label_mse_tensor = tensor_empty1.view(label.shape[0], -1)  # (batchSize, dimension)
                label_mse_tensor = label_mse_tensor.cuda()
            output, output2 = net(im)
            with torch.no_grad():
                l2_norm_trainSample = torch.cat((l2_norm_trainSample, torch.norm(output, p=2, dim=1), dim=0)
            loss1 = NaCLoss(output2, label_mse_tensor, delta)
            loss2 = SCLoss(map_PEDCC, label, output2)
            loss = loss1 + loss2
            length += output.pow(2).sum().item()
            num += output.shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.data
            train_acc += get_acc(output, label)
        delta = alpha * (epoch+1) * torch.mean(l2_norm_trainSample).data                                   
        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        if valid_data is not None:
            valid_loss = 0
            valid_acc = 0
            net = net.eval()
            for im, label in valid_data:
                if torch.cuda.is_available():
                    im = im.cuda()
                    label = label.cuda()
                output = net(im)[0]
                loss = criterion2(output, label)
                valid_loss += loss.data
                valid_acc += get_acc(output, label)
                length_test += output.pow(2).sum().item()/im.shape[0]
            epoch_str = (
                "Epoch %d. Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f, LR: %f, length: %f, length_test: %f"
                % (epoch, train_loss / len(train_data),
                   train_acc / len(train_data), valid_loss / len(valid_data),
                   valid_acc / len(valid_data), LR, length/num, length_test/len(valid_data)))
            loss_train.append(train_loss / len(train_data))
            loss_val.append(valid_loss / len(valid_data))
            acc_train.append(train_acc / len(train_data))
            acc_val.append(valid_acc / len(valid_data))
        else:
            epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " %
                         (epoch, train_loss / len(train_data),
                          train_acc / len(train_data)))
        prev_time = cur_time
        print(epoch_str + time_str)
        f = open(save_folder+'result.txt', 'a+')
        f.write(epoch_str + time_str + '\n')
        f.close()
        if (epoch+1) % 10 == 0:
            torch.save(net.state_dict(), save_folder + 'resnet' + str(epoch+1) + '_epoch.pth')

    history['loss_train'] = loss_train
    history['loss_val'] = loss_val
    history['acc_train'] = acc_train
    history['acc_val'] = acc_val
    draw(history)


def train():
    if args.dataset == 'CIFAR10':
        train_data = CIFAR10_train_data
        test_data = CIFAR10_test_data
        cfg = conf.CIFAR10
    elif args.dataset == 'CIFAR100':
        train_data = CIFAR100_train_data
        test_data = CIFAR100_test_data
        cfg = conf.CIFAR100
    elif args.dataset == 'TinyImageNet':
        train_data = TinyImageNet_train_data
        test_data = TinyImageNet_test_data
        cfg = conf.TinyImageNet
    elif args.dataset == 'Facescrubs':
        train_data = Facescrubs_train_data
        test_data = Facescrubs_test_data
        cfg = conf.Facescrubs
    elif args.dataset == 'Imagenet':
        train_data = Imagenet_train_data
        test_data = Imagenet_test_data
        cfg = conf.Imagenet
    else:
        print("dataset doesn't exist!")
        exit(0)

    cnn1 = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=cfg['num_classes'], feature_size=cfg['feature_size'], pedcc_type=cfg['PEDCC_Type'])
    train_loader = Data.DataLoader(dataset=train_data, batch_size=cfg['batch_size'], shuffle=True, num_workers=6, pin_memory=True)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=cfg['batch_size'], shuffle=False, num_workers=6, pin_memory=True)
    # start training
    train_start(cnn1, train_loader, test_loader, cfg, args.save_folder, cfg['num_classes'])

if __name__ == '__main__':
    train()
