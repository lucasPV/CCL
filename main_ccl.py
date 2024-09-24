from __future__ import print_function

import os
import sys
import argparse
import time
import math
import gc
import random

import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader

from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model
from networks.resnet_big import SupConResNet
from losses_ccl import CCL_Loss
from PIL import Image

import numpy as np

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 


# Load data from npy
# these files will be correctly loaded in the main function
glob_labels = np.load("features_extracted/miniImagenet_tr20ts80_split0_labels.npy")
glob_features = np.load("features_extracted/miniImagenet_tr20ts80_split0_features.npy")
glob_rks = np.load("rks_extracted/miniImagenet_tr20ts80_split0_rks.npy", allow_pickle=True)

# Transfer global features data to device (GPU is default)
device = 'cuda:0'

# To store features from previous epochs
prev_feat = dict()

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        """
        Custom dataset loader that relies on data from npy files

        Args:
            image_paths (list): Lista de caminhos para as imagens.
            labels (list): Lista de rótulos para as imagens.
            transform (callable, optional): Uma função/transformação opcional a ser aplicada nas imagens.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')  # Garantir que a imagem seja lida em modo RGB
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label, idx


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=20,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'path', 'food101', 'miniImagenet'], help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')

    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR'], help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')
    parser.add_argument('--ckpt_filename', type=str, default='pretrain_ckpt',
                        help='name of the ckpt file to be saved')
    parser.add_argument('--load_ckpt_filename', type=str, default='pretrain_ckpt',
                        help='name of the ckpt file to be loaded')
    parser.add_argument('--npy_file', type=str, default='',
                        help='name of the npy file containing the training set paths and labels')
    parser.add_argument('--features_file', type=str, default='',
                        help='name of the npy features file')
    parser.add_argument('--labels_file', type=str, default='',
                        help='name of the npy labels file')
    parser.add_argument('--rks_file', type=str, default='',
                        help='name of the npy rks file')
    parser.add_argument('--k_loss', type=int, default=50,
                        help='initial k for the loss')

    opt = parser.parse_args()

    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_folder is not None \
            and opt.mean is not None \
            and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './datasets/'
    opt.model_path = './save/SupCon/{}_models'.format(opt.dataset)
    opt.tb_path = './save/SupCon/{}_tensorboard'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'.\
        format(opt.method, opt.dataset, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def set_loader(opt):
    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'food101':
        mean = (0.5458, 0.4443, 0.3442)
        std = (0.2328, 0.2439, 0.2426)
    elif opt.dataset == 'miniImagenet':
        mean = (0.4732, 0.4489, 0.4033)
        std = (0.2346, 0.2298, 0.2301)
    elif opt.dataset == 'path':
        mean = eval(opt.mean)
        std = eval(opt.std)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    if opt.dataset in ['cifar100', 'cifar10', 'miniImagenet', 'food101']:
        # Load data from .npy
        tmp_data = np.load(opt.npy_file, allow_pickle=True)
        paths = [elem[0] for elem in tmp_data]
        labels = [elem[1] for elem in tmp_data]
        # Convert labels to integer
        label_encoder = LabelEncoder()
        labels_int = label_encoder.fit_transform(labels)
        # Load dataset with custom loader
        train_dataset = CustomDataset(image_paths=paths,
                                      labels=labels_int,
                                      transform=TwoCropTransform(train_transform))
    else:
        raise ValueError(opt.dataset)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)

    return train_loader


def set_model(opt):
    model = SupConResNet(name=opt.model)
    criterion = CCL_Loss(temperature=opt.temp)

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    # load ckpt model
    print("Loading model {} ....".format(opt.load_ckpt_filename))
    pretrained_path = opt.load_ckpt_filename
    if pretrained_path is not None:
        checkpoint = torch.load(pretrained_path)
        if "model" in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)

    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (images, labels, indices) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        if opt.method == 'SupCon':
            loss = criterion(features, labels, indices=indices,
                             saved_features=glob_features,
                             saved_labels=glob_labels,
                             saved_rks=glob_rks,
                             epoch=epoch,
                             k_start=opt.k_loss,
                             total_epochs=opt.epochs)
        elif opt.method == 'SimCLR':
            loss = criterion(features)
        else:
            raise ValueError('contrastive method not supported: {}'.
                             format(opt.method))

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()

        # update saved features
        #prev_feat = dict()
        for i, ind in enumerate(indices):
            feat = features[i][0].detach().clone().cpu()
            prev_feat[ind.item()] = feat

    return losses.avg


def main():
    global glob_features
    global glob_labels
    global glob_rks

    opt = parse_option()

    # Load data from npy
    glob_labels = np.load(opt.labels_file)
    glob_features = np.load(opt.features_file)
    glob_rks = np.load(opt.rks_file, allow_pickle=True)

    # Transfer global features data to device
    device = 'cuda:0'
    glob_features = torch.from_numpy(glob_features)
    glob_features = glob_features.to(device) 

    # build data loader
    train_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        logger.log_value('loss', loss, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # update knn features for next epoch
        glob_features_cpu = glob_features.to('cpu').clone()
        glob_features = glob_features_cpu
        for key in prev_feat.keys():
            glob_features[key] = prev_feat[key]
        
        glob_features = glob_features.to(device)
        
        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, '{filename}_epoch_{epoch}.pth'.format(filename=opt.ckpt_filename, 
                                                                       epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)


if __name__ == '__main__':
    main()
