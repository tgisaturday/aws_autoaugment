import copy
import os
import sys
import time
from collections import OrderedDict, defaultdict
import logging
import torch
import argparse
import numpy as np
import math
import torch.nn as nn
import itertools
import operator
import torch.backends.cudnn as cudnn
from tqdm import tqdm

#from FastAutoAugment.archive import remove_deplicates, policy_decoder
from augmentations import augment_list
from common import get_logger, add_filehandler
from data import get_dataloaders
from metrics import *
from networks import get_model, num_class
from PPO import Controller


class Memory:
    def __init__(self, path):
        self.file = path
        
    def add(self, index):
        if self.file == None:
            return
        fp = open(self.file, 'a')
        print(index,file=fp)
        fp.close()

    def dump(self):
        if self.file == None:
            return        
        action_index = []
        fp = open(self.file,'r')
        actions = fp.readlines()
        for action in actions:
            try:
                action_index.append(int(action))
            except:
                logger.info('invalid action string, skipping')
        return action_index
    
    def reset(self):
        if self.file == None:
            return        
        fp = open(self.file, 'w')        
        fp.close()


logger = get_logger('AWS AutoAugment')

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='./results')
parser.add_argument('--dataroot', type=str, default='./data')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--checkpoint',type=str)
parser.add_argument('--manual_seed', default=0, type=int)

parser.add_argument('--policy_checkpoint', type=str)
""" run config """
parser.add_argument('--init_lr', type=float, default=0.4)
parser.add_argument('--n_epochs', type=int, default=300)
parser.add_argument('--lr_schedule', type=str, default='cosine')
parser.add_argument('--cutout', type=int, default=16)
parser.add_argument('--label_smoothing', type=float, default=0.0)
parser.add_argument('--enlarge_batch', action='store_true')
parser.add_argument('--enlarge_batch_size',type=int, default=8)


parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar10', 'cifar100','imagenet'])
parser.add_argument('--model', type=str, default='wresnet28_10', choices=['wresnet28_10','shakeshake26_2x32d','pyramid'])
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--valid_size', type=int, default=10000)

parser.add_argument('--opt_type', type=str, default='sgd', choices=['sgd'])
parser.add_argument('--momentum', type=float, default=0.9)  # opt_param
parser.add_argument('--no_nesterov', action='store_true')  # opt_param
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--n_worker', type=int, default=16)

""" controller hyper-parameters """
parser.add_argument('--policy_embedding_size', type=int, default=32)
parser.add_argument('--policy_hidden_size', type=int, default=100)

        
def run_epoch(model, loader, loss_fn, optimizer, max_epoch, desc_default='', epoch=0):

    metrics = Accumulator()
    cnt = 0
    total_steps = len(loader)
    steps = 0
    for data, label in loader:
        steps += 1
        data, label = data.cuda(), label.cuda()

        if optimizer:
            optimizer.zero_grad()

        preds = model(data)
        loss = loss_fn(preds, label)

        if optimizer:
            loss.backward()
            optimizer.step()

        top1, top5 = accuracy(preds, label, (1, 5))
        metrics.add_dict({
            'loss': loss.item() * len(data),
            'top1': top1.item() * len(data),
            'top5': top5.item() * len(data),
        })
        cnt += len(data)

        del preds, loss, top1, top5, data, label
        
    if optimizer:
        logger.info('[%s %03d/%03d] %s lr=%.6f', desc_default, epoch+1, max_epoch, metrics / cnt, optimizer.param_groups[0]['lr'])
    else:
        logger.info('[%s %03d/%03d] %s', desc_default, epoch+1, max_epoch, metrics / cnt)

    metrics /= cnt
    if optimizer:
        metrics.metrics['lr'] = optimizer.param_groups[0]['lr']
    return metrics

def run_epoch_with_EB(model, loader, loss_fn, optimizer, max_epoch, desc_default='', epoch=0):

    metrics = Accumulator()
    cnt = 0
    total_steps = len(loader)
    steps = 0
    for data, label in loader:
        steps += 1
        label = label.cuda()
        if optimizer:
            optimizer.zero_grad()
        losses = []
        for i, datum in enumerate(data):
            datum = datum.cuda()
            preds = model(datum)
            loss = loss_fn(preds, label)
            losses.append(loss)
        loss = torch.mean(torch.stack(losses))

        if optimizer:
            loss.backward()
            optimizer.step()

        top1, top5 = accuracy(preds, label, (1, 5))
        metrics.add_dict({
            'loss': loss.item() * len(data),
            'top1': top1.item() * len(data),
            'top5': top5.item() * len(data),
        })
        cnt += len(data)

        del preds, loss, top1, top5, data, label
        
    if optimizer:
        logger.info('[%s %03d/%03d] %s lr=%.6f', desc_default, epoch+1, max_epoch, metrics / cnt, optimizer.param_groups[0]['lr'])
    else:
        logger.info('[%s %03d/%03d] %s', desc_default, epoch+1, max_epoch, metrics / cnt)

    metrics /= cnt
    if optimizer:
        metrics.metrics['lr'] = optimizer.param_groups[0]['lr']
    return metrics
def train_and_eval(args, model, epoch, policy, test_ratio=0.0, cv_fold=0, metric='last', EB=False):

    max_epoch = epoch
    trainsampler, trainloader, validloader, testloader_ = get_dataloaders(args, policy, test_ratio, split_idx=cv_fold, EB=EB)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.init_lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=not args.no_nesterov,
        )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch)
    start_epoch = 0
    if args.resume:
        logger.info('Loading pretrained weights')
        checkpoint = torch.load(args.checkpoint)
        if args.num_gpu > 1:
            model.module.load_state_dict(checkpoint['model_state_dict'])                  
        else:
            model.load_state_dict(checkpoint['model_state_dict'])      
        optimizer.load_state_dict(checkpoint['optimizer'])  
        start_epoch = checkpoint['epoch'] + 1
    result = OrderedDict()


    # train loop
    best_top1 = 0.0
    for epoch in range(start_epoch, max_epoch + 1):
        model.train()
        rs = dict()
        if EB:
            rs['train'] = run_epoch_with_EB(model, trainloader, criterion, optimizer,max_epoch, desc_default='train', epoch=epoch)            
        else:
            rs['train'] = run_epoch(model, trainloader, criterion, optimizer,max_epoch, desc_default='train', epoch=epoch)
        model.eval()
        scheduler.step()
        if math.isnan(rs['train']['loss']):
            raise Exception('train loss is NaN.')

        rs['test'] = run_epoch(model, testloader_, criterion, None,max_epoch, desc_default='*test', epoch=epoch)
        if metric != 'last':
            best_top1 = rs[metric]['top1']
        for key, setname in itertools.product(['loss', 'top1', 'top5'], ['train', 'test']):
            result['%s_%s' % (key, setname)] = rs[setname][key]
        result['epoch'] = epoch
        
        if args.num_gpu > 1:
            model_state_dict = model.module.state_dict()
        else:
            model_state_dict =model.state_dict() 
            
        if result['top1_test'] > best_top1:
            best_top1 = result['top1_test']
            log_fp = open(args.log_path, 'a')             
            logger.info('Epoch {0} new best top1: {1:03f}'.format(epoch+1,best_top1))
            print('Epoch {0} new best top1: {1:03f}'.format(epoch+1,best_top1), file=log_fp)       

            if os.path.isfile(args.path+'best_weight.pth'):
                os.remove(args.path+'best_weight.pth')
            #save current best
            if args.num_gpu > 1:
                model_state_dict = model.module.state_dict()
            else:
                model_state_dict =model.state_dict()
            torch.save({                        
                        'epoch': epoch,
                        'log': {
                            'train': rs['train'].get_dict(),
                            'test': rs['test'].get_dict(),
                            },
                        'optimizer': optimizer.state_dict(),
                        'model_state_dict':model_state_dict
                        }, args.path+'/best_weight.pth') 
            
            logger.info('Saving models to {}'.format(args.path+'best_weight.pth'))
            print('Saving models to {}'.format(args.path+'best_weight.pth'), file=log_fp)     
            log_fp.close()
        #save checkpoint    
        torch.save({                        
                    'epoch': epoch,
                    'log': {
                    'train': rs['train'].get_dict(),
                    'test': rs['test'].get_dict(),
                    },
                   'optimizer': optimizer.state_dict(),
                   'model_state_dict':model_state_dict
                    }, args.path+'/checkpoint.pth')              
    return best_top1

if __name__ == '__main__':
    args = parser.parse_args()
    args.path = args.path+'/'+args.dataset+'/'+args.model+'/'
    args.action_path = None
    args.log_path = args.path+'/logs.txt'    
    
    if not os.path.isdir(args.path):
        os.makedirs(args.path)
    if args.model == 'pyramid':
        args.conf = {
            'type': args.model,
            'dataset': args.dataset,
            #for pyramid+shakedrop
            'depth': 272,
            'alpha': 200,
            'bottleneck': True       
        } 
    else:
        args.conf = {
            'type': args.model,
            'dataset': args.dataset  
        }         
        
    torch.manual_seed(args.manual_seed)
    np.random.seed(args.manual_seed)   
    if torch.cuda.is_available():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
        cudnn.benchmark = True
        cudnn.enable = True
        torch.cuda.manual_seed(args.manual_seed)
    else:
        device = torch.device('cpu')
        
    args.num_gpu = torch.cuda.device_count()
    model = get_model(args.conf, num_class(args.dataset))
    if args.num_gpu > 1:
         model = nn.DataParallel(model)  
            
    memory = Memory(args.action_path)
    
    #load learned policy    
    controller = Controller(args.policy_embedding_size, args.policy_hidden_size, device).to(device)
    controller_checkpoint = torch.load(args.policy_checkpoint)
    controller.load_state_dict(controller_checkpoint['model_state_dict'])    
    actions_p, actions_log_p = controller.distribution()
    subpolicies, subpolicies_str, subpolicies_probs  = controller.convert(actions_p)
    
    subpolicies_str.sort(key = lambda subpolices_str: subpolices_str[2], reverse=True)
    del controller
    log_fp = open(args.log_path, 'a')    
    logger.info('------ Learned Policy with AWS Augment ------')
    print('------ Learned Policy with AWS Augment ------', file=log_fp)    
    for i, subpolicy in enumerate(subpolicies_str):
        if subpolicy[2] > 1e-6:
            logger.info('# Sub-policy {0}: {1}, {2} {3:06f}'.format(i+1, subpolicy[0],subpolicy[1],subpolicy[2]))
            print('# Sub-policy {0}: {1}, {2} {3:06f}'.format(i+1, subpolicy[0],subpolicy[1],subpolicy[2]), file=log_fp)
    log_fp.close()   
    

    best_top1 = train_and_eval(args, model, args.n_epochs, (subpolicies_probs,subpolicies,memory), EB = args.enlarge_batch)
    logger.info('[Best top1-test: {0:06f}'.format(best_top1))

        
