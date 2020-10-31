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
import torch.backends.cudnn as cudnn
from tqdm import tqdm

#from FastAutoAugment.archive import remove_deplicates, policy_decoder
from augmentations import augment_list
from common import get_logger, add_filehandler
from data import get_dataloaders
from metrics import *
from networks import get_model, num_class
from PPO import PPO



#from hyperopt import hp
#import ray
#import gorilla
#from ray.tune.trial import Trial
#from ray.tune.trial_runner import TrialRunner
#from ray.tune.suggest import HyperOptSearch
#from ray.tune import register_trainable, run_experiments

logger = get_logger('AWS AutoAugment')

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='./results')
parser.add_argument('--dataroot', type=str, default='./data')
parser.add_argument('--gpu', help='gpu available', default='0,1,2,3')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--checkpoint',type=str)
parser.add_argument('--manual_seed', default=0, type=int)

""" run config """
parser.add_argument('--n_epochs', type=int, default=10)
parser.add_argument('--init_lr', type=float, default=0.025)
parser.add_argument('--lr_schedule', type=str, default='cosine')
parser.add_argument('--cutout', type=int, default=16)
parser.add_argument('--label_smoothing', type=float, default=0.0)
# lr_schedule_param

parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar10', 'cifar100','imagenet'])
parser.add_argument('--model', type=str, default='wresnet28_10', choices=[ 'wresnet28_10'])
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--valid_size', type=int, default=10000)

parser.add_argument('--opt_type', type=str, default='sgd', choices=['sgd'])
parser.add_argument('--momentum', type=float, default=0.9)  # opt_param
parser.add_argument('--no_nesterov', action='store_true')  # opt_param
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--n_worker', type=int, default=16)

# policy search config
""" policy  search algo and warmup """
parser.add_argument('--policy_algo', type=str, default='rl', choices=['rl'])
parser.add_argument('--warmup_epochs', type=int, default=200)
parser.add_argument('--finetune_epochs', type=int, default=10)
parser.add_argument('--policy_steps', type=int, default=500)
parser.add_argument('--reward_ema_decay', type=float, default=0.9)

""" policy hyper-parameters """
parser.add_argument('--policy_init_type', type=str, default='uniform', choices=['normal', 'uniform'])
parser.add_argument('--policy_opt_type', type=str, default='adam', choices=['adam'])
parser.add_argument('--policy_batch_size', type=int, default=5)
parser.add_argument('--policy_lr', type=float, default=0.1)
parser.add_argument('--policy_adam_beta1', type=float, default=0.5)  # arch_opt_param
parser.add_argument('--policy_adam_beta2', type=float, default=0.999)  # arch_opt_param
parser.add_argument('--policy_adam_eps', type=float, default=1e-8)  # arch_opt_param
parser.add_argument('--policy_weight_decay', type=float, default=0)
parser.add_argument('--policy_clip_epsilon', type=float, default=0.2)
parser.add_argument('--policy_entropy_weight', type=float, default=1e-5)

""" controller hyper-parameters """
parser.add_argument('--policy_embedding_size', type=int, default=32)
parser.add_argument('--policy_hidden_size', type=int, default=100)
parser.add_argument('--policy_subpolices', type=int, default=25) # number of subpolicies
logger = get_logger('AWSAugment')
logger.setLevel(logging.INFO)

def run_epoch( model, loader, loss_fn, optimizer, max_epoch, desc_default='', epoch=0,  scheduler=None):

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

        if scheduler is not None:
            scheduler.step(epoch - 1 + float(steps) / total_steps)

        del preds, loss, top1, top5, data, label


    if optimizer:
        logger.info('[%s %03d/%03d] %s lr=%.6f', desc_default, epoch, max_epoch, metrics / cnt, optimizer.param_groups[0]['lr'])
    else:
        logger.info('[%s %03d/%03d] %s', desc_default, epoch, max_epoch, metrics / cnt)

    metrics /= cnt
    if optimizer:
        metrics.metrics['lr'] = optimizer.param_groups[0]['lr']
    return metrics

def train_and_eval(args, model, epoch, policy, test_ratio=0.0, cv_fold=0, shared=False, metric='last'):

    max_epoch = epoch
    trainsampler, trainloader, validloader, testloader_ = get_dataloaders(args, policy, test_ratio, split_idx=cv_fold)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.init_lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=not args.no_nesterov,
        )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch, eta_min=0.)
    
    result = OrderedDict()
    epoch_start = 1

    # train loop
    best_top1 = 0
    for epoch in range(epoch_start, max_epoch + 1):
        model.train()
        rs = dict()
        rs['train'] = run_epoch(model, trainloader, criterion, optimizer,max_epoch, desc_default='train', epoch=epoch, scheduler=scheduler)
        model.eval()

        if math.isnan(rs['train']['loss']):
            raise Exception('train loss is NaN.')

        if epoch % 5 == 0 or epoch == max_epoch:
            rs['valid'] = run_epoch(model, validloader, criterion, None,max_epoch, desc_default='valid', epoch=epoch)
            rs['test'] = run_epoch(model, testloader_, criterion, None,max_epoch, desc_default='*test', epoch=epoch)

            if metric == 'last' or rs[metric]['top1'] > best_top1:
                if metric != 'last':
                    best_top1 = rs[metric]['top1']
                for key, setname in itertools.product(['loss', 'top1', 'top5'], ['train', 'valid', 'test']):
                    result['%s_%s' % (key, setname)] = rs[setname][key]
                result['epoch'] = epoch
       
    result['top1_test'] = best_top1
    if shared:
        return model, result
    else:
        del model
        return result

if __name__ == '__main__':
    args = parser.parse_args()
    args.path = args.path+'/'+args.dataset+'/'+args.model    
    args.policy_path = args.path+'/policy.txt'
    args.conf = {
        'type': args.model,
        'dataset': args.dataset,
    }    
         
    torch.manual_seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
        cudnn.benchmark = True
        cudnn.enable = True
        logging.info('using gpu : {}'.format(args.gpu))
        torch.cuda.manual_seed(args.manual_seed)
    else:
        device = torch.device('cpu')
        logging.info('using cpu')    
        
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.makedirs(args.path, exist_ok=True)
    shared_weights = get_model(args.conf, num_class(args.dataset))   
    
    # stage 1 training to get aws
    shared_weights, result = train_and_eval(args, shared_weights, args.warmup_epochs, policy='uniform', test_ratio=0.2, cv_fold=0, shared=True)
    logger.info('[Stage 1 top1-valid: %3d', result['top1_valid'])
    logger.info('[Stage 1 top1-test: %3d', result['top1_test'])

    #stage 2 policy update with PPO
    betas = (args.policy_adam_beta1, args.policy_adam_beta2)
    
    policy = PPO(args.policy_lr, betas, args.policy_clip_epsilon, 
                            args.policy_entropy_weight,args.policy_embedding_size, 
                            args.policy_hidden_size, args.policy_subpolices,device)
    controller = policy.controller

    for t in range(args.policy_steps):
        curr_weights = copy.deepcopy(shared_weights)
        policy_fp = open(args.policy_path, 'a')
        logger.info('Controller: Epoch %d / %d' % (t+1, args.policy_steps))
        print('-----Controller: Epoch %d / %d-----' % (t+1, args.policy_steps), file=policy_fp)
        actions_p, actions_log_p, actions_index = controller.sample()
        subpolicies, subpolicies_str  = controller.convert(actions_index)
        for i, subpolicy in enumerate(subpolicies_str):
            logger.info('# Sub-policy {}: {}'.format(i+1, subpolicy))
            print('# Sub-policy {}: {}'.format(i+1, subpolicy), file=policy_fp)            
        policy_fp.close()
        result = train_and_eval(args, curr_weights, args.finetune_epochs, subpolicies , test_ratio=0.2, cv_fold=0)
        new_reward = result['top1_valid']
        logger.info('[Stage 1 top1-valid: %3d', result['top1_valid'])
        logger.info('[Stage 1 top1-test: %3d', result['top1_test'])            
        if t == 0:
            reward = new_reward
        else:
            reward = args.reward_ema_decay * reward + (1- args.reward_ema_decay) * new_reward           
        policy.update(actions_index, reward)
        
    logger.info('Best policies found.')         
    for i, subpolicy in enumerate(subpolicies):
        logger.info('# Sub-policy %d' % (i+1))
        logger.info(subpolicy)