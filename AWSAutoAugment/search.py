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


from augmentations import augment_list
from common import get_logger, add_filehandler
from data import get_dataloaders
from metrics import *
from networks import get_model, num_class
from PPO import PPO

logger = get_logger('AWS AutoAugment')

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='./results')
parser.add_argument('--dataroot', type=str, default='./data')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--checkpoint',type=str)
parser.add_argument('--manual_seed', default=0, type=int)

""" run config """
parser.add_argument('--init_lr', type=float, default=0.4)
#parser.add_argument('--finetune_lr', type=float, default=0.025)
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
parser.add_argument('--policy_steps', type=int, default=800)
parser.add_argument('--policy_update_epochs', type=int, default=10)
""" policy hyper-parameters """
parser.add_argument('--policy_init_type', type=str, default='uniform', choices=['normal', 'uniform'])
parser.add_argument('--policy_controller_type', type=str, default='lstm', choices=['lstm','fcn'])
parser.add_argument('--policy_opt_type', type=str, default='adam', choices=['adam'])
parser.add_argument('--policy_lr', type=float, default=0.1)

parser.add_argument('--policy_adam_beta1', type=float, default=0.5)  # arch_opt_param
parser.add_argument('--policy_adam_beta2', type=float, default=0.999)  # arch_opt_param
parser.add_argument('--policy_adam_eps', type=float, default=1e-8)  # arch_opt_param
parser.add_argument('--policy_adam_weight_decay', type=float, default=0)

parser.add_argument('--policy_clip_epsilon', type=float, default=0.2)
parser.add_argument('--policy_entropy_weight', type=float, default=1e-5)
parser.add_argument('--baseline_ema_weight', type=float, default=0.9)

""" controller hyper-parameters """
parser.add_argument('--policy_embedding_size', type=int, default=32)
parser.add_argument('--policy_hidden_size', type=int, default=100)
parser.add_argument('--policy_subpolices', type=int, default=20) # number of subpolicies to print


        
def run_epoch( model, loader, loss_fn, optimizer, max_epoch, desc_default='', epoch=0):

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
        logger.info('[%s %03d/%03d] %s lr=%.6f', desc_default, epoch, max_epoch, metrics / cnt, optimizer.param_groups[0]['lr'])
    else:
        logger.info('[%s %03d/%03d] %s', desc_default, epoch, max_epoch, metrics / cnt)

    metrics /= cnt
    if optimizer:
        metrics.metrics['lr'] = optimizer.param_groups[0]['lr']
    return metrics

def train_and_eval(args, model, optimizer,scheduler, epoch_start, max_epoch, policy, test_ratio=0.0, cv_fold=0, shared=False, metric='last'):
    
    trainsampler, trainloader, validloader, testloader_ = get_dataloaders(args, policy, test_ratio, split_idx=cv_fold)
    criterion = nn.CrossEntropyLoss()    
    result = OrderedDict()

    # train loop
    best_top1 = 0
    for epoch in range(epoch_start, max_epoch):
        epoch = epoch+1
        model.train()
        rs = dict()
        rs['train'] = run_epoch(model, trainloader, criterion, optimizer,max_epoch, desc_default='train', epoch=epoch)
        model.eval()
        scheduler.step()
        if math.isnan(rs['train']['loss']):
            raise Exception('train loss is NaN.')

        if epoch % 5 == 0 or epoch == max_epoch:
            rs['valid'] = run_epoch(model, validloader, criterion, None,max_epoch, desc_default='valid', epoch=epoch)
            rs['test'] = run_epoch(model, testloader_, criterion, None,max_epoch, desc_default='*test', epoch=epoch)
            if metric != 'last':
                best_top1 = rs[metric]['top1']
            for key, setname in itertools.product(['loss', 'top1', 'top5'], ['train', 'valid', 'test']):
                result['%s_%s' % (key, setname)] = rs[setname][key]
            result['epoch'] = epoch
     

    if shared:
        return model, result
    else:
        del model, optimizer, scheduler
        return result

if __name__ == '__main__':
    args = parser.parse_args()
    args.path = args.path+'/'+args.dataset+'/'+args.model+'/'
    args.policy_path = args.path+'/policy_logs.txt'
    if not os.path.isdir(args.path):
        os.makedirs(args.path)
        
    args.conf = {
        'type': args.model,
        'dataset': args.dataset,
    }    
    args.no_aug = False    
    add_filehandler(logger, args.policy_path)  
    
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
    
    args.epochs = args.warmup_epochs + args.finetune_epochs   
    
    shared_weights = get_model(args.conf, num_class(args.dataset))
    
    if args.num_gpu > 1:
         shared_weights = nn.DataParallel(shared_weights)  
            
    betas = (args.policy_adam_beta1, args.policy_adam_beta2)    
    policy = PPO(args.policy_lr, args.policy_update_epochs, betas, args.policy_clip_epsilon, 
                            args.policy_entropy_weight,args.policy_embedding_size, 
                            args.policy_hidden_size, args.baseline_ema_weight, 
                            args.policy_init_type, args.policy_controller_type, device)
    
    controller = policy.controller
    
    optimizer = torch.optim.SGD(
            shared_weights.parameters(),
            lr=args.init_lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=not args.no_nesterov,
        )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)    
    
    #uniform sampling for shared policy
    actions_p = torch.FloatTensor([1/(36*36) for i in range(36*36)])
    subpolicies, subpolicies_str, subpolicies_probs  = controller.convert(actions_p)
    
    args.epochs = args.warmup_epochs + args.finetune_epochs
    policy_start = 0
    if args.resume:
        logger.info('Loading pretrained shared weights.')
        checkpoint = torch.load(args.path+'shared_weights.pth')
        if args.num_gpu > 1:
            shared_weights.module.load_state_dict(checkpoint['model_state_dict'])                        
        else:
            shared_weights.load_state_dict(checkpoint['model_state_dict'])  
        #restore optimizer and scheduler
        optimizer.load_state_dict(checkpoint['optimizer'])                
        scheduler.load_state_dict(checkpoint['scheduler'])       
            
        try:
            policy_start = policy.load(args.path)
        except:
            logger.info('Policy checkpoint not found.')            
    else:
        # stage 1 training to get aws with uniform sampling
        shared_weights, result = train_and_eval(args, shared_weights, optimizer, scheduler,
                                                0, args.warmup_epochs, (subpolicies_probs,subpolicies), 
                                                test_ratio=0.2, cv_fold=0, shared=True)        
        
        logger.info('[Stage 1 top1-valid: %3f', result['top1_valid'])
        logger.info('[Stage 1 top1-test: %3f', result['top1_test'])
        if args.num_gpu > 1:
            torch.save({
                'optimizer': optimizer.state_dict(), 
                'scheduler': scheduler.state_dict(),                  
                'model_state_dict':shared_weights.module.state_dict(),
                }, args.path+'shared_weights.pth')
        else:
            torch.save({
                'optimizer': optimizer.state_dict(), 
                'scheduler': scheduler.state_dict(),                  
                'model_state_dict':shared_weights.state_dict(),
                       }, args.path+'shared_weights.pth')       

    
    #stage 2 policy update with PPO
    for t in range(policy_start, args.policy_steps):

        curr_weights = copy.deepcopy(shared_weights)
        optimizer = torch.optim.SGD(
                shared_weights.parameters(),
                lr=args.init_lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                nesterov=not args.no_nesterov,
            )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)   
        checkpoint = torch.load(args.path+'shared_weights.pth')        
        optimizer.load_state_dict(checkpoint['optimizer'])                
        scheduler.load_state_dict(checkpoint['scheduler'])     
        
        logger.info('Controller: Epoch %d / %d' % (t+1, args.policy_steps))
        
        actions_p, actions_log_p = controller.distribution()
        subpolicies, subpolicies_str, subpolicies_probs  = controller.convert(actions_p)
        
        subpolicies_str.sort(key = lambda subpolices_str: subpolices_str[2], reverse=True)
        for i, subpolicy in enumerate(subpolicies_str[:args.policy_subpolices]):
            logger.info('# Sub-policy {0}: {1}, {2} {3:06f}'.format(i+1, subpolicy[0],subpolicy[1],subpolicy[2]))

            
        logger.info('...')

                  
        for i, subpolicy in enumerate(subpolicies_str[len(subpolicies_str)-args.policy_subpolices:]):
            logger.info('# Sub-policy {0}: {1}, {2} {3:06f}'.format(i+len(subpolicies_str)-args.policy_subpolices+1,
                                                                    subpolicy[0],subpolicy[1],subpolicy[2]))

            
        result = train_and_eval(args, curr_weights,optimizer,scheduler,
                                args.warmup_epochs, args.epochs,(subpolicies_probs,subpolicies), test_ratio=0.2, cv_fold=0)

        new_acc = result['top1_valid']
        
        logger.info('---------------------------------------------------------')

        baseline = policy.baseline              
        policy_loss = policy.update(new_acc)               
        logger.info('loss: %3f, new_acc: %3f baseline: %3f' % (policy_loss, new_acc, baseline))  
        logger.info('---------------------------------------------------------')                   

        policy.save(t, args.path)
    
    logger.info('Best policies found.')    

    actions_p, actions_log_p = controller.distribution()    
    subpolicies, subpolicies_str, subpolicies_probs  = controller.convert(actions_p)    
    subpolicies_str.sort(key = lambda subpolices_str: subpolices_str[2], reverse=True)
    for i, subpolicy in enumerate(subpolicies_str):
        logger.info('# Sub-policy {0}: {1}, {2} {3:06f}'.format(i+1,  subpolicy[0],subpolicy[1],subpolicy[2]))

    torch.save({'model_state_dict':controller.state_dict()}, args.path+'policy_controller.pth')        
