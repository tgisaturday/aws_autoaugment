import copy
import os
import sys
import time
from collections import OrderedDict, defaultdict

import torch

import numpy as np

from tqdm import tqdm

#from FastAutoAugment.archive import remove_deplicates, policy_decoder
from AWSAutoAugment.augmentations import augment_list
from AWSAutoAugment.common import get_logger, add_filehandler
from AWSAutoAugment.data import get_dataloaders
from AWSAutoAugment.metrics import Accumulator
from AWSAutoAugment.networks import get_model, num_class
from AWSAutoAugment.train import train_and_eval
from AWSAutoAugment.PPO import Controller
from theconf import Config as C, ConfigArgumentParser


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
parser.add_argument('--manual_seed', default=0, type=int)

""" run config """
parser.add_argument('--n_epochs', type=int, default=10)
parser.add_argument('--init_lr', type=float, default=0.025)
parser.add_argument('--lr_schedule', type=str, default='cosine')
parser.add_argument('--cutout', type=int, default=0)
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
parser.add_argument('--policy_lr', type=float, default=0.1)
parser.add_argument('--policy_adam_beta1', type=float, default=0.5)  # arch_opt_param
parser.add_argument('--policy_adam_beta2', type=float, default=0.999)  # arch_opt_param
parser.add_argument('--policy_adam_eps', type=float, default=1e-8)  # arch_opt_param
parser.add_argument('--policy_weight_decay', type=float, default=0)
parser.add_argument('--policy_clip_epsilon', type=float, default=0.2)
parser.add_argument('--policy_entropy_weight', type=float, default=1e-5)


logger = get_logger('AWSAugment')
logger.setLevel(logging.INFO)

def run_epoch(model, loader, loss_fn, optimizer, desc_default='', epoch=0, writer=None, verbose=1, scheduler=None):
    tqdm_disable = bool(os.environ.get('TASK_NAME', ''))    # KakaoBrain Environment
    if verbose:
        loader = tqdm(loader, disable=tqdm_disable)
        loader.set_description('[%s %04d/%04d]' % (desc_default, epoch, C.get()['epoch']))

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
            if C.get()['optimizer'].get('clip', 5) > 0:
                nn.utils.clip_grad_norm_(model.parameters(), C.get()['optimizer'].get('clip', 5))
            optimizer.step()

        top1, top5 = accuracy(preds, label, (1, 5))
        metrics.add_dict({
            'loss': loss.item() * len(data),
            'top1': top1.item() * len(data),
            'top5': top5.item() * len(data),
        })
        cnt += len(data)
        if verbose:
            postfix = metrics / cnt
            if optimizer:
                postfix['lr'] = optimizer.param_groups[0]['lr']
            loader.set_postfix(postfix)

        if scheduler is not None:
            scheduler.step(epoch - 1 + float(steps) / total_steps)

        del preds, loss, top1, top5, data, label

    if tqdm_disable:
        if optimizer:
            logger.info('[%s %03d/%03d] %s lr=%.6f', desc_default, epoch, C.get()['epoch'], metrics / cnt, optimizer.param_groups[0]['lr'])
        else:
            logger.info('[%s %03d/%03d] %s', desc_default, epoch, C.get()['epoch'], metrics / cnt)

    metrics /= cnt
    if optimizer:
        metrics.metrics['lr'] = optimizer.param_groups[0]['lr']
    if verbose:
        for key, value in metrics.items():
            writer.add_scalar(key, value, epoch)
    return metrics

def train_and_eval(args, model, epoch, policy, test_ratio=0.0, cv_fold=0, shared=False):

    max_epoch = epoch
    trainsampler, trainloader, validloader, testloader_ = get_dataloaders(args, policy, test_ratio, split_idx=cv_fold)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
            model.parameters(),
            lr=arg.init_lr,
            momentum=args.momentum
            weight_decay=arg.weight_decay
            nesterov=not args.no_nesterov,
        )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=0.)
    
    result = OrderedDict()
    epoch_start = 1
    if save_path and os.path.exists(save_path):
        logger.info('%s file found. loading...' % save_path)
        data = torch.load(save_path)
        if 'model' in data or 'state_dict' in data:
            key = 'model' if 'model' in data else 'state_dict'
            logger.info('checkpoint epoch@%d' % data['epoch'])
            if not isinstance(model, DataParallel):
                model.load_state_dict({k.replace('module.', ''): v for k, v in data[key].items()})
            else:
                model.load_state_dict({k if 'module.' in k else 'module.'+k: v for k, v in data[key].items()})
            optimizer.load_state_dict(data['optimizer'])
            if data['epoch'] < epoch:
                epoch_start = data['epoch']
            else:
                only_eval = True
        else:
            model.load_state_dict({k: v for k, v in data.items()})
        del data
    else:
        logger.info('"%s" file not found. skip to pretrain weights...' % save_path)
        if only_eval:
            logger.warning('model checkpoint not found. only-evaluation mode is off.')
        only_eval = False

    # train loop
    best_top1 = 0
    for epoch in range(epoch_start, max_epoch + 1):
        model.train()
        rs = dict()
        rs['train'] = run_epoch(model, trainloader, criterion, optimizer, desc_default='train', epoch=epoch, writer=writers[0], verbose=True, scheduler=scheduler)
        model.eval()

        if math.isnan(rs['train']['loss']):
            raise Exception('train loss is NaN.')

        if epoch % 5 == 0 or epoch == max_epoch:
            rs['valid'] = run_epoch(model, validloader, criterion, None, desc_default='valid', epoch=epoch, writer=writers[1], verbose=True)
            rs['test'] = run_epoch(model, testloader_, criterion, None, desc_default='*test', epoch=epoch, writer=writers[2], verbose=True)

            if metric == 'last' or rs[metric]['top1'] > best_top1:
                if metric != 'last':
                    best_top1 = rs[metric]['top1']
                for key, setname in itertools.product(['loss', 'top1', 'top5'], ['train', 'valid', 'test']):
                    result['%s_%s' % (key, setname)] = rs[setname][key]
                result['epoch'] = epoch

                writers[1].add_scalar('valid_top1/best', rs['valid']['top1'], epoch)
                writers[2].add_scalar('test_top1/best', rs['test']['top1'], epoch)

                reporter(
                    loss_valid=rs['valid']['loss'], top1_valid=rs['valid']['top1'],
                    loss_test=rs['test']['loss'], top1_test=rs['test']['top1']
                )

                # save checkpoint
                if save_path:
                    logger.info('save model@%d to %s' % (epoch, save_path))
                    torch.save({
                        'epoch': epoch,
                        'log': {
                            'train': rs['train'].get_dict(),
                            'valid': rs['valid'].get_dict(),
                            'test': rs['test'].get_dict(),
                        },
                        'optimizer': optimizer.state_dict(),
                        'model': model.state_dict()
                    }, save_path)
                    torch.save({
                        'epoch': epoch,
                        'log': {
                            'train': rs['train'].get_dict(),
                            'valid': rs['valid'].get_dict(),
                            'test': rs['test'].get_dict(),
                        },
                        'optimizer': optimizer.state_dict(),
                        'model': model.state_dict()
                    }, save_path.replace('.pth', '_e%d_top1_%.3f_%.3f' % (epoch, rs['train']['top1'], rs['test']['top1']) + '.pth'))

    result['top1_test'] = best_top1
    if shared:
        return model, result
    else:
        del model
        return result

if __name__ == '__main__':
    args = parser.parse_args()
    args.path = args.path+'/'+args.dataset+'/'+args.model
    
    args.conf = {
        'type': args.model,
        'dataset': args.dataset,
    }    
         
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    np.random.seed(args.manual_seed)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.makedirs(args.path, exist_ok=True)
    curr_weights = get_model(conf, num_class(args.dataset))   
    
    current_policy = policy_decode(policy)
    # stage 1 training to get aws
    shared_weights, reward = train_and_eval(args, curr_weights, args.warmup_epoch, current_policy, test_ratio=0.0, cv_fold=0, shared=True)
    logger.info('[Stage 1 top1-test: %3d', reward)

    #stage 2 policy update with PPO
    betas = (args.policy_adam_beta1, args.policy_adam_beta2)
    controller = Controller(args, args.policy_lr, betas, args.policy_clip_epsilon, args.policy_entropy_weight)
    
    for t in range(args.policy_steps):
        curr_weights = shared_weights
        logger.info('Controller: Epoch %d / %d' % (t+1, args.policy_steps))
        softmaxes, log_softmaxes, subpolicies = controller.predict(SUBPOLICIES)
        for i, subpolicy in enumerate(subpolicies):
            logger.info('# Sub-policy %d' % (i+1))
            logger.info(subpolicy)

        new_reward = train_and_eval(args, curr_weights, args.finetune_epoch, subpolicies , test_ratio=0.0, cv_fold=0)
        if t == 0:
            reward = new_reward
        else:
            reward = args.reward_ema_decay * reward + (1- args.reward_ema_decay) * new_reward           
        controller.update(reward)
        
    logger.info('Best policies found.')         
    for i, subpolicy in enumerate(subpolicies):
        logger.info('# Sub-policy %d' % (i+1))
        logger.info(subpolicy)