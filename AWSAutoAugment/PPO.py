#based on https://github.com/MarSaKi/nasnet/blob/master/PPO.py
import os
import torch
import torch.nn as nn
import copy
from torch.distributions import Categorical

import torch.nn.functional as F
import logging
from torch.autograd import Variable

from augmentations import augment_list, augment_list_by_name
from common import get_logger

logger = get_logger('AWS AutoAugment')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        
class PPO(object):
    def __init__(self, lr, ppo_epochs, betas, clip_epsilon, entropy_weight,embedding_size, hidden_size, baseline_weight,
                 init_type, controller_type, device):
        self.lr = lr
        self.betas = betas
        self.clip_epsilon = clip_epsilon
        self.entropy_weight = entropy_weight   
        if controller_type == 'lstm':
            self.controller = LSTMController(embedding_size, hidden_size, device, init_type).to(device)
        elif controller_type == 'fcn':
            self.controller = FCNController(embedding_size, hidden_size, device, init_type).to(device) 
        else:
            raise TypeError('Unsupported controller type: ' % controller_type)
        self.optimizer = torch.optim.Adam(params=self.controller.parameters(), lr=self.lr, betas = self.betas)
        self.device = device
        self.baseline = 0.0
        self.baseline_weight = baseline_weight
        self.ppo_epochs = ppo_epochs
    def update(self, acc):
        actions_p, actions_log_p = self.controller.distribution()  
        actions_p_old = actions_p.clone().detach()
        actions_log_p_old = actions_log_p.clone().detach()        
        
        for ppo_epoch in range(self.ppo_epochs):
            loss = self.cal_loss(actions_p_old, actions_log_p_old, acc)
            logger.info('[rl(%s) %03d/%03d] loss %.4f'%('ppo',ppo_epoch+1,self.ppo_epochs, loss))            
            #update policy 
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()     
        logger.info('---------------------------------------------------------')      
        #update baseline for next time
        if self.baseline == 0.0:
            self.baseline = acc
        else:
            self.baseline = self.baseline * self.baseline_weight + acc* (1 - self.baseline_weight)        
        return loss


            
    def save(self,epoch, path):
        if os.path.isfile(path+'policy_checkpoint.pth'):
            os.remove(path+'policy_checkpoint.pth')        
        torch.save({                        
                    'epoch': epoch,
                    'baseline': self.baseline,
                    'optimizer': self.optimizer.state_dict(),
                    'model_state_dict':self.controller.state_dict(),
                        }, path+'policy_checkpoint.pth')
        
    def load(self, path):
        checkpoint = torch.load(path+'policy_checkpoint.pth')
        self.optimizer.load_state_dict(checkpoint['optimizer'])   
        self.controller.load_state_dict(checkpoint['model_state_dict']) 
        self.baseline = checkpoint['baseline']
        
        return checkpoint['epoch']+1
    
    def clip(self, actions_importance):
        lower = torch.ones_like(actions_importance).to(self.device) * (1 - self.clip_epsilon)
        upper = torch.ones_like(actions_importance).to(self.device) * (1 + self.clip_epsilon)

        actions_importance, _ = torch.min(torch.cat([actions_importance.unsqueeze(0), upper.unsqueeze(0)], dim=0), dim=0)
        actions_importance, _ = torch.max(torch.cat([actions_importance.unsqueeze(0), lower.unsqueeze(0)], dim=0), dim=0)

        return actions_importance   
    
    def cal_loss(self, actions_p_old, actions_log_p_old, acc):    
        actions_p, actions_log_p = self.controller.distribution()          
        actions_importance = actions_p / actions_p_old
        clipped_actions_importance = self.clip(actions_importance)
        reward = acc - self.baseline
        actions_reward = actions_importance * reward
        clipped_actions_reward = clipped_actions_importance * reward

        actions_reward, _ = torch.min(torch.cat([actions_reward.unsqueeze(0), clipped_actions_reward.unsqueeze(0)], dim=0), dim=0)
        policy_loss = -1 * torch.sum(actions_reward)
        entropy = -1 * torch.sum(actions_p * actions_log_p)
        entropy_bonus = -1 * entropy * self.entropy_weight

        return policy_loss + entropy_bonus
    
        
class LSTMController(nn.Module):
    def __init__(self, embedding_size, hidden_size, device, init_type = 'uniform'):
        super(LSTMController, self).__init__()        
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.len_OPS = 36 * 36 #number of operation candidates
        self.device = device       
        self.embedding = nn.Embedding(self.len_OPS, self.embedding_size)   
        self.lstm = nn.LSTMCell(self.embedding_size, self.hidden_size)
        self.decoder = nn.Linear(self.hidden_size, 1)
        self.sum_actions_p = None
        if init_type == 'uniform':
            self.init_parameters()
        
    def forward(self, input, h_t, c_t):
        input = self.embedding(input)
        h_t, c_t = self.lstm(input)
        logits = self.decoder(h_t)

        return h_t, c_t, logits
    
    def distribution(self):
        actions_p = []
        actions_log_p = []
        h_t, c_t = self.init_hidden()        
        for i in range(self.len_OPS):
            input = torch.LongTensor([i]).to(self.device)
            h_t, c_t, logits = self.forward(input,h_t,c_t)
            p = torch.sigmoid(logits)
            actions_p.append(p)
        actions_p = torch.cat(actions_p)
        self.sum_actions_p = torch.sum(actions_p)   
        actions_p = torch.div(actions_p, self.sum_actions_p)
        actions_log_p = torch.log(actions_p)
        
        return actions_p, actions_log_p

    def convert(self, actions_p):
        operations = []
        probs = []
        operations_str = []
        transformations = augment_list()
        transformations_str = augment_list_by_name()        
        for actions in range(self.len_OPS):
            op1_idx = actions // 36
            op2_idx = actions % 36 
            prob = actions_p[actions].item()
            probs.append(prob)
            operations.append([transformations[op1_idx],transformations[op2_idx], prob])
            operations_str.append([transformations_str[op1_idx],transformations_str[op2_idx],prob])            
        return operations, operations_str, probs
       

    def init_hidden(self):
        h_t = torch.zeros(1, self.hidden_size, dtype=torch.float, device=self.device)
        c_t = torch.zeros(1, self.hidden_size, dtype=torch.float, device=self.device)

        return (h_t, c_t)  
    
    def init_parameters(self):
        init_range = 0.1
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.fill_(0)
        
        
class FCNController(nn.Module):
    def __init__(self, embedding_size, hidden_size, device, init_type = 'uniform'):
        super(FCNController, self).__init__()        
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.len_OPS = 36 * 36 #number of operation candidates
        self.device = device       
        self.embedding = nn.Embedding(self.len_OPS, self.embedding_size)    
        self.policy = nn.Sequential(
                                    nn.Linear(self.embedding_size, self.hidden_size),
                                    nn.Tanh(),
                                    nn.Linear(self.hidden_size, self.hidden_size),
                                    nn.Tanh(),
                                    nn.Linear(self.hidden_size,1)) 
        self.sum_actions_p = None
        if init_type == 'uniform':
            self.init_parameters()
        
    def forward(self, input):
        input = self.embedding(input)
        logits = self.policy(input)
        return logits
    
    def distribution(self):
        actions_p = []
        actions_log_p = []
        
        for i in range(self.len_OPS):
            input = torch.LongTensor([i]).to(self.device)
            logits = self.forward(input)
            p = torch.sigmoid(logits)
            actions_p.append(p)
        actions_p = torch.cat(actions_p)
        self.sum_actions_p = torch.sum(actions_p)   
        actions_p = torch.div(actions_p, self.sum_actions_p)
        actions_log_p = torch.log(actions_p)
        
        return actions_p, actions_log_p
    
    def get_p(self, action_index):
        actions_p = []
        for i in action_index:
            input = torch.LongTensor([i]).to(self.device)
            logits = self.forward(input)
            p = torch.sigmoid(logits)
            actions_p.append(p)
        actions_p = torch.cat(actions_p)
        actions_p = torch.div(actions_p, self.sum_actions_p)
        actions_log_p = torch.log(actions_p)
        
        return actions_p, actions_log_p    
    
    def convert(self, actions_p):
        operations = []
        probs = []
        operations_str = []
        transformations = augment_list()
        transformations_str = augment_list_by_name()        
        for actions in range(self.len_OPS):
            op1_idx = actions // 36
            op2_idx = actions % 36 
            prob = actions_p[actions].item()
            probs.append(prob)
            operations.append([transformations[op1_idx],transformations[op2_idx], prob])
            operations_str.append([transformations_str[op1_idx],transformations_str[op2_idx],prob])            
        return operations, operations_str, probs
       
    def init_parameters(self):
        init_range = 0.1
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)        
        