#imported from https://github.com/MarSaKi/nasnet/blob/master/PPO.py

import torch
import torch.nn as nn
from torch.distributions import Categorical

import torch.nn.functional as F

from torch.autograd import Variable

from augmentations import augment_list, augment_list_by_name

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    

class PPO(object):
    def __init__(self, lr, betas, clip_epsilon, entropy_weight,embedding_size, hidden_size, baseline_weight, device):
        self.lr = lr
        self.betas = betas
        self.clip_epsilon = clip_epsilon
        self.entropy_weight = entropy_weight    
        self.controller = Controller(embedding_size, hidden_size, device).to(device)
        self.optimizer = torch.optim.Adam(params=self.controller.parameters(), lr=self.lr, betas = self.betas)
        self.device = device
        self.baseline = None
        self.baseline_weight = baseline_weight
        
    def update(self, acc): 
        actions_p, actions_log_p = self.controller.get_p()          
        if self.baseline == None:
            self.baseline = acc            
        
        else:
            loss = self.cal_loss(actions_p, actions_log_p, acc)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()            
            #update baseline for next step
            self.baseline = self.baseline * self.baseline_weight + acc* (1 - self.baseline_weight)          
       
    
    def clip(self, actions_importance):
        lower = torch.ones_like(actions_importance).to(self.device) * (1 - self.clip_epsilon)
        upper = torch.ones_like(actions_importance).to(self.device) * (1 + self.clip_epsilon)

        actions_importance, _ = torch.min(torch.cat([actions_importance.unsqueeze(0), upper.unsqueeze(0)], dim=0), dim=0)
        actions_importance, _ = torch.max(torch.cat([actions_importance.unsqueeze(0), lower.unsqueeze(0)], dim=0), dim=0)

        return actions_importance   
    
    def cal_loss(self, actions_p, actions_log_p, acc):
        actions_importance = actions_p
        clipped_actions_importance = self.clip(actions_importance)
        reward = acc - self.baseline
        actions_reward = actions_importance * reward
        clipped_actions_reward = clipped_actions_importance * reward

        actions_reward, _ = torch.min(torch.cat([actions_reward.unsqueeze(0), clipped_actions_reward.unsqueeze(0)], dim=0), dim=0)
        policy_loss = -1 * torch.sum(actions_reward)
        entropy = -1 * torch.sum(actions_p * actions_log_p)
        entropy_bonus = -1 * entropy * self.entropy_weight

        return policy_loss + entropy_bonus
    
        
class Controller(nn.Module):
    def __init__(self, embedding_size, hidden_size, device):
        super(Controller, self).__init__()        
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

    def forward(self, input):
        input = self.embedding(input)
        logits = self.policy(input)
        return logits
    
    def get_p(self):
        actions_p = []
        actions_log_p = []
        
        for i in range(self.len_OPS):
            input = torch.LongTensor([i]).to(self.device)
            logits = self.forward(input)
            p = torch.sigmoid(logits)
            actions_p.append(p)
        actions_p = torch.cat(actions_p)
        actions_p = torch.div(actions_p, torch.sum(actions_p))
        actions_log_p = torch.log(actions_p)
        
        return actions_p, actions_log_p
    
    
    def convert(self, actions_p):
        operations = []
        probs = []
        operations_str = []
        for actions in range(self.len_OPS):
            op1_idx = actions // 36
            op2_idx = actions % 36 
            transformations = augment_list()
            transformations_str = augment_list_by_name()
            prob = actions_p[actions].item()
            probs.append(prob)
            operations.append([transformations[op1_idx],transformations[op2_idx], prob])
            operations_str.append([transformations_str[op1_idx],transformations_str[op2_idx],prob])            
        return operations, operations_str, probs
       

        
