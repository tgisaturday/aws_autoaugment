#imported from https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO.py

import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.autograd import Variable

from AWSAutoAugment.augmentations import augment_list as transformations

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Operation:
    def __init__(self, types_softmax, probs_softmax, argmax=False):
        # Ekin Dogus says he sampled the softmaxes, and has not used argmax
        # We might still want to use argmax=True for the last predictions, to ensure
        # the best solutions are chosen and make it deterministic.

        self.type = types_softmax.argmax()
        self.transformation = transformations[self.type]
        self.prob = probs_softmax.argmax() / (OP_PROBS-1)
        self.magnitude = m*(t[2]-t[1]) + t[1]

    def __str__(self):
        return 'Operation %2d (P=%.3f)' % (self.type, self.prob)

class Subpolicy:
    def __init__(self, *operations):
        self.operations = operations

    def __call__(self, X):
        for op in self.operations:
            X = op(X)
        return X

    def __str__(self):
        ret = ''
        for i, op in enumerate(self.operations):
            ret += str(op)
            if i < len(self.operations)-1:
                ret += '\n'
        return ret    
    
# Experiment parameters

LSTM_UNITS = 100

SUBPOLICIES = 5
SUBPOLICY_OPS = 2

OP_TYPES = 36
OP_PROBS = 11
      
class Controller:
    def __init__(self, lr, betas, clip_epsilon, entropy_weight):
        self.lr = lr
        self.betas = betas
        self.clip_epsilon = clip_epsilon
        self.entropy_weight = entropy_weight     
        self.lstm = nn.LSTM(SUBPOLICIES, LSTM_UNITS)
        self.opt1_type = nn.Linear(LSTM_UNITS, OP_TYPES),
        self.opt1_prob = nn.Linear(LSTM_UNITS, OP_PROBS),    
        self.opt2_type = nn.Linear(LSTM_UNITS, OP_TYPES),         
        self.opt2_prob = nn.Linear(LSTM_UNITS, OP_PROBS), 
           
        self.init_parameters()
        
    def forward(self):
        x = np.zeros((1, SUBPOLICIES, 1))        
        x = self.lstm(x)
        opt1_type = self.opt1_type(x)
        opt1_prob = self.opt1_prob(x)      
        opt2_type = self.opt2_type(x)
        opt2_prob = self.opt2_prob(x)
        opt1_type_softmax = F.softmax(opt1_type, dim=-1)
        opt1_type_log = F.log_softmax(opt1_type, dim=-1)  
        opt2_type_softmax = F.softmax(opt2_type, dim=-1)
        opt2_type_log = F.log_softmax(opt2_type, dim=-1)
        
        opt1_prob_softmax = F.softmax(opt1_prob, dim=-1)
        opt1_prob_log = F.log_softmax(opt1_prob, dim=-1)  
        opt2_prob_softmax = F.softmax(opt2_prob, dim=-1)
        opt2_prob_log = F.log_softmax(opt2_prob, dim=-1)  
        
        return ((opt1_type_softmax,opt1_prob_softmax) , (opt2_type_softmax, opt2_prob_softmax)), ((opt1_type_log ,opt1_prob_log ) , (opt2_type_log , opt2_prob_log))
            
    def predict(self):
        softmaxes,log_softmaxes = self.forward()
        # convert softmaxes into subpolicies
        subpolicies = []
        for i in range(SUBPOLICIES):
            operations = []
            for j in range(SUBPOLICY_OPS):
                op = softmaxes[j]
                operations.append(Operation(op[0],op[1]))
            subpolicies.append(Subpolicy(*operations))
        return softmaxes, log_softmaxes, subpolicies
    
    def init_parameters(self):
        init_range = 0.1
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)
        self.opt1_type.bias.data.fill_(0)  
        self.opt2_type.bias.data.fill_(0)          
        self.opt1_prob.bias.data.fill_(0)          
        self.opt2_prob.bias.data.fill_(0)  


        
    def update(self, reward):
        actions_p, actions_log_p = self.forward()   
        loss = self.cal_loss(actions_p, actions_log_p, reward)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return self
    
    def clip(self, actions_importance):
        lower = torch.ones_like(actions_importance).to(self.device) * (1 - self.clip_epsilon)
        upper = torch.ones_like(actions_importance).to(self.device) * (1 + self.clip_epsilon)

        actions_importance, _ = torch.min(torch.cat([actions_importance.unsqueeze(0), upper.unsqueeze(0)], dim=0), dim=0)
        actions_importance, _ = torch.max(torch.cat([actions_importance.unsqueeze(0), lower.unsqueeze(0)], dim=0), dim=0)

        return actions_importance   
    
    def cal_loss(self, actions_p, actions_log_p, reward):
        actions_importance = actions_p
        clipped_actions_importance = self.clip(actions_importance)
        actions_reward = actions_importance * reward
        clipped_actions_reward = clipped_actions_importance * reward

        actions_reward, _ = torch.min(torch.cat([actions_reward.unsqueeze(0), clipped_actions_reward.unsqueeze(0)], dim=0), dim=0)
        policy_loss = -1 * torch.sum(actions_reward)
        entropy = -1 * torch.sum(actions_p * actions_log_p)
        entropy_bonus = -1 * entropy * self.entropy_weight

        return policy_loss + entropy_bonus