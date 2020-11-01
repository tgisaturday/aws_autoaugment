#imported from https://github.com/MarSaKi/nasnet/blob/master/PPO.py

import torch
import torch.nn as nn
from torch.distributions import Categorical

import torch.nn.functional as F

from torch.autograd import Variable

from augmentations import augment_list, augment_list_by_name

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
    

class PPO(object):
    def __init__(self, lr, betas, clip_epsilon, entropy_weight,embedding_size, hidden_size, subpolicies ,device):
        self.lr = lr
        self.betas = betas
        self.clip_epsilon = clip_epsilon
        self.entropy_weight = entropy_weight    
        self.controller = Controller(embedding_size, hidden_size, subpolicies ,device).to(device)
        self.optimizer = torch.optim.Adam(params=self.controller.parameters(), lr=self.lr, betas = self.betas)
        self.device = device
        
    def update(self, actions_index, reward): 
        actions_p, actions_log_p = self.controller.get_p(actions_index)        
        loss = self.cal_loss(actions_p, actions_log_p, reward)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
       
    
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
    
        
class Controller(nn.Module):
    def __init__(self, embedding_size, hidden_size, subpolicies ,device):
        super(Controller, self).__init__()        
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.subpolicies = subpolicies
        self.len_OPS = 36 * 36 #number of operation candidates
        self.device = device       
        
        self.embedding = nn.Embedding(self.len_OPS+1, self.embedding_size)    
        
        #operation 
        self.op_decoder = nn.Linear(hidden_size, self.len_OPS)
        
        self.rnn = nn.LSTMCell(self.embedding_size, hidden_size)

        self.init_parameters()
        
    def forward(self, input, h_t, c_t):
        input = self.embedding(input)
        h_t, c_t = self.rnn(input, (h_t, c_t))
        logits = self.op_decoder(h_t)
        return h_t, c_t, logits
    
    def sample(self):
        input = torch.LongTensor([self.len_OPS]).to(self.device)
        h_t, c_t = self.init_hidden()
        actions_p = []
        actions_log_p = []
        actions_index = []

        for subpolicy in range(self.subpolicies):
            input = torch.LongTensor([self.len_OPS]).to(self.device)
            h_t, c_t = self.init_hidden()            
            h_t, c_t, logits = self.forward(input, h_t, c_t)
            action_index = Categorical(logits=logits).sample()
            
            p = F.softmax(logits, dim=-1)[0,action_index]
            log_p = F.log_softmax(logits, dim=-1)[0,action_index]
            actions_p.append(p.detach())
            actions_log_p.append(log_p.detach())
            actions_index.append(action_index)

        actions_p = torch.cat(actions_p)
        actions_log_p = torch.cat(actions_log_p)
        actions_index = torch.cat(actions_index)

        return actions_p, actions_log_p, actions_index


    def get_p(self, actions_index):
        input = torch.LongTensor([self.len_OPS]).to(self.device)
        h_t, c_t = self.init_hidden()
        t = 0
        actions_p = []
        actions_log_p = []

        for subpolicy in range(self.subpolicies):
            h_t, c_t, logits = self.forward(input, h_t, c_t)
            action_index = actions_index[t].unsqueeze(0)
            t += 1
            p = F.softmax(logits, dim=-1)[0, action_index]
            log_p = F.log_softmax(logits, dim=-1)[0, action_index]
            actions_p.append(p)
            actions_log_p.append(log_p)
            
            input = action_index + self.len_OPS

        actions_p = torch.cat(actions_p)
        actions_log_p = torch.cat(actions_log_p)

        return actions_p, actions_log_p
            
    def convert(self, actions_index):
        operations = []
        operations_str = []
        for actions in actions_index:
            op1_idx = actions // 36
            op2_idx = actions % 36 -1
            transformations = augment_list()
            transformations_str = augment_list_by_name()
            operations.append([transformations[op1_idx],transformations[op2_idx]])
            operations_str.append([transformations_str[op1_idx],transformations_str[op2_idx]])            
        return operations, operations_str
    
    def init_hidden(self):
        h_t = torch.zeros(1, self.hidden_size, dtype=torch.float, device=self.device)
        c_t = torch.zeros(1, self.hidden_size, dtype=torch.float, device=self.device)

        return (h_t, c_t)    
    
    def init_parameters(self):
        init_range = 0.1
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)         
        self.op_decoder.bias.data.fill_(0)


        
