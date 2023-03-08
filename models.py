#%% 

import torch
from torch import nn 
from torch.distributions import Normal
import torch.nn.functional as F
from torchinfo import summary as torch_summary
from blitz.modules import BayesianLinear, BayesianLSTM

from utils import default_args, init_weights, weights
from maze import obs_size, action_size



class Model(nn.Module):

    def __init__(self, args = default_args, log_std_min=-20, log_std_max=2):
        super(Model, self).__init__()

        self.args = args
        self.log_std_min = log_std_min ; self.log_std_max = log_std_max
        
        self.gru = nn.GRU(
            input_size =  args.z_size,
            hidden_size = args.h_size,
            batch_first = True)
        
        self.zp_mu    = nn.Sequential(
            nn.Linear(args.h_size, args.z_size),
            nn.Tanh())
        self.zp_std   = nn.Sequential(
            nn.Linear(args.h_size, args.z_size),
            nn.Softplus())
        self.zq_mu    = nn.Sequential(
            nn.Linear(args.h_size * 2, args.z_size),
            nn.Tanh())
        self.zq_std   = nn.Sequential(
            nn.Linear(args.h_size * 2, args.z_size),
            nn.Softplus())
        
        self.new_o    = nn.Sequential(
            nn.Linear(obs_size + action_size, args.h_size),
            nn.LeakyReLU())
        
        self.pred_o   = nn.Sequential(
            nn.Linear(args.h_size, obs_size))
            #nn.Sigmoid())
        
        self.Q_1      = nn.Sequential(
            nn.Linear(args.h_size + action_size, 1),)
        self.Q_2      = nn.Sequential(
            nn.Linear(args.h_size + action_size, 1))
        
        self.a        = nn.Sequential(
            nn.Linear(args.h_size, args.h_size))
        self.a_mu     = nn.Linear(args.h_size, action_size)
        self.a_log_std_linear = nn.Linear(args.h_size, action_size)
        
        self.zp_mu.apply(init_weights)
        self.zp_std.apply(init_weights)
        self.zq_mu.apply(init_weights)
        self.zp_std.apply(init_weights)
        self.new_o.apply(init_weights)
        self.pred_o.apply(init_weights)
        self.Q_1.apply(init_weights)
        self.Q_2.apply(init_weights)
        self.a.apply(init_weights)
        self.a_mu.apply(init_weights)
        self.a_log_std_linear.apply(init_weights)
        self.to(self.args.device)
        
    def zp_from_hq_tm1(self, hq_tm1):
        mu = self.zp_mu(hq_tm1)
        std = self.zp_std(hq_tm1)
        dist = Normal(0, 1)
        e = dist.sample(std.shape)
        return(mu + e * std, mu, std)
    
    def zq_from_hq_tm1(self, hq_tm1, o_t, prev_action):
        x = torch.cat([o_t, prev_action], -1)
        x = self.new_o(x)
        x = torch.cat([hq_tm1, x], -1) 
        mu = self.zq_mu(x)
        std = self.zq_std(x)
        dist = Normal(0, 1)
        e = dist.sample(std.shape)
        return(mu + e * std, mu, std)
            
    # zp for hp, or zq for hq.                 
    def h(self, z_t, hq_tm1 = None):
        _, h_t = self.gru(z_t, hq_tm1.permute(1, 0, 2))  
        return(h_t.permute(1, 0, 2))
    
    def predict_o(self, hq_tm1):
        return(self.pred_o(hq_tm1))
    
    def get_Q_1(self, hq_t, action):
        x = torch.cat((hq_t, action), dim=-1)
        Q = self.Q_1(x)
        return(Q)
    
    def get_Q_2(self, hq_t, action):
        x = torch.cat((hq_t, action), dim=-1)
        Q = self.Q_2(x)
        return(Q)
    
    # Ted got this from: https://github.com/BY571/Soft-Actor-Critic-and-Extensions
    def a_mu_std(self, h_t):
        x = self.a(h_t)
        mu = self.a_mu(x)
        log_std = self.a_log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()
        return(mu, std)

    def evaluate_actor(self, h, epsilon=1e-6):
        mu, std = self.a_mu_std(h)
        dist = Normal(0, 1)
        e = dist.sample(std.shape).to(self.args.device)
        action = torch.tanh(mu + e * std)
        log_prob = Normal(mu, std).log_prob(mu + e * std) - \
            torch.log(1 - action.pow(2) + epsilon)
        log_prob = torch.mean(log_prob, -1).unsqueeze(-1)
        return(action, log_prob)

    def get_action(self, h):
        mu, std = self.a_mu_std(h)
        dist = Normal(0, 1)
        e = dist.sample(std.shape).to(self.args.device)
        action = torch.tanh(mu + e * std)
        return(action[0])
    


if __name__ == "__main__":
    
    args = default_args
    args.device = "cuda"
    
    model = Model(args)
    
    print("\n\n")
    print(model)
    print()
    print(torch_summary(model, ((1,default_args.h_size))))

# %%
