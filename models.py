#%% 

import torch
from torch import nn 
from torch.distributions import Normal
import torch.nn.functional as F
from torchinfo import summary as torch_summary
from blitz.modules import BayesianLinear, BayesianLSTM

from utils import default_args, init_weights, weights
from maze import obs_size, action_size



class Forward(nn.Module):

    def __init__(self, args = default_args):
        super(Forward, self).__init__()

        self.args = args
        
        self.gru = nn.GRU(
            input_size =  self.args.z_size,
            hidden_size = self.args.h_size,
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
        self.o        = nn.Sequential(
            nn.Linear(obs_size + action_size, args.h_size),
            nn.LeakyReLU())
        self.pred_o   = nn.Sequential(
            nn.Linear(args.h_size, obs_size),
            nn.Sigmoid())
        
    def zp_from_hq_tm1(self, hq_tm1):
        mu = self.zp_mu(hq_tm1)
        std = self.zp_std(hq_tm1)
        dist = Normal(0, 1)
        e      = dist.sample(std.shape)
        return(mu + e * std, mu, std)
    
    def zq_from_hq_tm1(self, hq_tm1, o_t, prev_action):
        x = torch.cat([o_t, prev_action], -1)
        x = self.o(x)
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
    
    def forward(self, hq_tm1):
        return(self.pred_o(hq_tm1))
    
        
        
class Actor(nn.Module):

    def __init__(self, args = default_args, log_std_min=-20, log_std_max=2):
        super(Actor, self).__init__()
        
        self.args = args

        self.log_std_min = log_std_min ; self.log_std_max = log_std_max
                
        self.lin = nn.Sequential(
            nn.Linear(args.h_size, args.hidden),
            nn.LeakyReLU())
        self.mu = nn.Linear(args.hidden, action_size)
        self.log_std_linear = nn.Linear(args.hidden, action_size)

        self.lin.apply(init_weights)
        self.mu.apply(init_weights)
        self.log_std_linear.apply(init_weights)
        self.to(self.args.device)

    def forward(self, h):
        x = self.lin(h)
        mu = self.mu(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return(mu, log_std)

    def evaluate(self, h, epsilon=1e-6):
        mu, log_std = self.forward(h)
        std = log_std.exp()
        dist = Normal(0, 1)
        e = dist.sample(std.shape).to(self.args.device)
        action = torch.tanh(mu + e * std)
        log_prob = Normal(mu, std).log_prob(mu + e * std) - \
            torch.log(1 - action.pow(2) + epsilon)
        log_prob = torch.mean(log_prob, -1).unsqueeze(-1)
        return(action, log_prob)

    def get_action(self, h):
        mu, log_std = self.forward(h)
        std = log_std.exp()
        dist = Normal(0, 1)
        e      = dist.sample(std.shape).to(self.args.device)
        action = torch.tanh(mu + e * std).cpu()
        return(action[0])
    
    
    
class Critic(nn.Module):

    def __init__(self, args = default_args):
        super(Critic, self).__init__()
        
        self.args = args
                        
        self.lin = nn.Sequential(
            nn.Linear(args.h_size + action_size, args.hidden),
            nn.LeakyReLU(),
            nn.Linear(args.hidden, 1))

        self.lin.apply(init_weights)
        self.to(args.device)

    def forward(self, h, action):
        x = torch.cat((h, action), dim=-1)
        x = self.lin(x).to("cpu")
        return(x)
    


if __name__ == "__main__":
    
    args = default_args
    args.device = "cuda"
    
    forward = Forward(args)
    
    print("\n\n")
    print(forward)
    print()
    print(torch_summary(forward, ((1,default_args.h_size))))
    
    

    actor = Actor(args)
    
    print("\n\n")
    print(actor)
    print()
    print(torch_summary(actor, ((1, default_args.h_size))))
    
    
    
    critic = Critic(args)
    
    print("\n\n")
    print(critic)
    print()
    print(torch_summary(critic, ((1, 10, obs_size), (1, 10, action_size), (1, 10, action_size))))

# %%
