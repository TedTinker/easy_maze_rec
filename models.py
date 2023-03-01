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
        
        self.zp = nn.Linear(args.h_size,               args.z_size)
        self.zq = nn.Linear(args.h_size + obs_size,    args.z_size) # obs should go through network. Prev action, too?
        self.o  = nn.Linear(args.h_size,               obs_size)
        
    def zp_from_hq_tm1(self, hq_tm1):
        return(self.zp(hq_tm1))
    
    def zq_from_hq_t_and_o_t(self, hp_t, o_t):
        x = torch.cat([hp_t, o_t], -1) 
        return(self.zq(x))
            
    # zp and hq to make hp, or zq and hq to make hq.                 
    def h(self, z_t, hq_tm1 = None):
        _, h_t = self.gru(z_t, hq_tm1.permute(1, 0, 2))  
        return(h_t.permute(1, 0, 2))
    
    def forward(self, hq_tm1):
        return(self.o(hq_tm1))
    
        
        
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
    
    
    
    w_mu, w_sigma, b_mu, b_sigma = weights(forward)
    
    errors_shape  = (3, 8, 10, 1)
    w_mu_shape    = (3, w_mu.shape[0])
    w_sigma_shape = (3, w_sigma.shape[0])
    b_mu_shape    = (3, b_mu.shape[0])
    b_sigma_shape = (3, b_sigma.shape[0])
    
    dkl_guesser = DKL_Guesser(args)

    print("\n\n")
    print(dkl_guesser)
    print()
    print(torch_summary(dkl_guesser, (
        errors_shape, 
        w_mu_shape, w_sigma_shape, b_mu_shape, b_sigma_shape,
        w_mu_shape, w_sigma_shape, b_mu_shape, b_sigma_shape)))
    


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
