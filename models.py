#%% 

import torch
from torch import nn 
from torch.distributions import Normal
import torch.nn.functional as F
from torchinfo import summary as torch_summary
from blitz.modules import BayesianLinear, BayesianLSTM

from utils import default_args, init_weights, weights
from maze import obs_size, action_size



class Summarizer(nn.Module):
    
    def __init__(self, args = default_args, bayes = False):
        super(Summarizer, self).__init__()
        
        self.args = args
        
        self.bayes = bayes
        if(bayes):
            self.lstm = BayesianLSTM(
                in_features = obs_size + action_size,
                out_features = self.args.hidden)
        else:
            self.lstm = nn.LSTM(
                input_size = obs_size + action_size,
                hidden_size = self.args.hidden,
                batch_first = True)
        
        self.lstm.apply(init_weights)
        
    def forward(self, obs, prev_action, hidden = None):
        obs = obs.to(self.args.device) ; prev_action = prev_action.to(self.args.device)
        x = torch.cat([obs, prev_action], -1)
        if(self.bayes): inner_state, hidden = self.lstm(x, hidden, None)    
        else:           inner_state, hidden = self.lstm(x, hidden)    
        return(inner_state, hidden)
    
    
    
class Forward(nn.Module):
    
    def __init__(self, args = default_args):
        super(Forward, self).__init__()
        
        self.args = args
        
        self.sum = Summarizer(self.args, args.forward_sum_bayes)
    
        self.lin = nn.Sequential(
            BayesianLinear(args.hidden + action_size, obs_size))
        
        self.lin.apply(init_weights)
        self.to(args.device)
        
    def forward(self, obs, prev_action, action, hidden = None):
        inner_state, hidden = self.sum(obs, prev_action, hidden)
        x = torch.cat([inner_state, action], -1)
        pred_obs = self.lin(x) # With sigmoid?
        return(pred_obs, inner_state, hidden)



def get_stats(stats):
    if(len(stats.shape) == 4): stats = stats.view(stats.shape[0], stats.shape[1]*stats.shape[2], stats.shape[3])
    mean   = torch.mean(stats, 1, False)
    q      = torch.quantile(stats, q = torch.tensor([0, .25, .5, .75, 1]).to(stats.device), dim = 1).permute(1, 2, 0).flatten(1)
    var    = torch.var(stats, dim = 1) 
    stats  = torch.cat([mean, q, var], dim = 1)
    return(stats)

new_dims = get_stats(torch.zeros((1,1,1))).shape[-1]



class DKL_Guesser(nn.Module):
    
    def __init__(self, args = default_args):
        super(DKL_Guesser, self).__init__()
        
        self.args = args
        
        self.errors = nn.Linear(1, args.dkl_hidden)
        self.weights = nn.Linear(4, args.dkl_hidden)
        self.bias = nn.Linear(4, args.dkl_hidden)
        self.dkl_out = nn.Linear((3 * new_dims + 1) * args.dkl_hidden, 1)
        
        self.errors.apply(init_weights)
        self.weights.apply(init_weights)
        self.bias.apply(init_weights)
        self.dkl_out.apply(init_weights)
        self.to(args.device)
        
    def forward(self, errors, 
                before_w_mu, before_w_sigma, before_b_mu, before_b_sigma,
                after_w_mu, after_w_sigma, after_b_mu, after_b_sigma):
                
        errors = self.errors(errors)
        errors_stats = get_stats(errors)
        
        change_w_mu  = after_w_mu - before_w_mu
        change_w_sigma = after_w_sigma - before_w_sigma
        weights = torch.cat([
            before_w_mu.unsqueeze(-1), change_w_mu.unsqueeze(-1),
            before_w_sigma.unsqueeze(-1), change_w_sigma.unsqueeze(-1)], dim = -1)
        weights = self.weights(weights)
        weights_stats = get_stats(weights)
        
        change_b_mu  = after_b_mu - before_b_mu
        change_b_sigma = after_b_sigma - before_b_sigma
        bias = torch.cat([
            before_b_mu.unsqueeze(-1), change_b_mu.unsqueeze(-1), 
            before_b_sigma.unsqueeze(-1), change_b_sigma.unsqueeze(-1)], dim = -1)
        bias = self.bias(bias)
        bias_stats = get_stats(bias)
        
        stats = torch.cat([errors_stats, weights_stats, bias_stats], dim = -1).unsqueeze(1).unsqueeze(1)
        stats = stats.tile((1, errors.shape[1], errors.shape[2], 1))
        together = torch.cat([errors, stats], dim = -1)
    
        dkl_out = self.dkl_out(together)
        return(F.softplus(dkl_out))
    
        
        
class Actor(nn.Module):

    def __init__(self, args = default_args, log_std_min=-20, log_std_max=2):
        super(Actor, self).__init__()
        
        self.args = args

        self.log_std_min = log_std_min ; self.log_std_max = log_std_max
        
        self.sum = Summarizer(self.args) 
        
        self.lin = nn.Sequential(
            nn.Linear(args.hidden, args.hidden),
            nn.LeakyReLU())
        self.mu = nn.Linear(args.hidden, action_size)
        self.log_std_linear = nn.Linear(args.hidden, action_size)

        self.lin.apply(init_weights)
        self.mu.apply(init_weights)
        self.log_std_linear.apply(init_weights)
        self.to(self.args.device)

    def forward(self, obs, prev_action, hidden = None):
        inner_state, hidden = self.sum(obs, prev_action, hidden)
        x = self.lin(inner_state)
        mu = self.mu(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return(mu, log_std, hidden)

    def evaluate(self, obs, prev_action, hidden = None, epsilon=1e-6):
        mu, log_std, hidden = self.forward(obs, prev_action, hidden)
        std = log_std.exp()
        dist = Normal(0, 1)
        e = dist.sample(std.shape).to(self.args.device)
        action = torch.tanh(mu + e * std)
        log_prob = Normal(mu, std).log_prob(mu + e * std) - \
            torch.log(1 - action.pow(2) + epsilon)
        log_prob = torch.mean(log_prob, -1).unsqueeze(-1)
        return(action, log_prob, hidden)

    def get_action(self, obs, prev_action, hidden = None):
        mu, log_std, hidden = self.forward(obs, prev_action, hidden)
        std = log_std.exp()
        dist = Normal(0, 1)
        e      = dist.sample(std.shape).to(self.args.device)
        action = torch.tanh(mu + e * std).cpu()
        return(action[0], hidden)
    
    
    
class Critic(nn.Module):

    def __init__(self, args = default_args):
        super(Critic, self).__init__()
        
        self.args = args
        
        self.sum = Summarizer(self.args)
                
        self.lin = nn.Sequential(
            nn.Linear(args.hidden + action_size, args.hidden),
            nn.LeakyReLU(),
            nn.Linear(args.hidden, 1))

        self.lin.apply(init_weights)
        self.to(args.device)

    def forward(self, obs, prev_action, action, hidden = None):
        inner_state, hidden = self.sum(obs, prev_action, hidden)
        x = torch.cat((inner_state, action), dim=-1)
        x = self.lin(x).to("cpu")
        return(x, hidden)
    


if __name__ == "__main__":
    
    args = default_args
    args.device = "cuda"
    
    forward = Forward(args)
    
    print("\n\n")
    print(forward)
    print()
    print(torch_summary(forward, ((1, 10, obs_size), (1, 10, action_size), (1, 10, action_size))))
    
    
    
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
    print(torch_summary(actor, ((1, 10, obs_size), (1, 10, action_size))))
    
    
    
    critic = Critic(args)
    
    print("\n\n")
    print(critic)
    print()
    print(torch_summary(critic, ((1, 10, obs_size), (1, 10, action_size), (1, 10, action_size))))

# %%
