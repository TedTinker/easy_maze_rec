#%% 

import argparse
import os 

if(os.getcwd().split("/")[-1] != "easy_maze"): os.chdir("easy_maze")
print(os.getcwd())

import torch
from blitz.modules import BayesianLinear, BayesianLSTM
from blitz.modules.base_bayesian_module import BayesianModule
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

# Meta 
parser.add_argument("--arg_title",          type=str,   default = "default") 
parser.add_argument("--name",               type=str,   default = "default") 
parser.add_argument("--id",                 type=int,   default = 0)
parser.add_argument('--device',             type=str,   default = "cpu")

# Maze 
parser.add_argument('--max_steps',          type=int,   default = 10)
parser.add_argument('--wall_punishment',    type=int,   default = -1)

# Module 
parser.add_argument('--hidden',             type=int,   default = 16)
parser.add_argument('--forward_sum_bayes',  type=bool,  default = True)
parser.add_argument("--dkl_rate",           type=float, default = .00005)   # Scale bayesian dkl
parser.add_argument('--forward_lr',         type=float, default = .01)
parser.add_argument('--clone_lr',           type=float, default = .001)
parser.add_argument('--actor_lr',           type=float, default = .01) 
parser.add_argument('--critic_lr',          type=float, default = .01) 
parser.add_argument('--alpha_lr',           type=float, default = .01) 

# Memory buffer
parser.add_argument('--capacity',           type=int,   default = 100)
parser.add_argument('--replacement',        type=str,   default = "index")
parser.add_argument('--selection',          type=str,   default = "uniform")
parser.add_argument('--power',              type=float, default = 1)

# Training 
parser.add_argument('--epochs',             type=int,   default = 1000)
parser.add_argument('--batch_size',         type=int,   default = 8)
parser.add_argument('--GAMMA',              type=int,   default = .99)
parser.add_argument("--d",                  type=int,   default = 2)        # Delay to train actors
parser.add_argument("--alpha",              type=str,   default = 0)        # Soft-Actor-Critic entropy aim
parser.add_argument("--target_entropy",     type=float, default = -2)       # Soft-Actor-Critic entropy aim
parser.add_argument("--naive_eta",          type=float, default = 1)        # Scale curiosity
parser.add_argument("--free_eta",           type=float, default = .005)      # Scale curiosity
parser.add_argument("--tau",                type=float, default = .05)      # For soft-updating target critics
parser.add_argument("--sample_elbo",        type=int,   default = 5)        # Samples for elbo
parser.add_argument("--curiosity",          type=str,   default = "none")   # Which kind of curiosity

# DKL Guesser 
parser.add_argument('--use_guesser',        type=bool,  default = False)
parser.add_argument('--dkl_hidden',         type=int,   default = 32)
parser.add_argument('--dkl_batch_size',     type=int,   default = 16)
parser.add_argument('--dkl_guesser_lr',     type=float, default = .01) 
parser.add_argument('--dkl_collect',        type=int,   default = 1) 
parser.add_argument('--dkl_buffer_capacity',type=int,   default = 128) 

# Saving data
parser.add_argument('--keep_data',          type=int,   default = 1)



try:
    default_args = parser.parse_args([])
    try:    args    = parser.parse_args()
    except: args, _ = parser.parse_known_args()
except:
    import sys ; sys.argv=[''] ; del sys           # Comment this out when using bash
    default_args = parser.parse_args([])
    try:    args    = parser.parse_args()
    except: args, _ = parser.parse_known_args()

if(default_args.alpha == "None"): default_args.alpha = None
if(args.alpha == "None"):         args.alpha = None



def get_args_name(default_args, args):
    name = "" ; first = True
    for arg in vars(default_args):
        if(arg in ["arg_title", "id"]): pass 
        else: 
            default, this_time = getattr(default_args, arg), getattr(args, arg)
            if(this_time == default): pass
            else: 
                if first: first = False
                else: name += "_"
                name += "{}_{}".format(arg, this_time)
    if(name == ""): name = "default" 
    return(name)



if(args.name[:3] != "___"):
    name = get_args_name(default_args, args)
    args.name = name

folder = "saved/" + args.arg_title
if(args.arg_title[:3] != "___" and args.arg_title != "default"):
    try: os.mkdir(folder)
    except: pass
if(default_args.alpha == "None"): default_args.alpha = None
if(args.alpha == "None"):         args.alpha = None

if(args == default_args): print("Using default arguments.")
else:
    for arg in vars(default_args):
        default, this_time = getattr(default_args, arg), getattr(args, arg)
        if(this_time == default): pass
        else: print("{}:\n\tDefault:\t{}\n\tThis time:\t{}".format(arg, default, this_time))
print("\n\n")



def init_weights(m):
    try:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)
    except: pass
    
    

def weights(model):
    weight_mu = [] ; weight_sigma = []
    bias_mu = [] ;   bias_sigma = []
    for module in model.modules():
        if isinstance(module, (BayesianModule)):
            if isinstance(module, (BayesianLinear)):
                weight_mu.append(module.weight_sampler.mu.clone().flatten())
                weight_sigma.append(torch.log1p(torch.exp(module.weight_sampler.rho.clone().flatten())))
                bias_mu.append(module.bias_sampler.mu.clone().flatten()) 
                bias_sigma.append(torch.log1p(torch.exp(module.bias_sampler.rho.clone().flatten())))
            if isinstance(module, (BayesianLSTM)):
                weight_mu.append(module.weight_ih_sampler.mu.clone().flatten())
                weight_sigma.append(torch.log1p(torch.exp(module.weight_ih_sampler.rho.clone().flatten())))
                weight_mu.append(module.weight_hh_sampler.mu.clone().flatten())
                weight_sigma.append(torch.log1p(torch.exp(module.weight_hh_sampler.rho.clone().flatten())))
                bias_mu.append(module.bias_sampler.mu.clone().flatten()) 
                bias_sigma.append(torch.log1p(torch.exp(module.bias_sampler.rho.clone().flatten())))
    if(weight_mu == []):
        return(torch.zeros([1]), torch.zeros([1]), torch.zeros([1]), torch.zeros([1]))
    return(
        torch.cat(weight_mu, -1).to("cpu"),
        torch.cat(weight_sigma, -1).to("cpu"),
        torch.cat(bias_mu, -1).to("cpu"),
        torch.cat(bias_sigma, -1).to("cpu"))
    

    
def dkl(mu_1, sigma_1, mu_2, sigma_2):
    sigma_1 = torch.pow(sigma_1, 2)
    sigma_2 = torch.pow(sigma_2, 2)
    term_1 = torch.pow(mu_2 - mu_1, 2) / sigma_2 
    term_2 = sigma_1 / sigma_2 
    term_3 = torch.log(term_2)
    out = (.5 * (term_1 + term_2 - term_3 - 1)).sum()
    out = torch.nan_to_num(out)
    return(out)


# %%
