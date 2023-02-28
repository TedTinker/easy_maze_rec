#%%

import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import torch.optim as optim
from blitz.losses import kl_divergence_from_nn as b_kl_loss

import numpy as np

from utils import default_args, dkl, weights
from maze import action_size
from buffer import RecurrentReplayBuffer, DKL_Buffer
from models import Forward, Actor, Critic



class Agent:
    
    def __init__(self, action_prior="normal", args = default_args):
        
        self.args = args
        self.steps = 0
        self.action_size = action_size
        
        self.target_entropy = self.args.target_entropy # -dim(A)
        self.alpha = 1
        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha_opt = optim.Adam(params=[self.log_alpha], lr=self.args.alpha_lr, weight_decay=0) 
        self._action_prior = action_prior
        
        self.eta = 1
        self.log_eta = torch.tensor([0.0], requires_grad=True)
        
        self.forward = Forward(self.args)
        self.forward_opt = optim.Adam(self.forward.parameters(), lr=self.args.forward_lr, weight_decay=0)   
                           
        self.actor = Actor(self.args)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=self.args.actor_lr, weight_decay=0)     
        
        self.critic1 = Critic(self.args)
        self.critic1_opt = optim.Adam(self.critic1.parameters(), lr=self.args.critic_lr, weight_decay=0)
        self.critic1_target = Critic(self.args)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2 = Critic(self.args)
        self.critic2_opt = optim.Adam(self.critic2.parameters(), lr=self.args.critic_lr, weight_decay=0) 
        self.critic2_target = Critic(self.args)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        self.restart_memory()
        
    def restart_memory(self):
        self.dkl_buffer = DKL_Buffer(self.args)
        self.memory = RecurrentReplayBuffer(self.args)

    def act(self, h):
        action = self.actor.get_action(h)
        return action
    
    
    
    def learn(self, batch_size, epochs):
                
        self.steps += 1

        all_obs, actions, rewards, dones, masks = self.memory.sample(batch_size)
        
        next_obs = all_obs[:,1:]
        obs = all_obs[:,:-1]
        
        all_actions = torch.cat([torch.zeros(actions[:,0].unsqueeze(1).shape), actions], dim = 1)
        prev_actions = all_actions[:,:-1]
        
        
        
        # Train Forward
        
        # Make sure all this stuff is arranged right! 
        hqs = [torch.zeros((obs.shape[0], 1, self.args.h_size))]
        zps = [torch.normal(0, 1, (obs.shape[0], 1, self.args.z_size))]
        zqs = []
        pred_obs = []
        
        for step in range(obs.shape[1]):
            zps.append(self.forward.zp_from_hq_tm1(hqs[-1]))
            zqs.append(self.forward.zq_from_hq_t_and_o_t(hqs[-1], obs[:,step].unsqueeze(1).detach()))
            hqs.append(self.forward.h(zqs[-1], hqs[-1]))
            pred_obs.append(self.forward(hqs[-1]))    
            
        hqs = torch.cat(hqs, -2)
        zps = torch.cat(zps, -2) ; zqs = torch.cat(zqs, -2)
        pred_obs = torch.cat(pred_obs,-2)
                            
        obs_errors = F.mse_loss(pred_obs, next_obs.detach(), reduction = "none") * masks.detach()
        z_errors = F.mse_loss(zps[:,1:], zqs, reduction = "none") * masks.detach()
        errors = torch.cat([obs_errors, z_errors], -1)
        forward_loss = errors.sum()
        
        self.forward_opt.zero_grad()
        forward_loss.backward()
        self.forward_opt.step()
        
        
        dkl_changes = 0 ; dkl_change = 0 # Do this based on difference between forward's z bayes layers?
        
            
        
        # Get curiosity          
        naive_curiosity = self.args.naive_eta * errors.sum(-1)
        free_curiosity = self.args.free_eta * dkl_changes  
        if(self.args.curiosity == "naive"):  curiosity = naive_curiosity.unsqueeze(-1)
        elif(self.args.curiosity == "free"): curiosity = free_curiosity.unsqueeze(-1)
        else:                                curiosity = torch.zeros(rewards.shape)
        
        extrinsic = torch.mean(rewards*masks.detach()).item()
        intrinsic_curiosity = curiosity.sum().item()
        rewards += curiosity
        
                
                
        # Train critics
        next_actions, log_pis_next = self.actor.evaluate(hqs.detach())
        next_actions = next_actions[:,1:] ; log_pis_next = log_pis_next[:,1:]
        Q_target1_next = self.critic1_target(hqs[:,1:].detach(), next_actions.detach())
        Q_target2_next = self.critic2_target(hqs[:,1:].detach(), next_actions.detach())
        Q_target_next = torch.min(Q_target1_next, Q_target2_next)
        if self.args.alpha == None: Q_targets = rewards.cpu() + (self.args.GAMMA * (1 - dones.cpu()) * (Q_target_next.cpu() - self.alpha * log_pis_next.cpu()))
        else:                       Q_targets = rewards.cpu() + (self.args.GAMMA * (1 - dones.cpu()) * (Q_target_next.cpu() - self.args.alpha * log_pis_next.cpu()))
        
        Q_1 = self.critic1(hqs[:,:-1].detach(), actions.detach())
        critic1_loss = 0.5*F.mse_loss(Q_1*masks.detach().cpu(), Q_targets.detach()*masks.detach().cpu())
        self.critic1_opt.zero_grad()
        critic1_loss.backward()
        self.critic1_opt.step()
        
        Q_2 = self.critic2(hqs[:,:-1].detach(), actions.detach())
        critic2_loss = 0.5*F.mse_loss(Q_2*masks.detach().cpu(), Q_targets.detach()*masks.detach().cpu())
        self.critic2_opt.zero_grad()
        critic2_loss.backward()
        self.critic2_opt.step()
        
        
        
        # Train alpha
        if self.args.alpha == None:
            actions, log_pis = self.actor.evaluate(hqs[:,:-1].detach())
            alpha_loss = -(self.log_alpha.cpu() * (log_pis.cpu() + self.target_entropy).detach().cpu())*masks.detach().cpu()
            alpha_loss = alpha_loss.sum() / masks.sum()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            self.alpha = torch.exp(self.log_alpha) 
        else:
            alpha_loss = None
            
            
        
        # Train actor
        if self.steps % self.args.d == 0:
            if self.args.alpha == None: alpha = self.alpha 
            else:                       
                alpha = self.args.alpha
                actions, log_pis = self.actor.evaluate(hqs[:,:-1].detach())

            if self._action_prior == "normal":
                loc = torch.zeros(self.action_size, dtype=torch.float64)
                scale_tril = torch.tensor([[1, 0], [1, 1]], dtype=torch.float64)
                policy_prior = MultivariateNormal(loc=loc, scale_tril=scale_tril)
                policy_prior_log_probs = policy_prior.log_prob(actions.cpu()).unsqueeze(-1)
            elif self._action_prior == "uniform":
                policy_prior_log_probs = 0.0
            Q = torch.min(
                self.critic1(hqs[:,:-1].detach(), actions), 
                self.critic2(hqs[:,:-1].detach(), actions)).sum(-1).unsqueeze(-1)
            intrinsic_entropy = torch.mean((alpha * log_pis.cpu())*masks.detach().cpu()).item()
            actor_loss = (alpha * log_pis.cpu() - policy_prior_log_probs - Q.cpu())*masks.detach().cpu()
            actor_loss = actor_loss.sum() / masks.sum()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            self.soft_update(self.critic1, self.critic1_target, self.args.tau)
            self.soft_update(self.critic2, self.critic2_target, self.args.tau)
            
        else:
            intrinsic_entropy = None
            actor_loss = None
        
        obs_loss = obs_errors.sum().item()
        z_loss = z_errors.sum().item()
        if(alpha_loss != None): alpha_loss = alpha_loss.item()
        if(actor_loss != None): actor_loss = actor_loss.item()
        if(critic1_loss != None): critic1_loss = critic1_loss.item()
        if(critic2_loss != None): critic2_loss = critic2_loss.item()
        losses = np.array([[obs_loss, z_loss, alpha_loss, actor_loss, critic1_loss, critic2_loss]])
        
        return(losses, extrinsic, intrinsic_curiosity, intrinsic_entropy, dkl_change, naive_curiosity.sum().detach(), free_curiosity)
                     
    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def state_dict(self):
        return(
            self.forward.state_dict(),
            self.actor.state_dict(),
            self.critic1.state_dict(),
            self.critic1_target.state_dict(),
            self.critic2.state_dict(),
            self.critic2_target.state_dict())

    def load_state_dict(self, state_dict):
        self.forward.load_state_dict(state_dict[0])
        self.actor.load_state_dict(state_dict[1])
        self.critic1.load_state_dict(state_dict[2])
        self.critic1_target.load_state_dict(state_dict[3])
        self.critic2.load_state_dict(state_dict[4])
        self.critic2_target.load_state_dict(state_dict[5])
        self.memory = RecurrentReplayBuffer(self.args)

    def eval(self):
        self.forward.eval()
        self.actor.eval()
        self.critic1.eval()
        self.critic1_target.eval()
        self.critic2.eval()
        self.critic2_target.eval()

    def train(self):
        self.forward.train()
        self.actor.train()
        self.critic1.train()
        self.critic1_target.train()
        self.critic2.train()
        self.critic2_target.train()
        
# %%
