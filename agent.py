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
from models import Forward, DKL_Guesser, Actor, Critic



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
        
        self.forward_clone = Forward(self.args)
        self.clone_opt = optim.Adam(self.forward_clone.parameters(), lr=self.args.clone_lr, weight_decay=0)
        
        self.dkl_guesser = DKL_Guesser(self.args)
        self.dkl_guesser_opt = optim.Adam(self.dkl_guesser.parameters(), lr=self.args.dkl_guesser_lr, weight_decay=0)
                           
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

    def act(self, obs, prev_action, hidden):
        action, hidden = self.actor.get_action(obs.unsqueeze(0), prev_action.unsqueeze(0), hidden)
        return action, hidden
    
    
    
    def learn(self, batch_size, epochs):
                
        self.steps += 1

        obs, actions, rewards, dones, masks = self.memory.sample(batch_size)
        
        all_obs = obs
        next_obs = obs[:,1:]
        obs = obs[:,:-1]
        
        all_actions = torch.cat([torch.zeros(actions[:,0].unsqueeze(1).shape), actions], dim = 1)
        prev_actions = all_actions[:,:-1]
        
        
                            
        # Train forward
        forward_errors = torch.zeros(rewards.shape)
        dkl_loss = 0
        for _ in range(self.args.sample_elbo):
            pred_obs, _, _ = self.forward(obs, prev_actions, actions)            
            errors = F.mse_loss(pred_obs, next_obs.detach(), reduction = "none") 
            errors = torch.sum(errors, -1).unsqueeze(-1)
            forward_errors += errors / self.args.sample_elbo
            dkl_loss += self.args.dkl_rate * b_kl_loss(self.forward) / self.args.sample_elbo
        forward_errors *= masks.detach()
        mse_loss = forward_errors.sum()
        forward_loss = mse_loss + dkl_loss
        
        old_state_dict = self.forward.state_dict() # For curiosity
        weights_before = weights(self.forward)
        self.forward_opt.zero_grad()
        forward_loss.backward()
        self.forward_opt.step()
        weights_after = weights(self.forward)
                
        dkl_change = dkl(weights_after[0], weights_after[1], weights_before[0], weights_before[1]) + \
            dkl(weights_after[2], weights_after[3], weights_before[2], weights_before[3])
        dkl_changes = torch.tile(dkl_change, rewards.shape)   
        dkl_changes *= masks
        if(dkl_changes.sum().item() != 0): dkl_change = dkl_changes.sum().item()
        
        
        
        if(epochs == 0 or (self.args.curiosity == "free" and epochs % self.args.dkl_collect == 0)):
            dkl_changes = torch.zeros(rewards.shape)
            for episode in range(rewards.shape[0]):
                for step in range(rewards.shape[1]):
                    if(masks[episode, step] == 0): dkl_changes[episode, step] = 0 ; break
                    self.forward_clone.load_state_dict(old_state_dict)
                    forward_errors_ = 0
                    dkl_loss_ = 0
                    for _ in range(self.args.sample_elbo):
                        pred_obs_, _, _ = self.forward_clone(
                            obs[episode, 0:step+1].unsqueeze(0), 
                            prev_actions[episode, 0:step+1].unsqueeze(0), 
                            actions[episode, 0:step+1].unsqueeze(0))
                        errors_ = F.mse_loss(pred_obs_[:,step], next_obs.detach()[episode, step].unsqueeze(0)) 
                        forward_errors_ += errors_ / self.args.sample_elbo
                        dkl_loss_ += self.args.dkl_rate * b_kl_loss(self.forward_clone) / self.args.sample_elbo
                    mse_loss_ = forward_errors_.sum()
                    forward_loss_ = mse_loss_ + dkl_loss_
            
                    weights_before_ = weights(self.forward_clone)
                    self.clone_opt.zero_grad()
                    forward_loss_.backward()
                    self.clone_opt.step()
                    weights_after_ = weights(self.forward_clone)
                    
                    dkl_change = dkl(weights_after_[0], weights_after_[1], weights_before_[0], weights_before_[1]) + \
                        dkl(weights_after_[2], weights_after_[3], weights_before_[2], weights_before_[3])
                    dkl_change, torch.exp(dkl_change)
                    dkl_changes[episode, step] = dkl_change
            dkl_changes *= masks 
            if(dkl_changes.sum().item() != 0): dkl_change = dkl_changes.sum().item()
            
            self.dkl_buffer.push(forward_errors, weights_before, weights_after, dkl_changes)
            
            
        
        # Train DKL Guesser
        forward_errors_, weights_before_, weights_after_, dkl_changes_ = self.dkl_buffer.sample()
        dkl_guess_ = self.dkl_guesser(
            forward_errors_,
            weights_before_[0], weights_before_[1], weights_before_[2], weights_before_[3],
            weights_after_[0],  weights_after_[1],  weights_after_[3],  weights_after_[3])
        
        guesser_loss = F.mse_loss(dkl_guess_, dkl_changes_)
        
        self.dkl_guesser_opt.zero_grad()
        guesser_loss.backward()
        self.dkl_guesser_opt.step()
            
            
                    
        dkl_guess = self.dkl_guesser(
            forward_errors.unsqueeze(0),
            weights_before[0].unsqueeze(0), weights_before[1].unsqueeze(0), weights_before[2].unsqueeze(0), weights_before[3].unsqueeze(0),
            weights_after[0].unsqueeze(0),  weights_after[1].unsqueeze(0),  weights_after[3].unsqueeze(0),  weights_after[3].unsqueeze(0))
                
        if(self.args.use_guesser == "True"): dkl_changes = dkl_guess
        
        
        
        # Get curiosity          
        naive_curiosity   = self.args.naive_eta   * forward_errors   
        naive_curiosity *= masks.detach() 
        free_curiosity = self.args.free_eta * dkl_changes  
        free_curiosity *= masks.detach()
        if(self.args.curiosity == "naive"):
            curiosity = naive_curiosity
            #print("\nMSE curiosity: {}, {}.\n".format(curiosity.shape, torch.sum(curiosity)))
        elif(self.args.curiosity == "free"):
            curiosity = free_curiosity
            #print("\nFEB curiosity: {}, {}.\n".format(curiosity.shape, torch.sum(curiosity)))
        else:
            curiosity = torch.zeros(rewards.shape)
        
        extrinsic = torch.mean(rewards*masks.detach()).item()
        intrinsic_curiosity = curiosity.sum().item()
        rewards += curiosity
        
        
        
        # Get inner states 
        with torch.no_grad():
            all_inner_states, _ = self.forward.sum(all_obs, all_actions)
            next_inner_states = all_inner_states[:,1:]
            inner_states = all_inner_states[:,:-1]
        
                
                
        # Train critics
        next_actions, log_pis_next, _ = self.actor.evaluate(all_obs.detach(), all_actions.detach())
        next_actions = next_actions[:,1:] ; log_pis_next = log_pis_next[:,1:]
        Q_target1_next, _ = self.critic1_target(next_obs.detach(), prev_actions.detach(), next_actions.detach())
        Q_target2_next, _ = self.critic2_target(next_obs.detach(), prev_actions.detach(), next_actions.detach())
        Q_target_next = torch.min(Q_target1_next, Q_target2_next)
        if self.args.alpha == None: Q_targets = rewards.cpu() + (self.args.GAMMA * (1 - dones.cpu()) * (Q_target_next.cpu() - self.alpha * log_pis_next.cpu()))
        else:                       Q_targets = rewards.cpu() + (self.args.GAMMA * (1 - dones.cpu()) * (Q_target_next.cpu() - self.args.alpha * log_pis_next.cpu()))
        
        Q_1, _ = self.critic1(obs.detach(), prev_actions.detach(), actions.detach())
        critic1_loss = 0.5*F.mse_loss(Q_1*masks.detach().cpu(), Q_targets.detach()*masks.detach().cpu())
        self.critic1_opt.zero_grad()
        critic1_loss.backward()
        self.critic1_opt.step()
        
        Q_2, _ = self.critic2(obs.detach(), prev_actions.detach(), actions.detach())
        critic2_loss = 0.5*F.mse_loss(Q_2*masks.detach().cpu(), Q_targets.detach()*masks.detach().cpu())
        self.critic2_opt.zero_grad()
        critic2_loss.backward()
        self.critic2_opt.step()
        
        
        
        # Train alpha
        if self.args.alpha == None:
            actions, log_pis, _ = self.actor.evaluate(obs.detach(), prev_actions.detach())
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
                actions, log_pis, _ = self.actor.evaluate(obs.detach(), prev_actions.detach())

            if self._action_prior == "normal":
                loc = torch.zeros(self.action_size, dtype=torch.float64)
                scale_tril = torch.tensor([[1, 0], [1, 1]], dtype=torch.float64)
                policy_prior = MultivariateNormal(loc=loc, scale_tril=scale_tril)
                policy_prior_log_probs = policy_prior.log_prob(actions.cpu()).unsqueeze(-1)
            elif self._action_prior == "uniform":
                policy_prior_log_probs = 0.0
            Q = torch.min(
                self.critic1(obs.detach(), prev_actions.detach(), actions)[0], 
                self.critic2(obs.detach(), prev_actions.detach(), actions)[0]).sum(-1).unsqueeze(-1)
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
        
        if(mse_loss != None): mse_loss = mse_loss.item()
        if(dkl_loss != None): dkl_loss = dkl_loss.item()
        if(guesser_loss != None): guesser_loss = guesser_loss.item()
        if(alpha_loss != None): alpha_loss = alpha_loss.item()
        if(actor_loss != None): actor_loss = actor_loss.item()
        if(critic1_loss != None): critic1_loss = critic1_loss.item()
        if(critic2_loss != None): critic2_loss = critic2_loss.item()
        losses = np.array([[mse_loss, dkl_loss, guesser_loss, alpha_loss, actor_loss, critic1_loss, critic2_loss]])
        
        return(losses, extrinsic, intrinsic_curiosity, intrinsic_entropy, dkl_change, naive_curiosity.sum().detach(), free_curiosity.sum().detach())
                     
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
