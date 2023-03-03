#%%

import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import torch.optim as optim
from blitz.losses import kl_divergence_from_nn as b_kl_loss

import numpy as np

from utils import default_args, dkl
from maze import action_size
from buffer import RecurrentReplayBuffer
from models import Forward, Actor, Critic



import torch 

from maze import T_Maze, action_size



class Agent:
    
    def __init__(self, action_prior="normal", args = default_args):
        
        self.args = args
        self.steps = 0
        self.action_size = action_size
        
        self.target_entropy = self.args.target_entropy 
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
        self.memory = RecurrentReplayBuffer(self.args)

    def act(self, h):
        action = self.actor.get_action(h)
        return action
    
    
    
    def learn(self, batch_size, epochs):
                
        self.steps += 1

        obs, actions, rewards, dones, masks = self.memory.sample(batch_size)
        batch_size = rewards.shape[0] ; steps = rewards.shape[1]
        
        next_obs = obs[:,1:]
        
        all_actions = torch.cat([torch.zeros(actions[:,0].unsqueeze(1).shape), actions], dim = 1)
        prev_actions = all_actions[:,:-1]
        
        print("\n\n")
        print("obs:\t{}.\nactions:\t{}.\nrewards:\t{}.\ndones:\t{}.\nmasks:\t{}.".format(
            obs.shape, actions.shape, rewards.shape, dones.shape, masks.shape))
        print("\n\n")
                
        
        
        # Train Forward
        pred_obs, dkls = self.learning_phase(batch_size, steps, obs, all_actions)
        hqs = self.active_inference_phase(batch_size, steps, obs, all_actions)
                

        obs_errors = F.mse_loss(pred_obs, next_obs.detach(), reduction = "none") # Is this correct?
        z_errors = dkls
        errors = torch.cat([obs_errors, z_errors], -1) * masks.detach() # plus complexity?
        forward_loss = errors.sum()
        
        self.forward_opt.zero_grad()
        forward_loss.backward()
        self.forward_opt.step()
        
        
        
        dkl_changes = 0 ; dkl_change = 0 # No longer relevent
        
            
        
        # Get curiosity          
        naive_curiosity = self.args.naive_eta * errors.sum(-1).detach()
        free_curiosity = self.args.free_eta * z_errors.sum(-1).detach()
        if(self.args.curiosity == "naive"):  curiosity = naive_curiosity.unsqueeze(-1)
        elif(self.args.curiosity == "free"): curiosity = free_curiosity.unsqueeze(-1)
        else:                                curiosity = torch.zeros(rewards.shape)
        
        extrinsic = torch.mean(rewards*masks.detach()).item()
        intrinsic_curiosity = curiosity.sum().item()
        print("Rewards: {}. Curiosity: {}.".format(rewards.shape, curiosity.shape))
        rewards += curiosity
        
                
                
        # Train critics
        new_actions, log_pis_next = self.actor.evaluate(hqs[:,1:].detach())
        print("\n\n")
        print("new actions: {}. log_pis: {}.".format(new_actions.shape, log_pis_next.shape))
        print("\n\n")
        Q_target1_next = self.critic1_target(hqs[:,1:].detach(), new_actions.detach())
        Q_target2_next = self.critic2_target(hqs[:,1:].detach(), new_actions.detach())
        Q_target_next = torch.min(Q_target1_next, Q_target2_next)
        print("\n\n")
        print("Q_target_next: {}. rewards: {}. dones: {}.".format(Q_target_next.shape, rewards.shape, dones.shape))
        print("\n\n")
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
            _, log_pis = self.actor.evaluate(hqs[:,:-1].detach())
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
        
        
        
    def learning_phase(self, batch_size, steps, obs, all_actions):
        hq = torch.zeros((batch_size, 1, self.args.h_size))
        pred_obs = []
        mu_ps = [] ; std_ps = []
        mu_qs = [] ; std_qs = []
        for step in range(steps):
            o = obs[:,step].unsqueeze(1).detach()     
            prev_a = all_actions[:, step].unsqueeze(1).detach()    
            
            zp, mu_p, std_p = self.forward.zp_from_hq_tm1(hq)           
            zq, mu_q, std_q = self.forward.zq_from_hq_tm1(hq, o, prev_a) 
            hq = self.forward.h(zq, hq)    

            mu_qs.append(mu_q) ; std_qs.append(std_q)
            mu_ps.append(mu_p) ; std_ps.append(std_p) 
            pred_obs.append(self.forward.predict_o(hq))       
                
        mu_ps = torch.cat(mu_ps, -2) ; std_ps = torch.cat(std_ps, -2)
        mu_qs = torch.cat(mu_qs, -2) ; std_qs = torch.cat(std_qs, -2)
        pred_obs = torch.cat(pred_obs,-2)
                
        dkls = dkl(mu_qs, std_qs, mu_ps, std_ps) 

        print("\n\n")
        print("obs:\t{}.\npred:\t{}.\nmu_ps:\t{}.\nstd_ps:\t{}.\nmu_qs:\t{}.\nstd_qs:\t{}.\ndkls:\t{}.".format(
            obs[:,1:].shape, pred_obs.shape, mu_ps.shape, std_ps.shape, mu_qs.shape, std_qs.shape, dkls.shape))
        print("\n\n")
        return(pred_obs, dkls)
            
            

    def active_inference_phase(self, batch_size, steps, obs, all_actions):
        hqs = [] ; hq = torch.zeros((batch_size, 1, self.args.h_size))
        for step in range(steps+1):
            o = obs[:,step].unsqueeze(1).detach()     
            prev_a = all_actions[:, step].unsqueeze(1).detach()    
            
            zq, mu_q, std_q = self.forward.zq_from_hq_tm1(hq, o, prev_a) 
            hq = self.forward.h(zq, hq)    

            hqs.append(hq)
                
        hqs = torch.cat(hqs, -2)
                
        print("\n\n")
        print("hqs:\t{}.".format(
            hqs[:,1:].shape))
        print("\n\n")
        return(hqs)



    def interaction_with_environment(self, push = True):
        done = False
        t_maze = T_Maze()
        steps = 0
        with torch.no_grad():
            hq = torch.zeros((1, 1, self.args.h_size))     
            zp = torch.normal(0, 1, (1, 1, self.args.z_size))                         
            a  = torch.zeros((1, action_size))
            while(done == False):
                steps += 1
                o = t_maze.obs()      
                
                hp = self.forward.h(zp, hq) 
                zp, _, _ = self.forward.zp_from_hq_tm1(hq)          
                zq, _, _ = self.forward.zq_from_hq_tm1(hq, o.unsqueeze(0), a.unsqueeze(0))    
                hq = self.forward.h(zq, hq)                            
                
                a = self.act(hp)   
                action = a.squeeze(0).tolist()
                r, spot_name, done = t_maze.action(action[0], action[1])
                no = t_maze.obs()
                if(steps >= self.args.max_steps): done = True ; r = -1
                if(push): self.memory.push(o, a, r, no, done, done, self)
        return(r, spot_name)
        
# %%