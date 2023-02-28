#%%

import numpy as np
import torch
import torch.nn.functional as F

from collections import namedtuple
from utils import default_args
from maze import obs_size, action_size



class DKL_Buffer:
    
    def __init__(self, args = default_args):
        self.args = args
        
        self.index = 0 ; self.filled = 0
        self.errors = torch.zeros((args.dkl_buffer_capacity, args.batch_size, args.max_steps, 1))
        self.b_0 = None 
        self.b_1 = None 
        self.b_2 = None 
        self.b_3 = None 
        self.a_0 = None 
        self.a_1 = None 
        self.a_2 = None 
        self.a_3 = None 
        self.dkl_changes = torch.zeros((args.dkl_buffer_capacity, args.batch_size, args.max_steps, 1))

    def push(self, errors, before, after, dkl_changes):
        if(self.b_0 == None):
            self.b_0 = torch.zeros((self.args.dkl_buffer_capacity, before[0].shape[-1])) 
            self.b_1 = torch.zeros((self.args.dkl_buffer_capacity, before[1].shape[-1])) 
            self.b_2 = torch.zeros((self.args.dkl_buffer_capacity, before[2].shape[-1])) 
            self.b_3 = torch.zeros((self.args.dkl_buffer_capacity, before[3].shape[-1])) 
            self.a_0 = torch.zeros((self.args.dkl_buffer_capacity, after[0].shape[-1])) 
            self.a_1 = torch.zeros((self.args.dkl_buffer_capacity, after[1].shape[-1])) 
            self.a_2 = torch.zeros((self.args.dkl_buffer_capacity, after[2].shape[-1])) 
            self.a_3 = torch.zeros((self.args.dkl_buffer_capacity, after[3].shape[-1])) 

        errors = F.pad(errors, pad=(0, 0, 0, self.args.max_steps - errors.shape[1], 0, self.args.batch_size - errors.shape[0]), mode="constant", value=0)
        self.errors[self.index] = errors.unsqueeze(0).detach()
        self.b_0[self.index] = before[0].detach()
        self.b_1[self.index] = before[1].detach()
        self.b_2[self.index] = before[2].detach()
        self.b_3[self.index] = before[3].detach()
        self.a_0[self.index] = after[0].detach()
        self.a_1[self.index] = after[1].detach()
        self.a_2[self.index] = after[2].detach()
        self.a_3[self.index] = after[3].detach()
        dkl_changes = F.pad(dkl_changes, pad=(0, 0, 0, self.args.max_steps - dkl_changes.shape[1], 0, self.args.batch_size - dkl_changes.shape[0]), mode="constant", value=0)
        self.dkl_changes[self.index] = dkl_changes.unsqueeze(0).detach()
        
        self.index += 1 
        self.index %= self.args.dkl_buffer_capacity
        if(self.filled < self.args.dkl_buffer_capacity): self.filled += 1
                
    def sample(self, batch_size = None):
        if(batch_size == None): batch_size = self.args.batch_size
        if(self.filled < batch_size): return self.sample(self.filled)
        
        options = [i for i in range(self.args.dkl_buffer_capacity) if i <= self.filled]
        choices = np.random.choice(options,size=batch_size, replace=False)

        errors = self.errors[choices]
        before = [self.b_0[choices], self.b_1[choices], self.b_2[choices], self.b_3[choices]]        
        after = [self.a_0[choices], self.a_1[choices], self.a_2[choices], self.a_3[choices]]
        dkl_changes = self.dkl_changes[choices]
                
        return(errors, before, after, dkl_changes)
        


RecurrentBatch = namedtuple('RecurrentBatch', 'o a r d m')

def as_probas(positive_values: np.array) -> np.array:
    return positive_values / np.sum(positive_values)

def as_tensor_on_device(np_array: np.array):
    return torch.tensor(np_array).float().to("cpu")

class RecurrentReplayBuffer:

    """Use this version when num_bptt == max_episode_len"""
    
    def __init__(
        self, args = default_args, segment_len=None  # for non-overlapping truncated bptt, maybe need a large batch size
    ):
    
        self.args = args
            
        # pointers
      
        self.index = 1
        self.episode_ptr = 0
        self.time_ptr = 0
      
        # trackers
      
        self.starting_new_episode = True
        self.num_episodes = 0
      
        # hyper-parameters
      
        self.capacity = self.args.capacity
        self.o_dim = obs_size
        self.a_dim = action_size
      
        self.max_episode_len = args.max_steps + 1
      
        if segment_len is not None:
            assert self.max_episode_len % segment_len == 0  # e.g., if max_episode_len = 1000, then segment_len = 100 is ok
      
        self.segment_len = segment_len
      
        # placeholders

        self.o = np.zeros((self.args.capacity, self.max_episode_len + 1, self.o_dim), dtype='float32')
        self.a = np.zeros((self.args.capacity, self.max_episode_len, self.a_dim), dtype='float32')
        self.r = np.zeros((self.args.capacity, self.max_episode_len, 1), dtype='float32')
        self.d = np.zeros((self.args.capacity, self.max_episode_len, 1), dtype='float32')
        self.m = np.zeros((self.args.capacity, self.max_episode_len, 1), dtype='float32')
        
        self.i = np.zeros((self.args.capacity,), dtype = int)
        self.ep_len = np.zeros((self.args.capacity,), dtype='float32')
        self.ready_for_sampling = np.zeros((self.args.capacity,), dtype='int')
        
        self.curiosity = np.zeros(self.args.capacity)
      


    def push(self, o, a, r, no, d, cutoff, agent):
            
        # zero-out current slot at the beginning of an episode
      
        if self.starting_new_episode:
            self.o[self.episode_ptr] = 0
            self.a[self.episode_ptr] = 0
            self.r[self.episode_ptr] = 0
            self.d[self.episode_ptr] = 0
            self.m[self.episode_ptr] = 0
            
            self.i[self.episode_ptr] = self.index
            self.ep_len[self.episode_ptr] = 0
            self.ready_for_sampling[self.episode_ptr] = 0
            self.starting_new_episode = False
      
        # fill placeholders
        
        self.o[self.episode_ptr, self.time_ptr] = o
        self.a[self.episode_ptr, self.time_ptr] = a
        self.r[self.episode_ptr, self.time_ptr] = r
        self.d[self.episode_ptr, self.time_ptr] = d
        self.m[self.episode_ptr, self.time_ptr] = 1
        self.ep_len[self.episode_ptr] += 1
      
        if d or cutoff:
      
            # fill placeholders
        
            self.o[self.episode_ptr, self.time_ptr+1] = no
            self.ready_for_sampling[self.episode_ptr] = 1
            
            # reset curiosity weights if needed
            if(self.args.selection == "curiosity" or self.args.replacement == "curiosity"):
                o = torch.from_numpy(self.o).to(self.args.device)
                a = torch.from_numpy(self.a).to(self.args.device)
                m = torch.from_numpy(self.m).to(self.args.device)
                curiosity = agent.transitioner.DKL(
                    o[:,:-1], a,
                    o[:,1:], m).cpu().numpy().squeeze(-1)
                curiosity = np.sum(curiosity, 1)
                curiosity = curiosity[curiosity != 0]
                self.curiosity = curiosity
        
            # reset pointers
        
            self.index += 1
            self.time_ptr = 0
            
            if(self.args.replacement == "index"):
                self.episode_ptr = (self.episode_ptr+1) % self.capacity
                
            if(self.args.replacement == "curiosity"):
                if(self.num_episodes+1 < self.capacity):
                    self.episode_ptr += 1
                else:
                    self.episode_ptr = np.argmin(self.curiosity)
                    
            # update trackers
        
            self.starting_new_episode = True
            if self.num_episodes < self.capacity:
                self.num_episodes += 1

        else:
      
            # update pointers
        
            self.time_ptr += 1
            

        
    
    def sample(self, batch_size):
      
        if(self.num_episodes < batch_size): return self.sample(self.num_episodes)
      
        # sample episode indices
              
        options = np.where(self.ready_for_sampling == 1)[0]
        if(self.args.selection == "uniform"):
            self.args.power = 0
            self.args.selection = "index"
        
        if(self.args.selection == "index"):
            indices = self.i[options]
            indices = indices - indices.min() + 1
            indices = np.power(indices, self.args.power)
            weights = as_probas(indices)
            
        if(self.args.selection == "curiosity"):
            weights = as_probas(np.power(self.curiosity, self.args.power))
            
        choices = np.random.choice(options, p=weights, size=batch_size, replace=False)
        ep_lens_of_choices = self.ep_len[choices]
      
        if self.segment_len is None:
      
            # grab the corresponding numpy array
            # and save computational effort for lstm
        
            max_ep_len_in_batch = int(np.max(ep_lens_of_choices))
        
            o = self.o[choices][:, :max_ep_len_in_batch+1, :]
            a = self.a[choices][:, :max_ep_len_in_batch, :]
            r = self.r[choices][:, :max_ep_len_in_batch, :]
            d = self.d[choices][:, :max_ep_len_in_batch, :]
            m = self.m[choices][:, :max_ep_len_in_batch, :]
        
            # convert to tensors on the right device
        
            o = as_tensor_on_device(o).view((batch_size, max_ep_len_in_batch+1, self.o_dim))
            a = as_tensor_on_device(a).view(batch_size, max_ep_len_in_batch, self.a_dim)
            r = as_tensor_on_device(r).view(batch_size, max_ep_len_in_batch, 1)
            d = as_tensor_on_device(d).view(batch_size, max_ep_len_in_batch, 1)
            m = as_tensor_on_device(m).view(batch_size, max_ep_len_in_batch, 1)
            return RecurrentBatch(o, a, r, d, m)
      
        else:
      
            num_segments_for_each_item = np.ceil(ep_lens_of_choices / self.segment_len).astype(int)
        
            o = self.o[choices]
            a = self.a[choices]
            r = self.r[choices]
            d = self.d[choices]
            m = self.m[choices]
        
            o_seg = np.zeros((batch_size, self.segment_len + 1, self.o_dim))
            a_seg = np.zeros((batch_size, self.segment_len, self.a_dim))
            r_seg = np.zeros((batch_size, self.segment_len, 1))
            d_seg = np.zeros((batch_size, self.segment_len, 1))
            m_seg = np.zeros((batch_size, self.segment_len, 1))
        
            for i in range(batch_size):
                start_idx = np.random.randint(num_segments_for_each_item[i]) * self.segment_len
                o_seg[i] = o[i][start_idx:start_idx + self.segment_len + 1]
                a_seg[i] = a[i][start_idx:start_idx + self.segment_len]
                r_seg[i] = r[i][start_idx:start_idx + self.segment_len]
                d_seg[i] = d[i][start_idx:start_idx + self.segment_len]
                m_seg[i] = m[i][start_idx:start_idx + self.segment_len]
        
            o_seg = as_tensor_on_device(o_seg)
            a_seg = as_tensor_on_device(a_seg)
            r_seg = as_tensor_on_device(r_seg)
            d_seg = as_tensor_on_device(d_seg)
            m_seg = as_tensor_on_device(m_seg)
            return RecurrentBatch(o_seg, a_seg, r_seg, d_seg, m_seg)
