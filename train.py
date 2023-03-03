#%%

import torch
import enlighten
from itertools import accumulate
from copy import deepcopy

from utils import default_args
from agent import Agent



class Trainer():
    def __init__(self, args = default_args, title = None):
        
        self.args = args
        self.title = title
        self.restart()
    
    def restart(self):
        self.e = 0
        self.agent = Agent(args = self.args)
        self.plot_dict = {
            "args" : self.args,
            "title" : self.title,
            "rewards" : [], "spot_names" : [], 
            "obs" : [], "z" : [], 
            "alpha" : [], "actor" : [], 
            "critic_1" : [], "critic_2" : [], 
            "extrinsic" : [], "intrinsic_curiosity" : [], 
            "intrinsic_entropy" : []}

    def train(self):
        self.agent.train()
        manager = enlighten.Manager(width = 150)
        E = manager.counter(total = self.args.epochs, desc = "{}:".format(self.title), unit = "ticks", color = "blue")
        while(True):
            E.update()
            r, spot_name = self.agent.interaction_with_environment()
            l, e, ic, ie = self.agent.learn(batch_size = self.args.batch_size, epochs = self.e)
            self.plot_dict["rewards"].append(r)
            self.plot_dict["spot_names"].append(spot_name)
            self.plot_dict["obs"].append(l[0][0])
            self.plot_dict["z"].append(l[0][1])
            self.plot_dict["alpha"].append(l[0][2])
            self.plot_dict["actor"].append(l[0][3])
            self.plot_dict["critic_1"].append(l[0][4])
            self.plot_dict["critic_2"].append(l[0][5])
            self.plot_dict["extrinsic"].append(e)
            self.plot_dict["intrinsic_curiosity"].append(ic)
            self.plot_dict["intrinsic_entropy"].append(ie)
            self.e += 1
            if(self.e >= self.args.epochs): 
                print("\n\nDone training!")
                break
        self.plot_dict["rewards"] = list(accumulate(self.plot_dict["rewards"]))
        
        for key in self.plot_dict.keys():
            if(key in ["args", "title"]): pass 
            else:
                self.plot_dict[key] = [v for i, v in enumerate(self.plot_dict[key]) if (i+1)%self.args.keep_data==0 or i==0 or (i+1)==len(self.plot_dict[key])]
        
        min_max_dict = {key : [] for key in self.plot_dict.keys()}
        for key in min_max_dict.keys():
            if(not key in ["args", "title", "spot_names"]):
                minimum = None ; maximum = None 
                l = self.plot_dict[key]
                l = deepcopy(l)
                l = [_ for _ in l if _ != None]
                if(l != []):
                    if(minimum == None):    minimum = min(l)
                    elif(minimum > min(l)): minimum = min(l)
                    if(maximum == None):    maximum = max(l) 
                    elif(maximum < max(l)): maximum = max(l)
                min_max_dict[key] = (minimum, maximum)
        return(self.plot_dict, min_max_dict)
    
# %%