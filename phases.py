import torch 

from maze import T_Maze, action_size



def learning_phase(agent):
    hqs = [] ; hq = torch.zeros((batch_size, 1, self.args.h_size))
            pred_obs = []
            mu_ps = [] ; std_ps = []
            mu_qs = [] ; std_qs = []
            for step in range(steps+1):
                o = obs[:,step].unsqueeze(1).detach()     
                prev_a = all_actions[:, step].unsqueeze(1).detach()    
                
                zp, mu_p, std_p = self.forward.zp_from_hq_tm1(hq)           
                zq, mu_q, std_q = self.forward.zq_from_hq_tm1(hq, o, prev_a) 
                hq = self.forward.h(zq, hq)    

                hqs.append(hq)      
                if(step != steps):
                    mu_qs.append(mu_q) ; std_qs.append(std_q)
                    mu_ps.append(mu_p) ; std_ps.append(std_p) 
                    pred_obs.append(self.forward(hq))       
                    
        hqs = torch.cat(hqs, -2) 
        mu_ps = torch.cat(mu_ps, -2) ; std_ps = torch.cat(std_ps, -2)
        mu_qs = torch.cat(mu_qs, -2) ; std_qs = torch.cat(std_qs, -2)
        pred_obs = torch.cat(pred_obs,-2)
                
        dkls = dkl(mu_qs, std_qs, mu_ps, std_ps) 
        
        print("\n\n")
        print("hqs:\t{}.\nobs:\t{}.\npred:\t{}.\nmu_ps:\t{}.\nstd_ps:\t{}.\nmu_qs:\t{}.\nstd_qs:\t{}.\ndkls:\t{}.".format(
            hqs.shape, next_obs.shape, pred_obs.shape, mu_ps.shape, std_ps.shape, mu_qs.shape, std_qs.shape, dkls.shape))
        print("\n\n")
        
        

def active_inference_phase():
    pass 



def interaction_with_environment(agent, push = True):
    done = False
    t_maze = T_Maze()
    steps = 0
    with torch.no_grad():
        hq = torch.zeros((1, 1, agent.args.h_size))     
        zp = torch.normal(0, 1, (1, 1, agent.args.z_size))                         
        a  = torch.zeros((1, action_size))
        while(done == False):
            steps += 1
            o = t_maze.obs()      
            
            hp = agent.forward.h(zp, hq) 
            zp, _, _ = agent.forward.zp_from_hq_tm1(hq)          
            zq, _, _ = agent.forward.zq_from_hq_tm1(hq, o.unsqueeze(0), a.unsqueeze(0))    
            hq = agent.forward.h(zq, hq)                            
            
            a = agent.act(hq)   
            action = a.squeeze(0).tolist()
            r, spot_name, done = t_maze.action(action[0], action[1])
            no = t_maze.obs()
            if(steps >= agent.args.max_steps): done = True ; r = -1
            if(push): agent.memory.push(o, a, r, no, done, done, agent)
    return(r, spot_name)