#%%
from random import choice
import torch



class Spot:
    
    def __init__(self, pos, exit_reward = None, name = "NONE"):
        self.pos = pos ; self.exit_reward = exit_reward
        self.name = name
        
        

class T_Maze:
    
    def __init__(self):
        self.maze = [
            Spot((0, 0)), Spot((0, 1)), 
            Spot((-1, 1), 1, "BAD"), Spot((1, 1)), Spot((1, 2)), 
            Spot((2, 2)), Spot((3, 2)), Spot((3, 1), 10, "GOOD")]
        self.agent_pos = (0, 0)
        
    def obs(self):
        right = 0 ; left = 0 ; up = 0 ; down = 0
        for spot in self.maze:
            if(spot.pos == (self.agent_pos[0]+1, self.agent_pos[1])): right = 1 
            if(spot.pos == (self.agent_pos[0]-1, self.agent_pos[1])): left = 1 
            if(spot.pos == (self.agent_pos[0], self.agent_pos[1]+1)): up = 1 
            if(spot.pos == (self.agent_pos[0], self.agent_pos[1]-1)): down = 1 
        return(torch.tensor([right, left, up, down]).unsqueeze(0).float())
        
    def action(self, x, y):
        if(abs(x) > abs(y)): y = 0 ; x = 1 if x > 0 else -1
        else:                x = 0 ; y = 1 if y > 0 else -1 
        new_pos = (self.agent_pos[0] + x, self.agent_pos[1] + y)
        for spot in self.maze:
            if(spot.pos == new_pos):
                self.agent_pos = new_pos 
                if(spot.exit_reward == None):
                    return(0, spot.name, False)
                else:
                    if(type(spot.exit_reward) == tuple):
                        return(choice(spot.exit_reward), spot.name, True)
                    else:
                        return(spot.exit_reward, spot.name, True)
        return(-1, "NONE", False)    
    
    def __str__(self):
        to_print = ""
        for y in [2, 1, 0]:
            for x in [-1, 0, 1, 2, 3]:
                portrayal = " "
                for spot in self.maze:
                    if(spot.pos == (x, y)): portrayal = "O"
                if(self.agent_pos == (x, y)): portrayal = "X"
                to_print += portrayal 
            to_print += "\n"
        return(to_print)
    
    
    
t_maze = T_Maze()
obs_size = t_maze.obs().shape[-1]
action_size = 2
    
if __name__ == "__main__":

    print(t_maze)
    print(t_maze.obs())
    
    actions = [[1,0], [0,1], [-1,0]]
    for action in actions:
        reward, name, done = t_maze.action(action[0], action[1])
        print("\n\n\n")
        print("Action: {}.".format(action), "\n") 
        print(t_maze)
        print("Reward: {}. Exit type: {}. Done: {}.".format(reward, name, done))
        print(t_maze.obs())


# %%
