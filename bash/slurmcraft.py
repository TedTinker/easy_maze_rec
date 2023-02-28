#%%
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--comp",         type=str,  default = "deigo")
parser.add_argument("--arg_title",    type=str,  default = "default")
parser.add_argument("--post",         type=str,  default = "False")
try:    args = parser.parse_args()
except: args, _ = parser.parse_known_args()

import os 
try:    os.chdir("easy_maze/bash")
except: pass



if(args.comp == "deigo"):
	partition = """
#SBATCH --partition=short
#SBATCH --time 2:00:00
"""

elif(args.comp == "saion"):
	partition = """
#SBATCH --partition=taniu
#SBATCH --gres=gpu:1
#SBATCH --time 48:00:00
"""



slurm_dict = {}
f = open("slurms.txt", "r")
slurms = f.readlines()
for line in slurms:
    if(line[0] in ["\n", "#"]): pass 
    else:
        arg_title, text = line.split(":")
        slurm_dict[arg_title.strip()] = text.strip()
        
        
        
if(args.post == "False"):
    with open("main_{}.slurm".format(args.arg_title), "a") as f: # First empty it!
        f.write(
"""
#!/bin/bash -l
{}
#SBATCH --mem=2G
##SBATCH --constraint 32

module load singularity
singularity exec t_maze.sif python easy_maze/main.py --id ${{SLURM_ARRAY_TASK_ID}} --arg_title {} {}
    """.format(partition, args.arg_title, slurm_dict[args.arg_title])[1:])
        
        
        
else:
    if(args.arg_title[:3] == "___"): 
        slurm_dict[args.arg_title] = "--name {}".format(args.arg_title)
        name = "final"
    else: name = args.arg_title
    with open("post_{}.slurm".format(name), "a") as f: # First empty it!
        f.write(
"""
#!/bin/bash -l
{}
#SBATCH --mem=2G
##SBATCH --constraint 32

module load singularity
singularity exec t_maze.sif python easy_maze/post_main.py --arg_title {} {}
""".format(partition, args.arg_title, slurm_dict[args.arg_title])[1:])
# %%

