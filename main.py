#%%

import pickle

from utils import args, folder
from train import Trainer

print("name + id: {}_{}".format(args.name, args.id))

import datetime 
start_time = datetime.datetime.now()

def reset_start_time():
    global start_time
    start_time = datetime.datetime.now()
    
def duration():
    global start_time
    change_time = datetime.datetime.now() - start_time
    change_time = change_time - datetime.timedelta(microseconds=change_time.microseconds)
    return(change_time)

try:
    print("\nTrying to load already-trained values...\n")
    with open("saved/" + args.arg_title + "/" + "plot_dict.pickle", "rb") as handle: 
        plot_dict = pickle.load(handle)
    with open("saved/" + args.arg_title + "/" + "min_max_dict.pickle", "rb") as handle: 
        min_max_dict = pickle.load(handle)
    print("Already trained!\n")
except: 
    print("No already-trained values. Training!\n")
    trainer = Trainer(args, "{}_{}".format(args.name, args.id))
    plot_dict, min_max_dict = trainer.train()

    with open(folder + "/plot_dict_{}.pickle".format(   str(args.id).zfill(3)), "wb") as handle:
        pickle.dump(plot_dict, handle)
    with open(folder + "/min_max_dict_{}.pickle".format(str(args.id).zfill(3)), "wb") as handle:
        pickle.dump(min_max_dict, handle)
    
print("Duration: {}".format(duration()))

# %%
