#%%

import os
import pickle

from utils import args, folder
from plotting import plots

if(args.arg_title[:3] != "___"):
    
    try:
        print("Trying to load already-processed values...\n")
        with open("saved/" + args.arg_title + "/" + "plot_dict.pickle", "rb") as handle: 
            plot_dict = pickle.load(handle)
        with open("saved/" + args.arg_title + "/" + "min_max_dict.pickle", "rb") as handle: 
            min_max_dict = pickle.load(handle)
        print("Already processed!\n")
    except:
        print("No already-processed values. Processing!\n")
        plot_dict = {} ; min_max_dict = {}
        files = os.listdir(folder) ; files.sort()
        
        for file in files:
            if(file.split("_")[0] == "plot"): d = plot_dict
            if(file.split("_")[0] == "min"):  d = min_max_dict
            with open(folder + "/" + file, "rb") as handle: 
                saved_d = pickle.load(handle) ; os.remove(folder + "/" + file)
            for key in saved_d.keys(): 
                if(not key in d): d[key] = []
                d[key].append(saved_d[key])
        d["title"] = args.name
            
        for key in min_max_dict.keys():
            if(not key in ["args", "title", "spot_names"]):
                minimum = None ; maximum = None
                for min_max in min_max_dict[key]:
                    if(minimum == None):        minimum = min_max[0]
                    elif(minimum > min_max[0]): minimum = min_max[0]
                    if(maximum == None):        maximum = min_max[1]
                    elif(maximum < min_max[1]): maximum = min_max[1]
                min_max_dict[key] = (minimum, maximum)
        
        with open(folder + "/plot_dict.pickle", "wb") as handle:
            pickle.dump(plot_dict, handle)
        with open(folder + "/min_max_dict.pickle", "wb") as handle:
            pickle.dump(min_max_dict, handle)
                
        print("Done with {}!".format(args.name))
    
else:
    
    plot_dicts = [] ; min_max_dicts = []
    
    order = args.name[3:-3]
    order = order.split("+")
    order = [o for o in order if o != "break"]
    
    print(order)

    for name in order:
        with open("saved/" + name + "/" + "plot_dict.pickle", "rb") as handle: 
            plot_dicts.append(pickle.load(handle))
        with open("saved/" + name + "/" + "min_max_dict.pickle", "rb") as handle: 
            min_max_dicts.append(pickle.load(handle))
            
    min_max_dict = {}
    for key in plot_dicts[0].keys():
        if(not key in ["title", "spot_names"]):
            minimum = None ; maximum = None
            for mm_dict in min_max_dicts:
                if(mm_dict[key] != (None, None)):
                    if(minimum == None):             minimum = mm_dict[key][0]
                    elif(minimum > mm_dict[key][0]): minimum = mm_dict[key][0]
                    if(maximum == None):             maximum = mm_dict[key][1]
                    elif(maximum < mm_dict[key][1]): maximum = mm_dict[key][1]
            min_max_dict[key] = (minimum, maximum)
            
    plots(plot_dicts, min_max_dict)
    print("Finished!")