import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import os
import concurrent.futures

# import my tANS function
from Functions.c_int import runner
from Functions.python import Utils, CompTensor

LUT_EXP = 8
LUT_SIZE = 2**LUT_EXP

def find_max_min_in_directory(directory, start):
    max_value = float('-inf')
    min_value = float('inf')
    max_file = ""
    min_file = ""
    
    for filename in os.listdir(directory):
        if filename.endswith(".npy") and filename.startswith(start):
            filepath = os.path.join(filename)
            
            file_max = int(filepath.split("_")[1].split(".")[0])
            file_min = int(filepath.split("_")[1].split(".")[0])
            
            if file_max > max_value:
                max_value = file_max
                max_file = filename
            if file_min < min_value:
                min_value = file_min
                min_file = filename
    
    return min_value, max_value

d_base = 'trace/'

models = os.listdir("trace")

models = [model for model in models if os.path.isdir(f"trace/{model}")]

# check if gpt2 in models and remove if it is
if "gpt2-xl" in models:
    models.remove("gpt2-xl")

def APack(model):
    
    print("Running model:", model)
    
    d = d_base + model + "/"
        
    # check if the model has already been processed
    range_ = find_max_min_in_directory(d, "input_")
    
    # create empty dataframe to store stats, if it doesnt exist
    if not os.path.exists(f"{d}stats_activations_apack_{LUT_EXP}.csv"):
        stats_apack = pd.DataFrame(columns = ["Layer", "Run Time", "Build Time", "Compression Ratio", "Bits per Symbol"])
        stats_apack.to_csv(f"{d}stats_activations_apack_{LUT_EXP}.csv", index = False)
    else:
        stats_apack = pd.read_csv(f"{d}stats_activations_apack_{LUT_EXP}.csv")

    # check if the model has already been processed
    if len(stats_apack) == range_[1] - range_[0]:
        print("Model has already been processed")
        return
            
    # importing the data
    print("\tImporting data")
    data = [np.load(f"{d}input_{i}.npy") for i in range(range_[0],range_[1])]


    # importing the symbol table
    print("\tImporting symbol tables")

    s_tabs = [ pd.read_csv(f"{d}input_{i}_flat.apack", sep = " ", header = None) for i in range(range_[0],range_[1])]

    for s_tab in s_tabs:            
        s_tab.columns = ["vmin","OL","abits","obits","vcnt"]

    # converting each data point to a symbol, offset pair
    comp_tensors = []
    for i, dat in enumerate(data):
        comp_tensors.append([CompTensor.CompTensor(d, s_tabs[i]) for d in dat])

    # Getting freqs, must be a power of 2
    print("\tGetting frequencies APack")
    freqs = []

    for s_tab in s_tabs:
        # Get frequencies from the symbol table
        freq = list(s_tab.vcnt)

        # rescale so the sum of freq is 2**10, this ensures the Coder works effieciently
        # before I was just rescaling to the most accurate power of 2, but the coder would time out
        # building the object
        # Note: the rescale_list_to_power_of_2 function ensures that the sum of the list is a power of 2
        #       and also that no element is zero (bumps up the smallest elements to 1)
        freq = Utils.rescale_list_to_power_of_2(freq, LUT_SIZE)
        
        # append to freqs
        freqs.append(freq)
        
    # offsets
    def int_to_binary_list(value, nbits):
        if value >= 2**nbits or value < 0:
            raise ValueError(f"Value {value} cannot be represented in {nbits} bits.")
        
        binary_list = [int(bit) for bit in bin(value)[2:].zfill(nbits)]
        return binary_list

    # make offset bitstream for one tensor

    offset_stream = []

    for i in range(len(comp_tensors)):
        offset_stream.append([])
        for j in range(len(comp_tensors[i])):
            offset_stream[i].append([])
            for k in range(len(comp_tensors[i][j].points)):
                offset_stream[i][j].extend(int_to_binary_list(comp_tensors[i][j].points[k].off, comp_tensors[i][j].points[k].OL))
                
    import time
    print("\tCompressing Activations APack")
    nbits = 8 # takes 4 bits to represent each symbol
    
    cur_stats = []
    
    for i in range(len(freqs)):
        
        # open the stats file
        stats_apack = pd.read_csv(f"{d}stats_activations_apack_{LUT_EXP}.csv")
        
        # check if the layer has already been processed
        if i in stats_apack["Layer"].values:
            cur_stats.append(dict(stats_apack[stats_apack["Layer"] == i].iloc[0]))
            continue
        
        run_times = []
        build_times = []
        comp_ratios = []
        bp_sym = []

        for j in range(len(comp_tensors[i])):
            # Compressing the symbols
            time_start = time.time()

            c = runner.initCoder(sum(freqs[i]), [i for i in range(len(freqs[i]))], freqs[i])

            time_end = time.time()
            build_time_taken = time_end - time_start

            msg = [p.symbol for p in comp_tensors[i][j].points]

            time_start = time.time()

            comp_bits = runner.encodeDecode(c, msg)
            
            time_end = time.time()
            run_time_taken = time_end - time_start
            
            runner.freeCoder(c)

            # Factoring in the offset bits  
            total_comp_bits = comp_bits + len(offset_stream[i][j])

            run_times.append(run_time_taken)
            build_times.append(build_time_taken)
            comp_ratios.append(len(msg) * nbits / total_comp_bits)
            bp_sym.append(total_comp_bits / len(msg))

        # update the stats dataframe
        cur_stats.append({"Layer": i,
                        "Run Time": np.mean(run_times),
                        "Build Time": np.mean(build_times),
                        "Compression Ratio": np.mean(comp_ratios),
                        "Bits per Symbol": np.mean(bp_sym)})
        
        stats_apack = pd.DataFrame(cur_stats)
        
        # save the stats to a csv file
        stats_apack.to_csv(f"{d}stats_activations_apack_{LUT_EXP}.csv", index = False)

    # save the stats to a csv file
    stats_apack.to_csv(f"{d}stats_activations_apack_{LUT_EXP}.csv", index = False)
        
    
   
def two56(model):
    
    print("Running model:", model)
    
    d = d_base + model + "/"
        
    range_ = find_max_min_in_directory(d, "input_")
        
    # create empty dataframe to store stats, if it doesnt exist
    if not os.path.exists(f"{d}stats_activations_256_{LUT_EXP}.csv"):
        stats_256 = pd.DataFrame(columns = ["Layer", "Run Time", "Build Time", "Compression Ratio", "Bits per Symbol"])
        stats_256.to_csv(f"{d}stats_activations_256_{LUT_EXP}.csv", index = False)
    else:
        stats_256 = pd.read_csv(f"{d}stats_activations_256_{LUT_EXP}.csv")

    # check if the model has already been processed
    if len(stats_256) == range_[1] - range_[0]:
        print("Model has already been processed")
        return
                
    # importing the data
    print("\tImporting data")
    data = [np.load(f"{d}input_{i}.npy") for i in range(range_[0],range_[1])]
    
    # Calculate frequency of each uint8 value
    def calculate_frequency(array):
        # Ensure the input array is of type uint8
        if array.dtype != np.uint8:
            raise ValueError("Input array must be of type uint8")
        
        # Initialize an array of zeros with a length of 256 to store frequencies
        frequency = np.zeros(256, dtype=int)
        
        # Iterate through the array and count the occurrences of each value
        for value in array:
            frequency[value] += 1
            
        return frequency

    freqs = [calculate_frequency(d) for d in data]

    # rescale the frequencies to a power of 2
    freqs = [Utils.rescale_list_to_power_of_2(freq, LUT_SIZE) for freq in freqs]

    import time
    print("\tCompressing Activations 256")
    nbits = 8 # takes 4 bits to represent each symbol
    
    cur_stats = []

    for i in range(len(freqs)):
        
        # open the stats file
        stats_256 = pd.read_csv(f"{d}stats_activations_256_{LUT_EXP}.csv")
        
        if i in stats_256["Layer"].values:
            cur_stats.append(dict(stats_256[stats_256["Layer"] == i].iloc[0]))
            continue
        
        run_times = []
        build_times = []
        comp_ratios = []
        bp_sym = []

        for j in range(len(data[i])):
            time_start = time.time()
            
            c = runner.initCoder(sum(freqs[i]), [k for k in range(len(freqs[i]))], freqs[i])
            
            time_end = time.time()
            build_time_taken = time_end - time_start

            msg = list(data[i][j].flatten())

            time_start = time.time()
            
            comp_bits = runner.encodeDecode(c, msg)
            
            time_end = time.time()
            run_time_taken = time_end - time_start

            runner.freeCoder(c)
            
            run_times.append(run_time_taken)
            build_times.append(build_time_taken)
            comp_ratios.append(len(msg) * nbits / comp_bits)
            bp_sym.append(comp_bits / len(msg))
            
        # update the stats dataframe
        cur_stats.append({"Layer": i,
                        "Run Time": np.mean(run_times),
                        "Build Time": np.mean(build_times),
                        "Compression Ratio": np.mean(comp_ratios),
                        "Bits per Symbol": np.mean(bp_sym)})
        
        stats_256 = pd.DataFrame(cur_stats)
        
        # save the stats to a csv file
        stats_256.to_csv(f"{d}stats_activations_256_{LUT_EXP}.csv", index = False)
        
    # save
    stats_256.to_csv(f"{d}stats_activations_256_{LUT_EXP}.csv", index = False)
   
def task(args):
    function, model = args
    function(model)
    
if __name__ == "__main__":
    funcs = [APack, two56]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(task, zip(funcs, models)))