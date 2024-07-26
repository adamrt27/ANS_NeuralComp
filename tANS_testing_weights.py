import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from tqdm import tqdm
import os

LUT_EXP = 8
LUT_SIZE = 2**LUT_EXP

# import my tANS function
from Functions import Coder, Utils, CompTensor

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

for model in models:

    print("Running model:", model)
    
    d = d_base + model + "/"

    range_ = find_max_min_in_directory(d, "weight_")

    # importing the symbol table
    print("\tImporting symbol tables")

    s_tabs = [ pd.read_csv(f"{d}weight_{i}_flat.apack", sep = " ", header = None) for i in range(range_[0],range_[1])]

    for s_tab in s_tabs:            
        s_tab.columns = ["vmin","OL","abits","obits","vcnt"]

    s_tabs[0]

    # importing the data
    print("\tImporting data")
    data = [np.load(f"{d}weight_{i}.npy") for i in range(range_[0],range_[1])]

    # converting each data point to a symbol, offset pair

    comp_tensors = []
    for i, dat in enumerate(tqdm(data, desc="\tConverting Data to CompTensors")):
        comp_tensors.append([CompTensor.CompTensor(dat.flatten(), s_tabs[i])])


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

    for i in range(len(comp_tensors)): # for each tensor
        offset_stream.append([])
        for j in range(len(comp_tensors[i])): # for each point in the tensor
            offset_stream[i].append([])
            for k in range(len(comp_tensors[i][j].points)): 
                offset_stream[i][j].extend(int_to_binary_list(comp_tensors[i][j].points[k].off, comp_tensors[i][j].points[k].OL))
                
    import time
    print("\tCompressing Weights APack")

    nbits = 8  # Takes 8 bits to represent each symbol

    all_run_times = []
    all_build_times = []
    all_comp_ratios = []
    all_bps = []

    for i in tqdm(range(len(freqs)), desc="\tCompressing Layers"):
        run_times = []
        build_times = []
        comp_ratios = []
        bp_sym = []
        j = 0

        # Compressing the symbols
        time_start = time.time()
        
        c = Coder.Coder(sum(freqs[i]), [i for i in range(len(freqs[i]))], freqs[i], fast=False)
        
        time_end = time.time()
        build_time_taken = time_end - time_start

        msg = [p.symbol for p in comp_tensors[i][j].points]

        time_start = time.time()
        out, comp_bits = c.encode_decode(msg)
        time_end = time.time()
        run_time_taken = time_end - time_start
        
        # Factoring in the offset bits  
        total_comp_bits = comp_bits + len(offset_stream[i][j])

        if out != msg:
            tqdm.write("Error in encoding and decoding")
            break
        
        all_run_times.append(run_time_taken)
        all_build_times.append(build_time_taken)
        all_comp_ratios.append(len(msg) * nbits / total_comp_bits)
        all_bps.append(total_comp_bits / len(msg))
            
        # Print average stats
        avg_run_time = np.mean(run_times)
        avg_build_time = np.mean(build_times)
        avg_comp_ratio = np.mean(comp_ratios)
        avg_bp_sym = np.mean(bp_sym)
        
    # display stats in a dataframe

    freqs = freqs

    stats_apack = pd.DataFrame({"Layer": [i for i in range(len(freqs))],
                        "Run Time": [np.mean(all_run_times[i]) for i in range(len(freqs))],
                        "Build Time": [np.mean(all_build_times[i]) for i in range(len(freqs))],
                        "Compression Ratio": [np.mean(all_comp_ratios[i]) for i in range(len(freqs))],
                        "Bits per Symbol": [np.mean(all_bps[i]) for i in range(len(freqs))]})

    # save the stats to a csv file
    stats_apack.to_csv(f"{d}stats_weights_apack_{LUT_EXP}.csv", index = False)

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
    print("\tCompressing Weights 256")

    nbits = 8  # Takes 8 bits to represent each symbol

    all_run_times = []
    all_build_times = []
    all_comp_ratios = []
    all_bps = []

    for i in tqdm(range(len(freqs)), desc="\tCompressing Layers"):
        run_times = []
        build_times = []
        comp_ratios = []
        bp_sym = []
        
        time_start = time.time()
        
        c = Coder.Coder(sum(freqs[i]), [i for i in range(len(freqs[i]))], freqs[i], fast=False)
        
        time_end = time.time()
        build_time_taken = time_end - time_start

        msg = list(data[i].flatten())

        time_start = time.time()
        out, comp_bits = c.encode_decode(msg)
        time_end = time.time()
        run_time_taken = time_end - time_start

        if out != msg:
            tqdm.write("Error in encoding and decoding")
            break
        
        run_times.append(run_time_taken)
        build_times.append(build_time_taken)
        comp_ratios.append(len(msg) * nbits / comp_bits)
        bp_sym.append(comp_bits / len(msg))
        
        # Print average stats
        avg_run_time = np.mean(run_times)
        avg_build_time = np.mean(build_times)
        avg_comp_ratio = np.mean(comp_ratios)
        avg_bp_sym = np.mean(bp_sym)
        
        
        # Add stats to all lists
        all_run_times.append(run_times)
        all_build_times.append(build_times)
        all_comp_ratios.append(comp_ratios)
        all_bps.append(bp_sym)
        
    freqs = freqs

    stats_256 = pd.DataFrame({"Layer": [i for i in range(len(freqs))],
                        "Run Time": [np.mean(all_run_times[i]) for i in range(len(freqs))],
                        "Build Time": [np.mean(all_build_times[i]) for i in range(len(freqs))],
                        "Compression Ratio": [np.mean(all_comp_ratios[i]) for i in range(len(freqs))],
                        "Bits per Symbol": [np.mean(all_bps[i]) for i in range(len(freqs))]})

    # save 
    stats_256.to_csv(f"{d}stats_weights_256_{LUT_EXP}.csv", index = False)

    # print the average compression ratio for each method
    print("\tAverage Compression Ratio Weights APack:", np.mean(stats_apack["Compression Ratio"]))
    print("\tAverage Compression Ratio Weights 256:", np.mean(stats_256["Compression Ratio"]))