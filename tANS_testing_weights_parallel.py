import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from tqdm import tqdm
import os
from multiprocessing import Pool

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
    
def task(model):

    print("Running model:", model)
    
    d = d_base + model + "/"
    
    # check if the model has already been processed
    settings = ["apack", "256"]

    range_ = find_max_min_in_directory(d, "weight_")

    if "apack" in settings:
        
        # create empty dataframe to store stats, if it doesnt exist
        if not os.path.exists(f"{d}stats_weights_apack_{LUT_EXP}.csv"):
            stats_apack = pd.DataFrame(columns = ["Layer", "Run Time", "Build Time", "Compression Ratio", "Bits per Symbol"])
            stats_apack.to_csv(f"{d}stats_weights_apack_{LUT_EXP}.csv", index = False)
        else:
            stats_apack = pd.read_csv(f"{d}stats_weights_apack_{LUT_EXP}.csv")

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

        for i in range(len(freqs)):
            # open the stats file
            stats_apack = pd.read_csv(f"{d}stats_weights_apack_{LUT_EXP}.csv")
            
            # check if the layer has already been processed
            if i in stats_apack["Layer"].values:
                print(f"\tLayer {i} has already been processed")
                continue
            
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

            # update the stats dataframe
            stats_apack = stats_apack.append({"Layer": i,
                                              "Run Time": run_time_taken,
                                              "Build Time": build_time_taken,
                                              "Compression Ratio": len(msg) * nbits / total_comp_bits,
                                              "Bits per Symbol": total_comp_bits / len(msg)}, ignore_index = True)
            
            # save the stats to a csv file
            stats_apack.to_csv(f"{d}stats_weights_apack_{LUT_EXP}.csv", index = False)

        # save the stats to a csv file
        stats_apack.to_csv(f"{d}stats_weights_apack_{LUT_EXP}.csv", index = False)

    if "256" in settings:
        
        # made dataframe to store stats, if it doesnt exist
        if not os.path.exists(f"{d}stats_weights_256_{LUT_EXP}.csv"):
            stats_256 = pd.DataFrame(columns = ["Layer", "Run Time", "Build Time", "Compression Ratio", "Bits per Symbol"])
            stats_256.to_csv(f"{d}stats_weights_256_{LUT_EXP}.csv", index = False)
        else:
            stats_256 = pd.read_csv(f"{d}stats_weights_256_{LUT_EXP}.csv")

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

        for i in tqdm(range(len(freqs)), desc="\tCompressing Layers"):
            
            # open the stats file
            stats_256 = pd.read_csv(f"{d}stats_weights_256_{LUT_EXP}.csv")
            
            # check if the layer has already been processed
            if i in stats_256["Layer"].values:
                print(f"\tLayer {i} has already been processed")
                continue
            
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
            
            # update the stats dataframe
            stats_256 = stats_256.append({"Layer": i,
                                          "Run Time": run_time_taken,
                                          "Build Time": build_time_taken,
                                          "Compression Ratio": len(msg) * nbits / comp_bits,
                                          "Bits per Symbol": comp_bits / len(msg)}, ignore_index = True)
            
            # save the stats to a csv file
            stats_256.to_csv(f"{d}stats_weights_256_{LUT_EXP}.csv", index = False)
            
        # save 
        stats_256.to_csv(f"{d}stats_weights_256_{LUT_EXP}.csv", index = False)

    # print the average compression ratio for each method
    if "apack" in settings:
        print("\tAverage Compression Ratio Activations APack:", np.mean(stats_apack["Compression Ratio"]))
    if "256" in settings:
        print("\tAverage Compression Ratio Activations 256:", np.mean(stats_256["Compression Ratio"]))
    
if __name__ == "__main__":
    with Pool() as p:
        p.map(task, models)