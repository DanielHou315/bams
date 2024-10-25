import pickle
import numpy as np
import sys
import os

def recurse_print(obj, spacing=0):
    if isinstance(obj, dict):
        print(" "*spacing,"{")
        for key, val in obj.items():
            print(" "*(spacing+2), key,":")
            recurse_print(val, spacing+4)
        print(" "*spacing,"}")
    elif isinstance(obj, list):
        print(" "*spacing,"[")
        for item in obj:
            recurse_print(item, spacing+2)
        print(" "*spacing,']')
    elif isinstance(obj, np.ndarray):
        print(" "*spacing, "Array of size", obj.shape)
    else:
        print(" "*spacing, obj)

def summarize(d, modifier="requested"):
    print(f"-------------------\nThe {modifier} dataset contains:")
    recurse_print(d)

def inspect(file_path):
    d = np.load(file_path, allow_pickle=True).item()
    summarize(d, "original")

def reduce_robot(d, num_sequences):
    reduced_dict = {}
    total_num_sequences = d["base_pos"].shape[0]
    for key, val in d.items():
        if isinstance(val, np.ndarray) and val.shape[0] == total_num_sequences:
            reduced_dict[key] = val[:num_sequences, ...]
        else:
            reduced_dict[key] = val
    return reduced_dict

def reduce_mabe(d, num_sequences):
    # Reduce
    reduced_dict = {}
    total_num_sequences = len(d["sequences"])
    for key, val in d.items():
        if key == "sequences":
            sequences = {}
            num_add = 0
            for k, v in val.items():
                sequences[k] = v
                num_add += 1
                if num_add >= num_sequences:
                    break
            reduced_dict["sequences"] = sequences
        else:
            reduced_dict[key] = val
    return reduced_dict
    

def reduce(file_path, num_sequences, type="mabe"):
    d = np.load(file_path, allow_pickle=True).item()
    # summarize(d, "original")

    if type == "mabe":
        reduced_dict = reduce_mabe(d, num_sequences)
    elif type == "robot":
        reduced_dict = reduce_robot(d, num_sequences)
    else:
        raise Exception("Unknown type to reduce size of!")
    
    # Generate new filename with sequence count
    base, ext = os.path.splitext(os.path.basename(file_path))
    reduced_name = f"{base}_reduced_{num_sequences}.npy"
    with open(reduced_name, 'wb') as f:
        pickle.dump(np.array(reduced_dict), f)
        print("-------------------\nReduced file saved as", 
              reduced_name, "with {:10.2f} MB".format(os.stat(reduced_name).st_size/(1024*1024)))
    # Summarize reduced set
    summarize(reduced_dict, "reduced")


if __name__ == "__main__":
    DEFAULT_SEQ_NUM = 3
    if len(sys.argv) < 3:
        print("Usage: python make_small.py <inspect/reduce-robot/reduce-mabe> <filename> [num_sequences (for reduce only)]")
    else:
        action = sys.argv[1].lower()
        file_path = sys.argv[2]
        
        if action == "inspect":
            inspect(file_path)
        elif action.startswith('reduce-'):
            reducible_targets = ["robot", "mabe"]
            target = action[7:]
            if target not in reducible_targets:
                print(f"Invalid Reduce Dataset Type {target}! Allowed types are {reducible_targets}")
                exit(1)
            if len(sys.argv) < 4:
                print(f"Number of sequences to retain not specified, default to {DEFAULT_SEQ_NUM}")
            num_sequences = int(sys.argv[3]) if len(sys.argv) >= 4 else DEFAULT_SEQ_NUM
            reduce(file_path, num_sequences, target)
        else:
            print("Unknown action. Use 'inspect' or 'reduce-robot' or 'reduce-mabe'.")