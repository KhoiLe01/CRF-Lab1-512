# Helper functions to read model and data files
import numpy as np

def read_model(model_path):
    with open(model_path, 'r') as f:
        model = f.readlines()
        model = [line.strip() for line in model]
    W = np.zeros((26, 128), dtype=np.float64)
    T = np.zeros((26, 26), dtype=np.float64)
    for i in range(26):
        W[i, :] = np.array(model[128*i:128*(i+1)], dtype=np.float64)
    for i in range(26):
        T[i, :] = np.array(model[128*26 + 26*i:128*26 + 26*(i+1)], dtype=np.float64)
    return W, T

def read_data(data_path):
    with open(data_path, 'r') as f:
        data = f.readlines()
        data = [line.strip() for line in data]
    X = {}
    Y = {}
    
    for i in range(len(data)):
        flags = data[i].split()
        word_id = int(flags[3])
        current_X = np.array(flags[5:], dtype=np.int8)
        current_Y = ord(flags[1]) - ord('a')
        if word_id not in X:
            X[word_id] = [current_X]
            Y[word_id] = [current_Y]
        else:
            X[word_id].append(current_X)
            Y[word_id].append(current_Y)
    return X, Y