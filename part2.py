import numpy as np
import os

from read_data import read_data, read_model
from scipy.optimize import check_grad, fmin_tnc

def computeZ(node_potentials, T):
    m, num_labels = node_potentials.shape
    # num_labels = 26
    
    f = np.ones((m, num_labels))
    
    f[0, :] = 0
    
    for s in range(1, m):
        exponent_terms = node_potentials[s-1, :][:, np.newaxis] + T + f[s-1, :][:, np.newaxis]
        
        M = np.max(exponent_terms, axis=0)
        
        f[s, :] = M + np.log(np.sum(np.exp(exponent_terms - M), axis=0))
    
    final_terms = node_potentials[m-1, :] + f[m-1, :]
    M_final = np.max(final_terms)
    Z = M_final + np.log(np.sum(np.exp(final_terms - M_final)))
        
    return f, Z

def computeZreverse(node_potentials, T):
    m = node_potentials.shape[0]
    num_labels = node_potentials.shape[1]
    
    f = np.ones((m, num_labels))
    
    f[m-1, :] = 0
    
    for s in range(m-2, -1, -1):
        exponent_terms = node_potentials[s+1, :] + T + f[s+1, :]
        
        M = np.max(exponent_terms, axis=1)
        
        f[s, :] = M + np.log(np.sum(np.exp(exponent_terms - M[:, np.newaxis]), axis=1))
    
    final_terms = node_potentials[0, :] + f[0, :]
    M_final = np.max(final_terms)
    Z = M_final + np.log(np.sum(np.exp(final_terms - M_final)))
        
    return f, Z

def computeYgivenX(X, W, Y, T):
    """
    X: m x 128
    W: 26 x 128
    Y: m for the labels (1 to 26)
    T: 26 x 26
     - T[i, j] is the transition potential from label i to label j
    """
    result = 0
    for i in range(Y.shape[0]):
        result += np.dot(X[i], W[Y[i]-1].T)
    for i in range(Y.shape[0] - 1):
        result += T[Y[i]-1, Y[i+1]-1]
    return result

def computegradient(X, Y, W, T):
    node_potentials = np.dot(X, W.T)
    
    f, Z = computeZ(node_potentials, T) # Z is the log partition function
    f_reverse, Z_reverse = computeZreverse(node_potentials, T)
    
    log_node_marginals = node_potentials + f + f_reverse - Z
    node_marginals = np.exp(log_node_marginals)
    
    edges_marginals = np.zeros((X.shape[0]-1, T.shape[0], T.shape[1]))
    for s in range(X.shape[0]-1):
        log_edge_marginals = f[s, :][:, np.newaxis] + f_reverse[s+1, :][np.newaxis, :] + node_potentials[s, :][:, np.newaxis] + node_potentials[s+1, :][np.newaxis, :] + T - Z
        edges_marginals[s] = np.exp(log_edge_marginals)
    
    grad_W = np.zeros_like(W)
    grad_T = np.zeros_like(T)
    
    for i in range(Y.shape[0]):
        grad_W[Y[i]-1] += X[i] # Observed
    
    grad_W -= np.dot(node_marginals.T, X) # Expected
    
    for s in range(X.shape[0]-1):
        grad_T[Y[s]-1, Y[s+1]-1] += 1 # Observed
        grad_T -= edges_marginals[s] # Expected
    
    return grad_W, grad_T

def getlogYgivenX(X, Y, W, T):
    node_potentials = np.dot(X, W.T)
    
    f, Z = computeZ(node_potentials, T)
    
    logP_Y_given_X = computeYgivenX(X, W, Y, T) - Z
    
    return logP_Y_given_X


# Test using scipy
def crf_obj_grad(params, X_list, Y_list, C, num_labels, feature_dim):
    """
    Wrapper to compute objective and gradient for scipy.optimize.
    """
    W = params[:num_labels * feature_dim].reshape(num_labels, feature_dim)
    T = params[num_labels * feature_dim:].reshape(num_labels, num_labels)
    
    total_log_p = 0
    grad_W = np.zeros_like(W)
    grad_T = np.zeros_like(T)
    n = len(X_list)

    # y given x
    for X, Y in zip(X_list, Y_list):
        log_p = getlogYgivenX(X, Y, W, T)
        gW, gT = computegradient(X, Y, W, T)
        total_log_p += log_p
        grad_W += gW
        grad_T += gT

    reg_W = 0.5 * np.sum(W**2)
    reg_T = 0.5 * np.sum(T**2)
    obj = -(C/n) * total_log_p + reg_W + reg_T
    
    final_grad_W = -(C/n) * grad_W + W
    final_grad_T = -(C/n) * grad_T + T
    
    return obj, np.concatenate([final_grad_W.flatten(), final_grad_T.flatten()])

def get_obj(params, *args):
    return crf_obj_grad(params, *args)[0]

def get_grad(params, *args):
    return crf_obj_grad(params, *args)[1]


# Addition gradient of lost function with respect to W and T for regularization


def test():
    
    m, d, k = 4, 3, 3  # 4 letters, 3 features, 3 labels
    np.random.seed(42)

    # Generate synthetic data
    X_test = [np.random.randn(m, d)]
    Y_test = [np.random.randint(1, k + 1, size=m)] # Labels 1 to k
    C = 10.0

    # Initial random parameters
    W_init = np.random.randn(k, d)
    T_init = np.random.randn(k, k)
    params_init = np.concatenate([W_init.flatten(), T_init.flatten()])

    # --- Execute Gradient Check ---
    # check_grad returns the norm of the difference between your grad and numerical grad
    error = check_grad(get_obj, get_grad, params_init, X_test, Y_test, C, k, d)

    print(f"Gradient Error: {error}")
    if error < 1e-4:
        print("SUCCESS: Analytical gradient matches numerical gradient.")
    else:
        print("FAILURE: Check your forward-backward or gradient accumulation logic.")

def save_solution(W, T, filename='result/solution.txt'):
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w') as f:
        # Write W (Node Weights)
        for row in W:
            f.write(' '.join(map(str, row)) + '\n')
        
        # Write T (Transition Matrix)
        for row in T:
            f.write(' '.join(map(str, row)) + '\n')
    print(f"Solution saved to {filename}")        

def part2a():
    W, T = read_model('data/model.txt')
    X, Y = read_data('data/train.txt')
    
    # for weight_ID in range(W.shape[0]):
    #     W_c = W[weight_ID:weight_ID+1]
    result_W = np.zeros_like(W)
    result_T = np.zeros_like(T)
    log_p_avg = 0
    
    for word_id in X:
        X_t = np.array(X[word_id], dtype=np.float64)
        Y_t = np.array(Y[word_id], dtype=np.int8)
        grad_W, grad_T = computegradient(X_t, Y_t, W, T)
        log_p = getlogYgivenX(X_t, Y_t, W, T)
        # print(f"Gradient for W on word {word_id}: {grad_W}")
        # print(f"Gradient for T on word {word_id}: {grad_T}")
        
        result_W += grad_W
        result_T += grad_T
        log_p_avg += log_p

    result_W /= len(X)
    result_T /= len(X)
    log_p_avg /= len(X)
    
    print(result_W)
    print(result_T)
    print(log_p_avg)

def part2b():
    num_labels = 26
    num_features = 128
    C = 1000
    W_init = np.zeros((num_labels, num_features))
    T_init = np.zeros((num_labels, num_labels))
    params_init = np.concatenate([W_init.flatten(), T_init.flatten()])
    
    X, Y = read_data('data/train.txt')
    X_list = [np.array(word, dtype=np.float64) for word in X.values()]
    Y_list = [np.array(labels, dtype=np.int32) for labels in Y.values()]
    # print(X_list.shape, Y_list.shape)
    
    optimal_params, nfeval, rc = fmin_tnc(func=crf_obj_grad, x0=params_init, args=(X_list, Y_list, C, num_labels, num_features),bounds=None)

    # Reshape the optimal parameters back 
    W_opt = optimal_params[:num_labels * num_features].reshape(num_labels, num_features)
    T_opt = optimal_params[num_labels * num_features:].reshape(num_labels, num_labels)
    
    save_solution(W_opt, T_opt)
    
    return W_opt, T_opt

if __name__ == "__main__":
    # part2a()
    W_opt, T_opt = part2b()
    # test()