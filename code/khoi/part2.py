import numpy as np
import os

from read_data import read_data, read_model
from scipy.optimize import check_grad, fmin_tnc
from part1c import viterbi_decoder

def my_callback(xk):
    print(f"Current parameters: {xk}")

# Compute the forward messages
def computeZ(node_potentials, T):
    m, num_labels = node_potentials.shape    
    f = np.ones((m, num_labels), dtype=np.float64)
    
    f[0, :] = 0
    
    for s in range(1, m):
        exponent_terms = node_potentials[s-1, :][:, np.newaxis] + T + f[s-1, :][:, np.newaxis]
        
        M = np.max(exponent_terms, axis=0)
        
        f[s, :] = M + np.log(np.sum(np.exp(exponent_terms - M), axis=0))
    
    final_terms = node_potentials[m-1, :] + f[m-1, :]
    M_final = np.max(final_terms)
    Z = M_final + np.log(np.sum(np.exp(final_terms - M_final)))
        
    return f, Z

# Backward messages
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

# Compute Y given X by simply summing up the node and edge potentials
def computeYgivenX(X, W, Y, T):
    result = np.float64(0)
    for i in range(Y.shape[0]):
        result += np.dot(X[i], W[Y[i]].T)
    for i in range(Y.shape[0] - 1):
        result += T[Y[i], Y[i+1]]
    return result

# Compute node and edge marginals for part 4
def compute_marginals(X, W, T):
    node_potentials = np.dot(X, W.T)
    
    f, Z = computeZ(node_potentials, T)
    f_reverse, Z_reverse = computeZreverse(node_potentials, T)
    
    log_node_marginals = node_potentials + f + f_reverse - Z
    node_marginals = np.exp(log_node_marginals)
    
    edges_marginals = np.zeros((X.shape[0]-1, T.shape[0], T.shape[1]))
    for s in range(X.shape[0]-1):
        log_edge_marginals = f[s, :][:, np.newaxis] + f_reverse[s+1, :][np.newaxis, :] + node_potentials[s, :][:, np.newaxis] + node_potentials[s+1, :][np.newaxis, :] + T - Z
        edges_marginals[s] = np.exp(log_edge_marginals)
        
    return node_marginals, edges_marginals

# Compute gradient of W and T
def computegradient(X, Y, W, T):
    node_potentials = np.dot(X, W.T)
    
    f, Z = computeZ(node_potentials, T)
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
        grad_W[Y[i]] += X[i] # Observed
    
    grad_W -= np.dot(node_marginals.T, X) # Expected
    
    for s in range(X.shape[0]-1):
        grad_T[Y[s], Y[s+1]] += 1 # Observed
        grad_T -= edges_marginals[s] # Expected
    
    return grad_W, grad_T

# log(Y|X) = Y|X - logZ
def getlogYgivenX(X, Y, W, T):
    node_potentials = np.dot(X, W.T)
    
    f, Z = computeZ(node_potentials, T)
    
    logP_Y_given_X = computeYgivenX(X, W, Y, T) - Z
    
    return logP_Y_given_X


# Wrapper for computing gradient and objective
def crf_obj_grad(params, X_list, Y_list, C, num_labels, feature_dim):
    W = params[:num_labels * feature_dim].reshape(num_labels, feature_dim)
    T = params[num_labels * feature_dim:].reshape(num_labels, num_labels)
    
    total_log_p = 0
    grad_W = np.zeros_like(W, dtype=np.float64)
    grad_T = np.zeros_like(T, dtype=np.float64)
    
    final_grad_W = np.zeros_like(W, dtype=np.float64)
    final_grad_T = np.zeros_like(T, dtype=np.float64)
    
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

# For testing in scipy
def get_obj(params, *args):
    return crf_obj_grad(params, *args)[0]

# For testing in scipy
def get_grad(params, *args):
    return crf_obj_grad(params, *args)[1]

# Check that node and edge marginals sum to 1
def check_marginals_sum_to_1():
    # Load real data
    W, T = read_model('data/model.txt')
    X_dict, _ = read_data('data/train.txt')
    
    total_samples = len(X_dict)
    passed_samples = 0
    failed_samples = 0

    print(f"Total samples to check: {total_samples}")

    for key, X_val in X_dict.items():
        X = np.array(X_val, dtype=np.float64)
        m = X.shape[0]
        
        # Compute potentials
        node_potentials = np.dot(X, W.T)
        
        f, Z = computeZ(node_potentials, T)
        f_reverse, Z_reverse = computeZreverse(node_potentials, T)
        
        # Z from forward and backward should match
        if not np.isclose(Z, Z_reverse):
            print(f"FAIL [ID {key}]: Z mismatch! Forward: {Z}, Backward: {Z_reverse}")
            failed_samples += 1
            continue

        # node marginals
        log_node_marginals = node_potentials + f + f_reverse - Z
        node_marginals = np.exp(log_node_marginals)
        node_sums = np.sum(node_marginals, axis=1)
        
        if not np.allclose(node_sums, 1.0):
             print(f"FAIL [ID {key}]: Node marginals do NOT sum to 1")
             failed_samples += 1
             continue

        # Edge marginals
        all_edges_pass = True
        for s in range(m - 1):
            log_edge_m = f[s][:, None] + f_reverse[s+1][None, :] + \
                         node_potentials[s][:, None] + node_potentials[s+1][None, :] + \
                         T - Z
            edge_m = np.exp(log_edge_m)
            total = np.sum(edge_m)
            if not np.isclose(total, 1.0):
                all_edges_pass = False
                break
                
        if not all_edges_pass:
            print(f"FAIL [ID {key}]: Edge marginals do NOT sum to 1")
            failed_samples += 1
            continue
            
        passed_samples += 1

    print(f"\nFinal Result: {passed_samples}/{total_samples} passed.")

# Test the gradient using scipy
def test():
    
    m, d, k = 4, 3, 3  # 4 letters, 3 features, 3 labels
    np.random.seed(42)

    # Generate random data
    X_test = [np.random.randn(m, d)]
    Y_test = [np.random.randint(1, k + 1, size=m)]
    C = 10

    W_init = np.random.randn(k, d)
    T_init = np.random.randn(k, k)
    params_init = np.concatenate([W_init.flatten(), T_init.flatten()])

    error = check_grad(get_obj, get_grad, params_init, X_test, Y_test, C, k, d)

    print(f"Gradient Error: {error}")

# Write solution
def save_solution(W, T, filename='result/solution.txt'):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w') as f:
        for row in W:
            f.write('\n'.join(map(str, row)) + '\n')
        
        for row in T:
            f.write('\n'.join(map(str, row)) + '\n')
    
    f.close()

def part2a():
    W, T = read_model('data/model.txt')
    X, Y = read_data('data/train.txt')
    
    result_W = np.zeros_like(W)
    result_T = np.zeros_like(T)
    log_p_avg = 0
    
    for word_id in X:
        X_t = np.array(X[word_id], dtype=np.float64)
        Y_t = np.array(Y[word_id], dtype=np.int8)
        grad_W, grad_T = computegradient(X_t, Y_t, W, T)
        log_p = getlogYgivenX(X_t, Y_t, W, T)
        
        result_W += grad_W
        result_T += grad_T
        log_p_avg += log_p

    result_W /= len(X)
    result_T /= len(X)
    log_p_avg /= len(X)
    
    print(f"Average Log-Likelihood: {log_p_avg}")
    save_solution(result_W, result_T, filename='result/gradient.txt')

def prediction(W, T):
    X_dict, _ = read_data('data/test.txt')
    
    predictions = []
    
    sorted_keys = sorted(X_dict.keys())

    for word_id in sorted_keys:
        X_t = np.array(X_dict[word_id], dtype=np.float64)
        m = X_t.shape[0]
        
        y_best, _ = viterbi_decoder(X_t, W, T)
            
        predictions.extend(y_best)
        
    # Save predictions
    output_file = 'result/prediction.txt'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        for label in predictions:
            f.write(f'{label}\n')
            
    print(f"Predictions saved to {output_file}")

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
    
    optimal_params, nfeval, rc = fmin_tnc(func=crf_obj_grad, x0=params_init, args=(X_list, Y_list, C, num_labels, num_features), bounds=None)
    print(f"Optimization finished. Evaluations: {nfeval}, Return Code: {rc}")

    # Reshape the optimal parameters back 
    W_opt = optimal_params[:num_labels * num_features].reshape(num_labels, num_features)
    T_opt = optimal_params[num_labels * num_features:].reshape(num_labels, num_labels)
    
    save_solution(W_opt, T_opt, filename='result/solution.txt')
    prediction(W_opt, T_opt)
    
    return W_opt, T_opt

if __name__ == "__main__":
    # test()
    # part2a()
    # check_marginals_sum_to_1()
    # W_opt, T_opt = part2b()
    
    # Separate test
    # W, T = read_model('result/solution.txt')
    # prediction(W, T)
