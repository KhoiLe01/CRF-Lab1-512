from numpy.random import rand
from part2 import getlogYgivenX, computegradient, save_solution, crf_obj_grad
from part1c import viterbi_decoder
import numpy as np
from read_data import read_data
from scipy.optimize import fmin_tnc

# Global history for plotting
lbfgs_eval_x = []
lbfgs_eval_obj = []
lbfgs_iter_x = []
lbfgs_iter_err = []
eval_counter = 0

# Cached data for callback
CB_X_list_train = None
CB_Y_list_train = None
CB_X_list_test = None
CB_Y_list_test = None

# Get objective value
def get_obj(params, X_list, Y_list, C, num_labels, feature_dim):
    W = params[:num_labels * feature_dim].reshape(num_labels, feature_dim)
    T = params[num_labels * feature_dim:].reshape(num_labels, num_labels)
    total_log_p = 0
    for X, Y in zip(X_list, Y_list):
        log_p = getlogYgivenX(X, Y, W, T)
        total_log_p += log_p
    obj = -(C/len(X_list)) * total_log_p + 0.5 * np.sum(W**2) + 0.5 * np.sum(T**2)
    return obj

# Get wordwise error
def get_wordwise_error(params, X_list, Y_list, C, num_labels, feature_dim):
    W = params[:num_labels * feature_dim].reshape(num_labels, feature_dim)
    T = params[num_labels * feature_dim:].reshape(num_labels, num_labels)
    total = 0
    correct = 0
    
    for X, Y in zip(X_list, Y_list):
        predicted_Y, _ = viterbi_decoder(X, W, T)
        correct += 1 if np.array_equal(predicted_Y, Y + 1) else 0
        total += 1
    return 1- (correct/total)

# Wrapper to count function evaluations and record objective
# Call the crf_obj_grad from part 2
def wrapped_crf_obj_grad(params, X_list, Y_list, C, num_labels, feature_dim):
    global eval_counter, lbfgs_eval_x, lbfgs_eval_obj
    eval_counter += 1
    
    obj, grad = crf_obj_grad(params, X_list, Y_list, C, num_labels, feature_dim)
    
    # Store objective at this evaluation
    lbfgs_eval_x.append(eval_counter)
    lbfgs_eval_obj.append(obj)
    
    return obj, grad

# Callback function for logging
def callback_lbfgs(params):
    global eval_counter, lbfgs_iter_x, lbfgs_iter_err
    
    wordwise_accuracy = get_wordwise_error(params, CB_X_list_test, CB_Y_list_test, C=1000, num_labels=26, feature_dim=128)
    
    lbfgs_iter_x.append(eval_counter)
    lbfgs_iter_err.append(wordwise_accuracy)
    
    current_obj = lbfgs_eval_obj[-1] if lbfgs_eval_obj else 0.0
    
    print(f"Iteration {len(lbfgs_iter_x)} (Eval {eval_counter}): Objective = {current_obj:.4f}, Test Error = {wordwise_accuracy:.4f}")
    
# Stochastic gradient for MCMC and MCMC RB. Objective is not needed here
# The objective is only calculated if needed for graph
def crf_stochastic_obj_grad(params, X_list, Y_list, C, num_labels, feature_dim, batch_indices):
    # global eval_counter
    # eval_counter += 1 # Increment for every function evaluation
    
    W = params[:num_labels * feature_dim].reshape(num_labels, feature_dim)
    T = params[num_labels * feature_dim:].reshape(num_labels, num_labels)
    
    # batch_log_p = 0
    grad_W = np.zeros_like(W)
    grad_T = np.zeros_like(T)
    if batch_indices is not None:
        B_size = len(batch_indices)
    else:
        B_size = len(X_list)
        batch_indices = range(B_size)

    for idx in batch_indices:
        X, Y = X_list[idx], Y_list[idx]
        # log_p = getlogYgivenX(X, Y, W, T)
        gW, gT = computegradient(X, Y, W, T)
        
        # batch_log_p += log_p
        grad_W += gW
        grad_T += gT
    
    final_grad_W = -(C/B_size) * grad_W + W
    final_grad_T = -(C/B_size) * grad_T + T
    
    return np.concatenate([final_grad_W.flatten(), final_grad_T.flatten()])

# Stochastic objective and gradient for CRF with MCMC
def computegradient_mcmc(X, Y, W, T, S=5):
    m = X.shape[0]
    num_labels = W.shape[0]
    
    # Node potential
    node_pot = np.dot(X, W.T) # (m, 26)
    
    # Init Y
    Y_curr = np.zeros(m, dtype=np.int32)
    for i in range(m):
        log_p = node_pot[i]
        p = np.exp(log_p - np.max(log_p))
        p /= np.sum(p)
        Y_curr[i] = np.random.choice(num_labels, p=p)
        
    expected_grad_W = np.zeros_like(W)
    expected_grad_T = np.zeros_like(T)
    
    for s in range(S):
        # Sample even indices
        for i in range(1, m, 2):
            log_p = node_pot[i].copy()
            if i > 0:
                log_p += T[Y_curr[i-1], :]
            if i < m - 1:
                log_p += T[:, Y_curr[i+1]]
            p = np.exp(log_p - np.max(log_p))
            p /= np.sum(p)
            Y_curr[i] = np.random.choice(num_labels, p=p)
            
        # Sample odd indices
        for i in range(0, m, 2):
            log_p = node_pot[i].copy()
            if i > 0:
                log_p += T[Y_curr[i-1], :]
            if i < m - 1:
                log_p += T[:, Y_curr[i+1]]
            p = np.exp(log_p - np.max(log_p))
            p /= np.sum(p)
            Y_curr[i] = np.random.choice(num_labels, p=p)
            
        # Accumulate
        for i in range(m):
            expected_grad_W[Y_curr[i]] += X[i]
        
        for i in range(m-1):
            expected_grad_T[Y_curr[i], Y_curr[i+1]] += 1
            
    expected_grad_W /= S
    expected_grad_T /= S
    
    # Compute final gradient (Observed - Expected)
    grad_W = np.zeros_like(W)
    grad_T = np.zeros_like(T)
    
    for i in range(m):
        grad_W[Y[i]] += X[i]
    for i in range(m-1):
        grad_T[Y[i], Y[i+1]] += 1
        
    grad_W -= expected_grad_W
    grad_T -= expected_grad_T
    
    return grad_W, grad_T

# Stochastic objective and gradient for CRF with MCMC RB
def computegradient_mcmc_rb(X, Y, W, T, S=5):
    m = X.shape[0]
    num_labels = W.shape[0]
    
    # Node potential
    node_pot = np.dot(X, W.T) # (m, 26)
    
    # Init Y
    Y_curr = np.zeros(m, dtype=np.int32)
    for i in range(m):
        log_p = node_pot[i]
        p = np.exp(log_p - np.max(log_p))
        p /= np.sum(p)
        Y_curr[i] = np.random.choice(num_labels, p=p)
        
    expected_grad_W = np.zeros_like(W)
    expected_grad_T = np.zeros_like(T)
    
    for s in range(S):
        # Sample even indices
        for i in range(1, m, 2):
            log_p = node_pot[i].copy()
            if i > 0:
                log_p += T[Y_curr[i-1], :]
            if i < m - 1:
                log_p += T[:, Y_curr[i+1]]
            p = np.exp(log_p - np.max(log_p))
            p /= np.sum(p)
            
            expected_grad_W += np.outer(p, X[i])
            
            # Update T (connected edges)
            if i > 0:
                expected_grad_T[Y_curr[i-1], :] += p
            if i < m - 1:
                expected_grad_T[:, Y_curr[i+1]] += p
            
            Y_curr[i] = np.random.choice(num_labels, p=p)
            
        # Sample odd indices
        for i in range(0, m, 2):
            log_p = node_pot[i].copy()
            if i > 0:
                log_p += T[Y_curr[i-1], :]
            if i < m - 1:
                log_p += T[:, Y_curr[i+1]]
            p = np.exp(log_p - np.max(log_p))
            p /= np.sum(p)
            
            expected_grad_W += np.outer(p, X[i])
            
            if i > 0:
                 expected_grad_T[Y_curr[i-1], :] += p
            if i < m - 1:
                 expected_grad_T[:, Y_curr[i+1]] += p
                 
            Y_curr[i] = np.random.choice(num_labels, p=p)

    expected_grad_W /= S
    expected_grad_T /= (2 * S)
    
    # Compute final gradient (Observed - Expected)
    grad_W = np.zeros_like(W)
    grad_T = np.zeros_like(T)
    
    for i in range(m):
        grad_W[Y[i]] += X[i]
    for i in range(m-1):
        grad_T[Y[i], Y[i+1]] += 1
        
    grad_W -= expected_grad_W
    grad_T -= expected_grad_T
    
    return grad_W, grad_T

# Marginals, not gradients for MCMC
def compute_marginals_mcmc(X, W, T, S=5):
    m = X.shape[0]
    num_labels = W.shape[0]
    
    # Node potential
    node_pot = np.dot(X, W.T) # (m, 26)
    
    # Init Y
    Y_curr = np.zeros(m, dtype=np.int32)
    for i in range(m):
        log_p = node_pot[i]
        p = np.exp(log_p - np.max(log_p))
        p /= np.sum(p)
        Y_curr[i] = np.random.choice(num_labels, p=p)
        
    node_marginals = np.zeros((m, num_labels))
    edge_marginals = np.zeros((m-1, num_labels, num_labels))
    
    for s in range(S):
        # Sample even indices
        for i in range(1, m, 2):
            log_p = node_pot[i].copy()
            if i > 0:
                log_p += T[Y_curr[i-1], :]
            if i < m - 1:
                log_p += T[:, Y_curr[i+1]]
            p = np.exp(log_p - np.max(log_p))
            p /= np.sum(p)
            Y_curr[i] = np.random.choice(num_labels, p=p)
            
        # Sample odd indices
        for i in range(0, m, 2):
            log_p = node_pot[i].copy()
            if i > 0:
                log_p += T[Y_curr[i-1], :]
            if i < m - 1:
                log_p += T[:, Y_curr[i+1]]
            p = np.exp(log_p - np.max(log_p))
            p /= np.sum(p)
            Y_curr[i] = np.random.choice(num_labels, p=p)
            
        # Accumulate
        for i in range(m):
            node_marginals[i, Y_curr[i]] += 1
        
        for i in range(m-1):
            edge_marginals[i, Y_curr[i], Y_curr[i+1]] += 1
            
    node_marginals /= S
    edge_marginals /= S
    
    return node_marginals, edge_marginals

# Marginals, not gradients for MCMC RB
def compute_marginals_mcmc_rb(X, W, T, S=5):
    m = X.shape[0]
    num_labels = W.shape[0]
    
    # Node potential
    node_pot = np.dot(X, W.T) # (m, 26)
    
    # Init Y 
    Y_curr = np.zeros(m, dtype=np.int32)
    for i in range(m):
        log_p = node_pot[i]
        p = np.exp(log_p - np.max(log_p))
        p /= np.sum(p)
        Y_curr[i] = np.random.choice(num_labels, p=p)
        
    node_marginals = np.zeros((m, num_labels))
    edge_marginals = np.zeros((m-1, num_labels, num_labels))
    
    for s in range(S):
        # Sample even indices
        for i in range(1, m, 2):
            log_p = node_pot[i].copy()
            if i > 0:
                log_p += T[Y_curr[i-1], :]
            if i < m - 1:
                log_p += T[:, Y_curr[i+1]]
            p = np.exp(log_p - np.max(log_p))
            p /= np.sum(p)
            
            node_marginals[i, :] += p
            
            if i > 0:
                edge_marginals[i-1, Y_curr[i-1], :] += p
            if i < m - 1:
                edge_marginals[i, :, Y_curr[i+1]] += p
            
            Y_curr[i] = np.random.choice(num_labels, p=p)
            
        # Sample odd indices
        for i in range(0, m, 2):
            log_p = node_pot[i].copy()
            if i > 0:
                log_p += T[Y_curr[i-1], :]
            if i < m - 1:
                log_p += T[:, Y_curr[i+1]]
            p = np.exp(log_p - np.max(log_p))
            p /= np.sum(p)
            
            node_marginals[i, :] += p
            
            if i > 0:
                 edge_marginals[i-1, Y_curr[i-1], :] += p
            if i < m - 1:
                 edge_marginals[i, :, Y_curr[i+1]] += p
                 
            Y_curr[i] = np.random.choice(num_labels, p=p)

    node_marginals /= S
    edge_marginals /= (2 * S)
    
    return node_marginals, edge_marginals

# Stochastic objective and gradient for CRF with MCMC
def crf_mcmc_obj_grad(params, X_list, Y_list, C, num_labels, feature_dim, batch_indices, S=5):
    W = params[:num_labels * feature_dim].reshape(num_labels, feature_dim)
    T = params[num_labels * feature_dim:].reshape(num_labels, num_labels)
    
    batch_log_p = 0
    grad_W = np.zeros_like(W)
    grad_T = np.zeros_like(T)
    if batch_indices is not None:
        B_size = len(batch_indices)
    else:
        B_size = len(X_list)
        batch_indices = range(B_size)

    for idx in batch_indices:
        X, Y = X_list[idx], Y_list[idx]
        log_p = getlogYgivenX(X, Y, W, T)
        gW, gT = computegradient_mcmc(X, Y, W, T, S)
        
        batch_log_p += log_p
        grad_W += gW
        grad_T += gT

    obj = -(C/B_size) * batch_log_p + 0.5 * np.sum(W**2) + 0.5 * np.sum(T**2)
    
    final_grad_W = -(C/B_size) * grad_W + W
    final_grad_T = -(C/B_size) * grad_T + T
    
    return obj, np.concatenate([final_grad_W.flatten(), final_grad_T.flatten()])

# Wrapper for LBFGS with MCMC
def wrapped_crf_mcmc_obj_grad(params, X_list, Y_list, C, num_labels, feature_dim, S=5):
    global eval_counter, lbfgs_eval_x, lbfgs_eval_obj
    eval_counter += 1
    
    # Using None for batch_indices means use all data
    obj, grad = crf_mcmc_obj_grad(params, X_list, Y_list, C, num_labels, feature_dim, None, S)
    # obj = get_obj(params, X_list, Y_list, C, num_labels, feature_dim)
    
    lbfgs_eval_x.append(eval_counter)
    lbfgs_eval_obj.append(obj)
    
    return obj, grad

# gradient descent algorithm for 4a
def gradient_descent(params_init, X_list, Y_list, C, num_labels, feature_dim, B, learning_rate, n_iters):
    params = params_init.copy()
    n_total = len(X_list)
    history = []
    acc_list = []
    number_of_passes = 0
    
    for i in range(n_iters):
        # Success a pass
        if i * B >= number_of_passes * n_total:
            number_of_passes += 1
            obj = get_obj(params, X_list, Y_list, C, num_labels, feature_dim)
            acc = get_wordwise_error(params, CB_X_list_test, CB_Y_list_test, C, num_labels, feature_dim) if CB_X_list_test is not None else 0
            history.append(obj)
            acc_list.append(acc)
            print(f"Iteration {i}: Objective approx = {obj:.4f}, Error = {acc:.4f}")
        # batch
        batch_indices = np.random.choice(n_total, B, replace=False)
        
        gradient = crf_stochastic_obj_grad(
            params, X_list, Y_list, C, num_labels, feature_dim, batch_indices
        )
        
        # w = w - eta * gradient
        params = params - learning_rate * gradient            
    return params, history, acc_list

# gradient descent algorithm with momentum for 4a
def gradient_descent_momentum(params_init, X_list, Y_list, C, num_labels, feature_dim, B, learning_rate, n_iters, momentum):
    params = params_init.copy()
    n_total = len(X_list)
    history = []
    acc_list = []
    change = 0
    number_of_passes = 0
    
    for i in range(n_iters):
        # Success a pass
        if i * B >= number_of_passes * n_total:
            number_of_passes += 1
            obj = get_obj(params, X_list, Y_list, C, num_labels, feature_dim)
            acc = get_wordwise_error(params, CB_X_list_test, CB_Y_list_test, C, num_labels, feature_dim) if CB_X_list_test is not None else 0
            history.append(obj)
            acc_list.append(acc)
            print(f"Iteration {i}: Objective approx = {obj:.4f}, Error = {acc:.4f}")
        batch_indices = np.random.choice(n_total, B, replace=False)
        
        gradient = crf_stochastic_obj_grad(
            params, X_list, Y_list, C, num_labels, feature_dim, batch_indices
        )
        # Momentum
        new_change = learning_rate * gradient + momentum * change
        
        params = params - new_change
        change = new_change
        
            
    return params, history, acc_list

# gradient descent algorithm with MCMC for 4b
def gradient_descent_mcmc(params_init, X_list, Y_list, C, num_labels, feature_dim, B, learning_rate, n_iters, S=5):
    params = params_init.copy()
    n_total = len(X_list)
    history_obj = []
    history_acc = []
    number_of_passes = 0
    
    for i in range(n_iters):
        # Success a pass
        if i * B >= number_of_passes * n_total:
            number_of_passes += 1
            obj = get_obj(params, X_list, Y_list, C, num_labels, feature_dim)
            history_obj.append(obj)
            
            # Check accuracy if test data available
            acc = 0
            if CB_X_list_test is not None:
                 acc = get_wordwise_error(params, CB_X_list_test, CB_Y_list_test, C, num_labels, feature_dim)
            history_acc.append(acc)
            
            print(f"Iteration {i}: Objective approx = {obj:.4f}, Error = {acc:.4f}")
        # batch
        batch_indices = np.random.choice(n_total, B, replace=False)
        
        # Use MCMC version of obj_grad
        obj, gradient = crf_mcmc_obj_grad(
            params, X_list, Y_list, C, num_labels, feature_dim, batch_indices, S
        )
        
        # w = w - eta * gradient
        params = params - learning_rate * gradient
                    
    return params, history_obj, history_acc

# gradient descent with momentum and MCMC for 4b
def gradient_descent_momentum_mcmc(params_init, X_list, Y_list, C, num_labels, feature_dim, B, learning_rate, n_iters, momentum, S=5):
    params = params_init.copy()
    n_total = len(X_list)
    history_obj = []
    history_acc = []
    change = 0
    
    number_of_passes = 0
    
    for i in range(n_iters):
        # Success a pass
        if i * B >= number_of_passes * n_total:
            number_of_passes += 1
            obj = get_obj(params, X_list, Y_list, C, num_labels, feature_dim)
            history_obj.append(obj)
            
            # Check accuracy
            acc = 0
            if CB_X_list_test is not None:
                 acc = get_wordwise_error(params, CB_X_list_test, CB_Y_list_test, C, num_labels, feature_dim)
            history_acc.append(acc)

            print(f"Iteration {i}: Objective approx = {obj:.4f}, Error = {acc:.4f}")
        batch_indices = np.random.choice(n_total, B, replace=False)
        
        # Use MCMC version of obj_grad
        obj, gradient = crf_mcmc_obj_grad(
            params, X_list, Y_list, C, num_labels, feature_dim, batch_indices, S
        )
        
        new_change = learning_rate * gradient + momentum * change
        
        params = params - new_change
        
        change = new_change
        
            
    return params, history_obj, history_acc


def main(mode='4a'):
    C = 1000
    B = 10 # 50 too high, 15-20 is a bit high, 10-5 is good
    learning_rate = 0.0003 # 0.1 and 0.01 too high. 0.001 is still a bit unstable and 0.0005 is a bit better, 0.0001 is good
    momentum = 0.2
    num_labels = 26
    feature_dim = 128
    max_fun_eval = 100
    
    # Reset globals
    global eval_counter, lbfgs_eval_x, lbfgs_eval_obj, lbfgs_iter_x, lbfgs_iter_err, CB_X_list_train, CB_Y_list_train, CB_X_list_test, CB_Y_list_test
    
    X, Y = read_data('data/train.txt')
    X_list = [np.array(word, dtype=np.float64) for word in X.values()]
    Y_list = [np.array(labels, dtype=np.int32) for labels in Y.values()]
    
    # Load test data for callback
    CB_X_list_train = X_list
    CB_Y_list_train = Y_list
    
    X_test, Y_test = read_data('data/test.txt')
    CB_X_list_test = [np.array(word, dtype=np.float64) for word in X_test.values()]
    CB_Y_list_test = [np.array(labels, dtype=np.int32) for labels in Y_test.values()]
    
    n = len(X_list)
    
    number_of_iterations = int(max_fun_eval*n / B)
    
    if mode == '4a': # Run 4a first
        params_sgd = np.zeros(num_labels * feature_dim + num_labels * num_labels)
        optimal_params_sgd, history, error = gradient_descent(params_sgd, X_list, Y_list, C, num_labels, feature_dim, B, learning_rate, n_iters=number_of_iterations)
        
        params_sgd_momentum = np.zeros(num_labels * feature_dim + num_labels * num_labels)
        optimal_params_sgd_momentum, history_momentum, error_momentum = gradient_descent_momentum(params_sgd_momentum, X_list, Y_list, C, num_labels, feature_dim, B, learning_rate, n_iters=number_of_iterations, momentum=momentum)
        
        eval_counter = 0
        lbfgs_eval_x = []
        lbfgs_eval_obj = []
        lbfgs_iter_x = []
        lbfgs_iter_err = [] 
        
        params_lbfgs_mcmc = np.zeros(num_labels * feature_dim + num_labels * num_labels)
        try:
            opt_lbfgs, nfeval, rc = fmin_tnc(
                func=wrapped_crf_obj_grad, 
                x0=params_lbfgs_mcmc, 
                args=(X_list, Y_list, C, num_labels, feature_dim), 
                bounds=None, 
                callback=callback_lbfgs,
                maxfun=max_fun_eval # Limit
            )
        except Exception as e:
            print(f"LBFGS failed: {e}")
        
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 10))
            
            plt.subplot(2, 1, 1)
            
            sgd_x = range(max_fun_eval)
            mom_x = range(max_fun_eval)
            
            plt.plot(sgd_x, history, label='SGD')
            plt.plot(mom_x, history_momentum, label='Momentum')
            if len(lbfgs_eval_x) > 0:
                plt.plot(lbfgs_eval_x, lbfgs_eval_obj, label='LBFGS')
                
            plt.xlabel('Iterations / Evaluations')
            plt.ylabel('Objective Function Value')
            plt.title(f'Objective Function Decay (MCMC B={B} lr={learning_rate} mom={momentum})')
            plt.legend()
            plt.grid(True)
            
            # Error
            plt.subplot(2, 1, 2)
            plt.plot(sgd_x, error, label='SGD')
            plt.plot(mom_x, error_momentum, label='Momentum')
            if len(lbfgs_iter_x) > 0:
                plt.plot(lbfgs_iter_x, lbfgs_iter_err, marker='o', label='LBFGS') # lbfgs_iter_err contains accuracy here
                
            plt.xlabel('Iterations / Evaluations')
            plt.ylabel('Word-wise Error')
            plt.title(f'Test Error (MCMC B={B} lr={learning_rate} mom={momentum})')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(f'result/4a_mcmc_performance_B{B}_lr{str(learning_rate).replace(".", "_")}_mom{str(momentum).replace(".", "_")}.png')
            print(f"Plot saved to result/4a_mcmc_performance_B{B}_lr{str(learning_rate).replace(".", "_")}_mom{str(momentum).replace(".", "_")}.png")
            # plt.show()
        except Exception as e:
            print(f"Plotting failed: {e}")

    else: # Run 4b with MCMC
        for S in [2, 5, 10, 50]:
            print(f"--- Running MCMC Experiments with S={S} ---")

            # 1. SGD with MCMC
            print("Running SGD (MCMC)...")
            params_sgd_mcmc = np.zeros(num_labels * feature_dim + num_labels * num_labels)
            opt_sgd, hist_sgd_obj, hist_sgd_acc = gradient_descent_mcmc(params_sgd_mcmc, X_list, Y_list, C, num_labels, feature_dim, B, learning_rate, n_iters=number_of_iterations, S=S)
            
            # 2. Momentum with MCMC
            print("Running Momentum (MCMC)...")
            params_mom_mcmc = np.zeros(num_labels * feature_dim + num_labels * num_labels)
            opt_mom, hist_mom_obj, hist_mom_acc = gradient_descent_momentum_mcmc(params_mom_mcmc, X_list, Y_list, C, num_labels, feature_dim, B, learning_rate, n_iters=number_of_iterations, momentum=momentum, S=S)
            
            # 3. LBFGS with MCMC
            print("Running LBFGS (MCMC)...")
            eval_counter = 0
            lbfgs_eval_x = []
            lbfgs_eval_obj = []
            lbfgs_iter_x = []
            lbfgs_iter_err = [] 
            
            # LBFGS for 4b is very inconsistent, so only use this for 4a
            # Slightly better if set eta = 0.1 and stepmx=1, but still very bad.
            # params_lbfgs_mcmc = np.zeros(num_labels * feature_dim + num_labels * num_labels)
            # try:
            #     opt_lbfgs, nfeval, rc = fmin_tnc(
            #         func=wrapped_crf_mcmc_obj_grad, 
            #         x0=params_lbfgs_mcmc, 
            #         args=(X_list, Y_list, C, num_labels, feature_dim, S), 
            #         bounds=None, 
            #         eta=0.1,
            #         stepmx=1,
            #         callback=callback_lbfgs,
            #         # maxfun=max_fun_eval # Limit
            #     )
            # except Exception as e:
            #     print(f"LBFGS MCMC failed: {e}")
            
            # Plotting
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(12, 10))
                
                plt.subplot(2, 1, 1)
                
                sgd_x = range(max_fun_eval)
                mom_x = range(max_fun_eval)
                
                plt.plot(sgd_x, hist_sgd_obj, label='SGD (MCMC)')
                plt.plot(mom_x, hist_mom_obj, label='Momentum (MCMC)')
                if len(lbfgs_eval_x) > 0:
                    plt.plot(lbfgs_eval_x, lbfgs_eval_obj, label='LBFGS')
                    
                plt.xlabel('Iterations / Evaluations')
                plt.ylabel('Objective Function Value')
                plt.title(f'Objective Function Decay (MCMC S={S})')
                plt.legend()
                plt.grid(True)
                
                # Error
                plt.subplot(2, 1, 2)
                plt.plot(sgd_x, hist_sgd_acc, label='SGD (MCMC)')
                plt.plot(mom_x, hist_mom_acc, label='Momentum (MCMC)')
                if len(lbfgs_iter_x) > 0:
                    plt.plot(lbfgs_iter_x, lbfgs_iter_err, marker='o', label='LBFGS') # lbfgs_iter_err contains accuracy here
                    
                plt.xlabel('Iterations / Evaluations')
                plt.ylabel('Word-wise Error')
                plt.title(f'Test Error (MCMC S={S})')
                plt.legend()
                plt.grid(True)
                
                plt.tight_layout()
                plt.savefig(f'result/4a_mcmc_performance_B{B}_lr{str(learning_rate).replace(".", "_")}_mom{str(momentum).replace(".", "_")}_S{S}.png')
                print(f"Plot saved to result/4a_mcmc_performance_B{B}_lr{str(learning_rate).replace(".", "_")}_mom{str(momentum).replace(".", "_")}_S{S}.png")
                # plt.show()
            except Exception as e:
                print(f"Plotting failed: {e}")
    
if __name__ == "__main__":
    main(mode='4a') # Change to '4b' to run MCMC experiments