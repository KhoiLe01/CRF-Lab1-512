from numpy.random import rand
from part2a import getlogYgivenX, computegradient, save_solution
import numpy as np
from read_data import read_data

# Stochastic objective and gradient for CRF
def crf_stochastic_obj_grad(params, X_list, Y_list, C, num_labels, feature_dim, batch_indices):
    # global eval_counter
    # eval_counter += 1 # Increment for every function evaluation
    
    W = params[:num_labels * feature_dim].reshape(num_labels, feature_dim)
    T = params[num_labels * feature_dim:].reshape(num_labels, num_labels)
    
    batch_log_p = 0
    grad_W = np.zeros_like(W)
    grad_T = np.zeros_like(T)
    B_size = len(batch_indices)
    n_total = len(X_list)

    for idx in batch_indices:
        X, Y = X_list[idx], Y_list[idx]
        log_p = getlogYgivenX(X, Y, W, T)
        gW, gT = computegradient(X, Y, W, T)
        
        batch_log_p += log_p
        grad_W += gW
        grad_T += gT

    obj = -(C/B_size) * batch_log_p + 0.5 * np.sum(W**2) + 0.5 * np.sum(T**2)
    
    final_grad_W = -(C/B_size) * grad_W + W
    final_grad_T = -(C/B_size) * grad_T + T
    
    return obj, np.concatenate([final_grad_W.flatten(), final_grad_T.flatten()])

# gradient descent algorithm
def gradient_descent(params_init, X_list, Y_list, C, num_labels, feature_dim, B, learning_rate, n_iters):
    params = params_init.copy()
    n_total = len(X_list)
    history = []
    
    for i in range(n_iters):
        # batch
        batch_indices = np.random.choice(n_total, B, replace=False)
        
        obj, gradient = crf_stochastic_obj_grad(
            params, X_list, Y_list, C, num_labels, feature_dim, batch_indices
        )
        
        # w = w - eta * gradient
        params = params - learning_rate * gradient
        
        # log
        if i % 10 == 0:
            history.append(obj)
            print(f"Iteration {i}: Objective approx = {obj:.4f}")
            
    return params, history

# gradient descent algorithm with momentum
def gradient_descent_momentum(params_init, X_list, Y_list, C, num_labels, feature_dim, B, learning_rate, n_iters, momentum):
    params = params_init.copy()
    n_total = len(X_list)
    history = []
    change = 0
    
    for i in range(n_iters):
        batch_indices = np.random.choice(n_total, B, replace=False)
        
        obj, gradient = crf_stochastic_obj_grad(
            params, X_list, Y_list, C, num_labels, feature_dim, batch_indices
        )
        
        new_change = learning_rate * gradient + momentum * change
        
        params = params - new_change
        change = new_change
        
        if i % 10 == 0:
            history.append(obj)
            print(f"Iteration {i}: Objective approx = {obj:.4f}")
            
    return params, history

def main():
    C = 1000
    B = 10
    learning_rate = 0.01
    momentum = 0.5
    num_labels = 26
    feature_dim = 128
    
    eval_counter = 0
    objective_history = []
    test_error_history = []
    
    X, Y = read_data('data/train.txt')
    X_list = [np.array(word, dtype=np.float64) for word in X.values()]
    Y_list = [np.array(labels, dtype=np.int32) for labels in Y.values()]
    
    params_sgd = np.zeros(num_labels * feature_dim + num_labels * num_labels)
    optimal_params_sgd, history = gradient_descent(params_sgd, X_list, Y_list, C, num_labels, feature_dim, B, learning_rate, n_iters=1000)
    
    W_opt = optimal_params_sgd[:num_labels * feature_dim].reshape(num_labels, feature_dim)
    T_opt = optimal_params_sgd[num_labels * feature_dim:].reshape(num_labels, num_labels)
    save_solution(W_opt, T_opt, "result/sgd.txt")
    
    params_sgd_momentum = np.zeros(num_labels * feature_dim + num_labels * num_labels)
    optimal_params_sgd_momentum, history_momentum = gradient_descent_momentum(params_sgd_momentum, X_list, Y_list, C, num_labels, feature_dim, B, learning_rate, n_iters=1000, momentum=momentum)
    
    W_opt = optimal_params_sgd_momentum[:num_labels * feature_dim].reshape(num_labels, feature_dim)
    T_opt = optimal_params_sgd_momentum[num_labels * feature_dim:].reshape(num_labels, num_labels)
    save_solution(W_opt, T_opt, "result/sgd_momentum.txt")
    
if __name__ == "__main__":
    main()