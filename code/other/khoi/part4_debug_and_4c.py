import numpy as np
from scipy.special import rel_entr
from part4 import compute_marginals_mcmc, compute_marginals_mcmc_rb
from part2 import compute_marginals
from read_data import read_data, read_model
import matplotlib.pyplot as plt

# KL divergence
def calculate_kl_divergence(p, q):
    epsilon = 1e-10
    p = np.clip(p, epsilon, 1.0) # Avoid log and division by zero
    q = np.clip(q, epsilon, 1.0)
    
    return np.sum(p * np.log(p / q))

def debug_sampler_4b(X_list, Y_list, num_labels, feature_dim):
    print("\n--- Running KL Divergence Comparison (RB vs MCMC) ---")
    
    # 1st word
    idx = 0 
    X = X_list[idx]
    

    # Load model parameters
    W, T = read_model('data/model.txt')
    
    # marginals using DP
    true_node_marginals, true_edge_marginals = compute_marginals(X, W, T)
    
    print(f"{'S':<10} | {'Node KL (MCMC)':<18} | {'Edge KL (MCMC)':<18} | {'Node KL (RB)':<18} | {'Edge KL (RB)':<18}")
    print("-" * 90)

    # 4. Run MCMC with increasing S
    sample_sizes = [5, 10, 50, 100, 500, 1000]
    
    kl_node_mcmc_list = []
    kl_edge_mcmc_list = []
    kl_node_rb_list = []
    kl_edge_rb_list = []
    
    for S in sample_sizes:
        # MCMC
        mcmc_node_m, mcmc_edge_m = compute_marginals_mcmc(X, W, T, S=S)
        
        # Nodes
        kl_node_mcmc = 0
        for s in range(X.shape[0]):
             kl_node_mcmc += calculate_kl_divergence(true_node_marginals[s], mcmc_node_m[s])
        
        # Edges
        kl_edge_mcmc = 0
        for s in range(X.shape[0]-1):
             kl_edge_mcmc += calculate_kl_divergence(true_edge_marginals[s].flatten(), mcmc_edge_m[s].flatten())

        # MCMC RB
        rb_node_m, rb_edge_m = compute_marginals_mcmc_rb(X, W, T, S=S)
        
        # Nodes
        kl_node_rb = 0
        for s in range(X.shape[0]):
             kl_node_rb += calculate_kl_divergence(true_node_marginals[s], rb_node_m[s])
        
        # Edges
        kl_edge_rb = 0
        for s in range(X.shape[0]-1):
             kl_edge_rb += calculate_kl_divergence(true_edge_marginals[s].flatten(), rb_edge_m[s].flatten())
             
        kl_node_mcmc_list.append(kl_node_mcmc)
        kl_edge_mcmc_list.append(kl_edge_mcmc)
        kl_node_rb_list.append(kl_node_rb)
        kl_edge_rb_list.append(kl_edge_rb)
        
        print(f"{S:<10} | {kl_node_mcmc:<18.4f} | {kl_edge_mcmc:<18.4f} | {kl_node_rb:<18.4f} | {kl_edge_rb:<18.4f}")

    try:
        plt.figure(figsize=(10, 6))
        
        plt.plot(sample_sizes, kl_node_mcmc_list, marker='o', label='Node Marginal KL (Standard MCMC)')
        plt.plot(sample_sizes, kl_edge_mcmc_list, marker='s', label='Edge Marginal KL (Standard MCMC)')
        plt.plot(sample_sizes, kl_node_rb_list, marker='^', label='Node Marginal KL (Rao-Blackwell)')
        plt.plot(sample_sizes, kl_edge_rb_list, marker='d', label='Edge Marginal KL (Rao-Blackwell)')
        
        plt.xlabel('Number of Samples (S)')
        plt.ylabel('Sum of KL Divergence')
        plt.title('Convergence of MCMC vs Rao-Blackwellized MCMC')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.grid(True, which="both", ls="-")
        
        plt.savefig('result/kl_divergence_comparison.png')
        print("\nPlot saved to result/kl_divergence_comparison.png")
        # plt.show()
    except Exception as e:
        print(f"Plotting failed: {e}")

if __name__ == "__main__":
    # Load data once
    train_data, train_labels = read_data('data/train.txt')
    X_list = [np.array(word, dtype=np.float64) for word in train_data.values()]
    Y_list = [np.array(labels, dtype=np.int32) for labels in train_labels.values()]
    
    num_labels = 26
    feature_dim = 128
    
    debug_sampler_4b(X_list, Y_list, num_labels, feature_dim)
