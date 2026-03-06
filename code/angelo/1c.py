from itertools import product
import numpy as np

# returns objective value
def objective_value_function(X, W, T, y):
    m = len(X)
    
    # length check
    assert len(y) == m
    # feature dim match
    assert W.shape[1] == X.shape[1]
    assert T.shape == (26,26)

    # node_score = sum_s dot(W[y[s]], X[s])
    node_score = 0.0
    for s in range(m):
        node_score += np.dot(W[y[s]] , X[s])

    # edge_score = sum_s T[y[s], y[s+1]]
    edge_score = 0.0
    for s in range(m - 1):
        edge_score += T[y[s], y[s+1]]

    return node_score + edge_score

def brute_force_decoder(X, W, T):
    m = len(X)

    best_y = None
    best_score = -np.inf

    # iterate over all seq of length m with labels 0..25
    for y_candidate in product(range(26), repeat=m):
        # compute objective value
        score = objective_value_function(X, W, T, y_candidate)
        # if better than current best, update both
        if score > best_score:
            best_score = score
            best_y = y_candidate

    return best_y, best_score

def main():
    print("Hello from crf-lab1-512!")


if __name__ == "__main__":
    main()
