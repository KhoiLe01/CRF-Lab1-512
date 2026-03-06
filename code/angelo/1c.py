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

def max_sum_decoder(X, W, T):
    m = len(X)
    num_labels = 26

    dp = np.full((m, num_labels), -np.inf) # dp[s,c] is best score of any sequence up to position s ending in label c
    bp = np.full((m, num_labels), -1, dtype=int) # bp[s,c] best prev label p that leads to dp[s,c]

    # base case: position 0
    for c in range(num_labels):
        dp[0, c] = np.dot(W[c], X[0])

    # recurrence
    for s in range(1, m):
        for c in range(num_labels):
            best_prev_score = -np.inf
            best_prev_label = -1
            
            node_score = np.dot(W[c], X[s]) # for efficiency, since doesn't depend on p we compute it once before

            for p in range(num_labels):
                cand = dp[s-1, p] + T[p, c] + node_score
                
                if cand > best_prev_score:
                    best_prev_score = cand
                    best_prev_label = p
            
            dp[s,c] = best_prev_score
            bp[s,c] = best_prev_label

    # termination
    best_last = int(np.argmax(dp[m-1]))
    best_score = float(dp[m-1, best_last])

    # backtrack
    best_y = [0] * m
    best_y[m-1] = best_last

    for s in range(m-1, 0, -1):
        best_y[s-1] = int(bp[s, best_y[s]])

    return best_y, best_score

def main():
    print("Hello from crf-lab1-512!")


if __name__ == "__main__":
    main()
