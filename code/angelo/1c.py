from itertools import product
import numpy as np
from pathlib import Path

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

    dp = np.full((m, num_labels), -np.inf, dtype=float) # dp[s,c] is best score of any sequence up to position s ending in label c
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

def load_decode_input(filepath, transpose_T=False):
    vec = np.loadtxt(filepath, dtype=float).reshape(-1)

    m = 100
    d = 128
    num_labels = 26

    num_x = m * d
    num_w = num_labels * d
    num_t = num_labels * num_labels

    expected = num_x + num_w + num_t
    assert vec.size == expected, f"Expected {expected} values, got {vec.size}"

    offset = 0

    X = vec[offset : offset + num_x].reshape(m, d)
    offset += num_x

    W = vec[offset : offset + num_w].reshape(num_labels, d)
    offset += num_w

    # decode_input stores T11, T12, ..., T1,26, T2,1, ..., T26,26.
    # with np default C-order reshape this maps directly to T[i, j] = T_{i,j}.
    T = vec[offset : offset + num_t].reshape(num_labels, num_labels)

    if transpose_T:
        T = T.T

    return X, W, T

def write_decode_output(y, filepath):
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        for label in y:
            f.write(f"{label + 1}\n")

def tiny_sanity_check():
    rng = np.random.default_rng(0)

    m = 3
    d = 128
    X = rng.normal(size=(m, d))
    W = rng.normal(size=(26, d))
    T = rng.normal(size=(26, 26))

    y_bf, score_bf = brute_force_decoder(X, W, T)
    y_dp, score_dp = max_sum_decoder(X, W, T)

    print("Tiny sanity check")
    print("Brute-force y:", y_bf)
    print("DP y         :", y_dp)
    print("Brute-force score:", score_bf)
    print("DP score         :", score_dp)

    assert tuple(y_dp) == tuple(y_bf)
    assert np.isclose(score_bf, score_dp)

    return True, score_bf, score_dp


def decode_real_case(input_path="data/decode_input.txt", transpose_T=False):
    X, W, T = load_decode_input(input_path, transpose_T=transpose_T)
    print("Loaded shapes: X", X.shape, "W", W.shape, "T", T.shape)
    y_star, best_score = max_sum_decoder(X, W, T)
    write_decode_output(y_star, "result/decode_output.txt")

    print("Best objective value:", best_score)
    print("Wrote result/decode_output.txt")

    return y_star, best_score


def main():
    tiny_sanity_check() # verify DP against brute force on a tiny case
    decode_real_case(input_path="data/decode_input.txt", transpose_T=False) # Decode the real 100-letter case

if __name__ == "__main__":
    main()
