import numpy as np
from pathlib import Path
from scipy.optimize import fmin_tnc

def logsumexp(arr):
    M = np.max(arr)
    return M + np.log(np.sum(np.exp(arr - M)))

def precompute_node_scores(X, W):
    return X @ W.T

def forward_log_messages(phi, T):
    m, C = phi.shape
    alpha = np.zeros((m, C), dtype=float)

    alpha[0, :] = 0.0

    for i in range(1, m):
        for c in range(C):
            vals = alpha[i - 1, :] + phi[i - 1, :] + T[:, c]
            alpha[i, c] = logsumexp(vals)

    return alpha

def backward_log_messages(phi, T):
    m, C = phi.shape
    beta = np.zeros((m, C), dtype=float)

    # base case
    beta[m - 1, :] = 0.0

    # recurrence
    for i in range(m - 2, -1, -1):
        for c in range(C):
            vals = phi[i + 1, :] + T[c, :] + beta[i + 1, :]
            beta[i, c] = logsumexp(vals)

    return beta

def compute_logZ(phi, alpha):
    return logsumexp(phi[-1, :] + alpha[-1, :])

def compute_node_log_marginals(phi, alpha, beta, logZ):
    return alpha + phi + beta - logZ

def compute_edge_log_marginals(phi, alpha, beta, T, logZ):
    m, C = phi.shape
    edge_log_marg = np.zeros((m - 1, C, C), dtype=float)

    for i in range(m - 1):
        for p in range(C):
            edge_log_marg[i, p, :] = (
                alpha[i, p]
                + phi[i, p]
                + T[p, :]
                + phi[i + 1, :]
                + beta[i + 1, :]
                - logZ
            )

    return edge_log_marg

def log_likelihood_and_gradient_single(X, y, W, T):
    phi = precompute_node_scores(X, W)
    alpha = forward_log_messages(phi, T)
    beta = backward_log_messages(phi, T)
    logZ = compute_logZ(phi, alpha)

    m = len(X)

    gold_score = 0.0
    for s in range(m):
        gold_score += phi[s, y[s]]
    for s in range(m - 1):
        gold_score += T[y[s], y[s + 1]]

    logp = gold_score - logZ

    node_log_marg = compute_node_log_marginals(phi, alpha, beta, logZ)
    node_marg = np.exp(node_log_marg)

    edge_log_marg = compute_edge_log_marginals(phi, alpha, beta, T, logZ)
    edge_marg = np.exp(edge_log_marg)

    grad_W = np.zeros_like(W)
    for s in range(m):
        for c in range(W.shape[0]):
            coeff = (1.0 if y[s] == c else 0.0) - node_marg[s, c]
            grad_W[c] += coeff * X[s]

    grad_T = np.zeros_like(T)
    for s in range(m - 1):
        grad_T[y[s], y[s + 1]] += 1.0
        grad_T -= edge_marg[s]

    return logp, grad_W, grad_T

def average_log_likelihood_and_gradient(words, W, T):
    total_logp = 0.0
    total_grad_W = np.zeros_like(W)
    total_grad_T = np.zeros_like(T)

    n = len(words)

    for word in words:
        X = word["X"]
        y = word["y"]

        logp, grad_W, grad_T = log_likelihood_and_gradient_single(X, y, W, T)

        total_logp += logp
        total_grad_W += grad_W
        total_grad_T += grad_T

    avg_logp = total_logp / n
    avg_grad_W = total_grad_W / n
    avg_grad_T = total_grad_T / n

    return avg_logp, avg_grad_W, avg_grad_T

def pack_params(W, T):
    return np.concatenate([W.reshape(-1), T.reshape(-1)])

def unpack_params(theta, num_labels=26, feat_dim=128):
    num_w = num_labels * feat_dim
    num_t = num_labels * num_labels

    W = theta[:num_w].reshape(num_labels, feat_dim)
    T = theta[num_w:num_w + num_t].reshape(num_labels, num_labels)

    return W, T

def parse_ocr_file(filepath):
    words = []
    cur_X = []
    cur_y = []

    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            letter = parts[1]
            next_id = int(parts[2])
            x = np.array([float(v) for v in parts[5:]], dtype=float)

            assert x.shape[0] == 128

            cur_X.append(x)
            cur_y.append(ord(letter) - ord("a"))

            if next_id == -1:
                words.append({
                    "X": np.vstack(cur_X),
                    "y": np.array(cur_y, dtype=int),
                })
                cur_X = []
                cur_y = []

    assert len(cur_X) == 0 and len(cur_y) == 0
    return words

def write_vector(filepath, vec):
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, vec, fmt="%.18e")


def objective_and_gradient(theta, words, C=1000):
    W, T = unpack_params(theta)

    avg_logp, avg_grad_W_logp, avg_grad_T_logp = average_log_likelihood_and_gradient(words, W, T)

    obj = -C * avg_logp + 0.5 * np.sum(W * W) + 0.5 * np.sum(T * T)

    grad_W = -C * avg_grad_W_logp + W
    grad_T = -C * avg_grad_T_logp + T

    grad = pack_params(grad_W, grad_T)

    return obj, grad

def max_sum_decoder(X, W, T):
    m = len(X)
    num_labels = 26

    dp = np.full((m, num_labels), -np.inf, dtype=float)
    bp = np.full((m, num_labels), -1, dtype=int)

    for c in range(num_labels):
        dp[0, c] = np.dot(W[c], X[0])

    for s in range(1, m):
        for c in range(num_labels):
            node_score = np.dot(W[c], X[s])
            best_prev_score = -np.inf
            best_prev_label = -1

            for p in range(num_labels):
                cand = dp[s - 1, p] + T[p, c] + node_score
                if cand > best_prev_score:
                    best_prev_score = cand
                    best_prev_label = p

            dp[s, c] = best_prev_score
            bp[s, c] = best_prev_label

    best_last = int(np.argmax(dp[m - 1]))
    best_score = float(dp[m - 1, best_last])

    best_y = [0] * m
    best_y[m - 1] = best_last

    for s in range(m - 1, 0, -1):
        best_y[s - 1] = int(bp[s, best_y[s]])

    return best_y, best_score

def write_predictions(filepath, predictions):
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        for y in predictions:
            f.write(f"{y + 1}\n")   # convert 0-based to 1-based


def main():
    words = parse_ocr_file("data/train.txt")

    theta0 = np.zeros(26*128 + 26*26)

    theta_opt, nfeval, rc = fmin_tnc(
        func=lambda th: objective_and_gradient(th, words),
        x0=theta0,
        bounds=None
    )

    write_vector("result/solution.txt", theta_opt)
    
    W_opt, T_opt = unpack_params(theta_opt)

    final_obj, _ = objective_and_gradient(theta_opt, words)
    print("final_objective =", final_obj)
    print("nfeval =", nfeval)
    print("rc =", rc)

    test_words = parse_ocr_file("data/test.txt")

    predictions = []
    for word in test_words:
        y_pred, _ = max_sum_decoder(word["X"], W_opt, T_opt)
        predictions.extend(y_pred)

    write_predictions("result/prediction.txt", predictions)
    print("wrote result/solution.txt")
    print("wrote result/prediction.txt")


if __name__ == "__main__":
    main()
