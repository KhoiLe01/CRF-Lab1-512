import numpy as np
from pathlib import Path

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

    sanity_check_marginals(node_marg, edge_marg)

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

def sanity_check_marginals(node_marg, edge_marg, tol=1e-6):
    m = node_marg.shape[0]

    for s in range(m):
        assert abs(np.sum(node_marg[s]) - 1.0) < tol, f"node marginals at position {s} do not sum to 1"

    for s in range(m - 1):
        assert abs(np.sum(edge_marg[s]) - 1.0) < tol, f"edge marginals at edge {s} do not sum to 1"

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

def write_gradient(filepath, grad_W, grad_T):
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    vec = pack_params(grad_W, grad_T)
    np.savetxt(path, vec, fmt="%.18e")


def load_model(filepath, num_labels=26, feat_dim=128):
    vec = np.loadtxt(filepath, dtype=float).reshape(-1)

    num_w = num_labels * feat_dim
    num_t = num_labels * num_labels

    W = vec[:num_w].reshape(num_labels, feat_dim)
    T = vec[num_w:num_w + num_t].reshape(num_labels, num_labels)

    return W, T

#with open("../../data/train.txt", "r") as f:
#    for _ in range(3):
#        print(f.readline().strip())

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

def main():
    words = parse_ocr_file("data/train.txt")
    W, T = load_model("data/model.txt")

    avg_logp, avg_grad_W, avg_grad_T = average_log_likelihood_and_gradient(words, W, T)
    write_gradient("result/gradient.txt", avg_grad_W, avg_grad_T)

    print("num_words =", len(words))
    print("avg_log_likelihood =", avg_logp)

if __name__ == "__main__":
    main()
