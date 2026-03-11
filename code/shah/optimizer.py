#!/usr/bin/env python
# coding: utf-8

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ALPHABET = "abcdefghijklmnopqrstuvwxyz"


@dataclass
class WordSample:
    x_seq: List[List[float]]
    y_seq: List[int]  # 0-indexed labels


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def logsumexp(values: Sequence[float]) -> float:
    m = max(values)
    return m + math.log(sum(math.exp(v - m) for v in values))


def parse_model_vector(path: str | Path, c: int = 26, d: int = 128) -> Tuple[List[List[float]], List[List[float]]]:
    """Load model vector from Eq.(7): [w1',...,wC',T11,T12,...,TCC]' ."""
    vals = [float(tok) for tok in Path(path).read_text().replace("\\n", " ").split()]
    expected = c * d + c * c
    if len(vals) != expected:
        raise ValueError(f"Expected {expected} values in model vector, got {len(vals)} from {path}")

    w_flat = vals[: c * d]
    t_flat = vals[c * d :]

    w = [w_flat[i * d : (i + 1) * d] for i in range(c)]
    t = [t_flat[i * c : (i + 1) * c] for i in range(c)]
    return w, t


def parse_train_txt(path: str | Path) -> List[WordSample]:
    """Parse OCR-style train rows: id, letter, next_id, word_id, position, x_1..x_128."""
    rows_by_word: Dict[int, List[Tuple[int, int, List[float]]]] = {}

    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 6:
            raise ValueError(f"Malformed train row (expected >=6 columns): {line}")

        letter = parts[1].lower()
        if letter not in ALPHABET:
            raise ValueError(f"Unexpected label {letter!r} in row: {line}")

        word_id = int(parts[3])
        position = int(parts[4])
        features = [float(v) for v in parts[5:]]

        y = ALPHABET.index(letter)
        rows_by_word.setdefault(word_id, []).append((position, y, features))

    words: List[WordSample] = []
    for _, rows in sorted(rows_by_word.items(), key=lambda kv: kv[0]):
        rows.sort(key=lambda z: z[0])
        x_seq = [r[2] for r in rows]
        y_seq = [r[1] for r in rows]
        words.append(WordSample(x_seq=x_seq, y_seq=y_seq))

    return words


def forward_backward_log_marginals(
    x_seq: Sequence[Sequence[float]],
    w: Sequence[Sequence[float]],
    t: Sequence[Sequence[float]],
) -> Tuple[List[List[float]], List[List[List[float]]], float]:
    """Return node marginals, edge marginals, and logZ for one word."""
    m = len(x_seq)
    c = len(w)

    emissions = [[dot(w[j], x_seq[s]) for j in range(c)] for s in range(m)]

    fwd = [[float("-inf")] * c for _ in range(m)]
    back = [[float("-inf")] * c for _ in range(m)]

    for j in range(c):
        fwd[0][j] = emissions[0][j]

    for s in range(1, m):
        for j in range(c):
            fwd[s][j] = emissions[s][j] + logsumexp([fwd[s - 1][i] + t[i][j] for i in range(c)])

    for j in range(c):
        back[m - 1][j] = 0.0

    for s in range(m - 2, -1, -1):
        for i in range(c):
            back[s][i] = logsumexp([t[i][j] + emissions[s + 1][j] + back[s + 1][j] for j in range(c)])

    log_z = logsumexp(fwd[m - 1])

    node_marginals = [[0.0] * c for _ in range(m)]
    for s in range(m):
        for j in range(c):
            node_marginals[s][j] = math.exp(fwd[s][j] + back[s][j] - log_z)

    edge_marginals = [[[0.0] * c for _ in range(c)] for _ in range(m - 1)]
    for s in range(m - 1):
        for i in range(c):
            for j in range(c):
                edge_marginals[s][i][j] = math.exp(fwd[s][i] + t[i][j] + emissions[s + 1][j] + back[s + 1][j] - log_z)

    for s in range(len(node_marginals)):
        if abs(sum(node_marginals[s]) - 1.0) > 1e-6:
            raise AssertionError("Node marginals do not sum to 1")
    for s in range(len(edge_marginals)):
        total = 0.0
        for i in range(len(w)):
            total += sum(edge_marginals[s][i])
        if abs(total - 1.0) > 1e-6:
            raise AssertionError("Edge marginals do not sum to 1")

    return node_marginals, edge_marginals, log_z


def average_gradients_over_train(
    words: Sequence[WordSample],
    w: Sequence[Sequence[float]],
    t: Sequence[Sequence[float]],
) -> Tuple[List[List[float]], List[List[float]], float]:
    """Compute 1/n sum_t grad_w log p(y^t|X^t), grad_T log p(y^t|X^t), and avg log-likelihood."""
    n = len(words)
    c = len(w)
    d = len(w[0])

    grad_w = [[0.0] * d for _ in range(c)]
    grad_t = [[0.0] * c for _ in range(c)]
    loglik_sum = 0.0

    for sample in words:
        x_seq = sample.x_seq
        y_seq = sample.y_seq
        m = len(x_seq)

        node_marg, edge_marg, log_z = forward_backward_log_marginals(x_seq, w, t)

        score = sum(dot(w[y_seq[s]], x_seq[s]) for s in range(m))
        score += sum(t[y_seq[s]][y_seq[s + 1]] for s in range(m - 1))
        loglik_sum += score - log_z

        for s in range(m):
            xs = x_seq[s]
            for cls in range(c):
                coeff = (1.0 if y_seq[s] == cls else 0.0) - node_marg[s][cls]
                if coeff == 0.0:
                    continue
                for k in range(d):
                    grad_w[cls][k] += coeff * xs[k]

        for s in range(m - 1):
            y_left = y_seq[s]
            y_right = y_seq[s + 1]
            for i in range(c):
                for j in range(c):
                    coeff = (1.0 if (i == y_left and j == y_right) else 0.0) - edge_marg[s][i][j]
                    grad_t[i][j] += coeff

    inv_n = 1.0 / n
    for cls in range(c):
        for k in range(d):
            grad_w[cls][k] *= inv_n
    for i in range(c):
        for j in range(c):
            grad_t[i][j] *= inv_n

    return grad_w, grad_t, loglik_sum * inv_n


def write_gradient_vector(path: str | Path, grad_w: Sequence[Sequence[float]], grad_t: Sequence[Sequence[float]]) -> None:
    """Write [grad_w1',...,grad_wC', grad_T11,grad_T12,...,grad_TCC]' to file."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    vals: List[float] = []
    for row in grad_w:
        vals.extend(row)
    for row in grad_t:
        vals.extend(row)

    out.write_text("\n".join(f"{v:.12f}" for v in vals) + "\n")


def run_train_gradient_pipeline(
    model_path: str | Path = "data/model.txt",
    train_path: str | Path = "data/train.txt",
    output_path: str | Path = "result/gradient.txt",
) -> Tuple[List[List[float]], List[List[float]], float]:
    w, t = parse_model_vector(model_path)
    words = parse_train_txt(train_path)
    grad_w, grad_t, avg_loglik = average_gradients_over_train(words, w, t)
    write_gradient_vector(output_path, grad_w, grad_t)
    return grad_w, grad_t, avg_loglik


def self_test() -> None:
    # Tiny consistency checks on probabilities and dimensions.
    w = [[0.2, -0.1], [0.0, 0.3], [-0.2, 0.4]]
    t = [[0.1, -0.2, 0.0], [0.05, 0.2, -0.1], [-0.3, 0.1, 0.15]]
    words = [
        WordSample(x_seq=[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], y_seq=[0, 1, 2]),
        WordSample(x_seq=[[0.3, 0.7], [0.5, 0.5]], y_seq=[2, 1]),
    ]

    node, edge, _ = forward_backward_log_marginals(words[0].x_seq, w, t)
    for s in range(len(node)):
        if abs(sum(node[s]) - 1.0) > 1e-6:
            raise AssertionError("Node marginals do not sum to 1")
    for s in range(len(edge)):
        total = 0.0
        for i in range(len(w)):
            total += sum(edge[s][i])
        if abs(total - 1.0) > 1e-6:
            raise AssertionError("Edge marginals do not sum to 1")

    gw, gt, ll = average_gradients_over_train(words, w, t)
    if len(gw) != 3 or len(gw[0]) != 2 or len(gt) != 3 or len(gt[0]) != 3:
        raise AssertionError("Gradient shape mismatch")
    print("Self-test passed. avg_loglik=", ll)


g_w, g_T, avg_loglik = run_train_gradient_pipeline("data/model.txt", "data/train.txt", "result/gradient.txt")
print(f"Average log-likelihood (1/n sum_t log p(y^t|X^t)): {avg_loglik}")
#print(f"grad_w: {g_w}, grad_T: {g_T}")


@dataclass
class DecodeResult:
    labels: List[int]
    score: float

def decode_viterbi(X: List[List[float]], W: List[List[float]], T: List[List[float]]) -> DecodeResult:
    """O(m|Y|^2) dynamic-programming decoder for equation (3).

    Recurrence:
      dp[s][j] = <w_j, x_s> + max_i (dp[s-1][i] + T[i][j])
    with base dp[0][j] = <w_j, x_0>.
    """
    m = len(X)
    d = len(X[0])
    C = len(W)
    assert all(len(w) == d for w in W)
    assert len(T) == C and all(len(row) == C for row in T)

    emissions = [[dot(X[s], W[j]) for j in range(C)] for s in range(m)]

    dp = [[float("-inf")] * C for _ in range(m)]
    back = [[-1] * C for _ in range(m)]

    for j in range(C):
        dp[0][j] = emissions[0][j]

    for s in range(1, m):
        for j in range(C):
            best_prev_score = float("-inf")
            best_prev_label = -1
            for i in range(C):
                cand = dp[s - 1][i] + T[i][j]
                if cand > best_prev_score:
                    best_prev_score = cand
                    best_prev_label = i
            dp[s][j] = emissions[s][j] + best_prev_score
            back[s][j] = best_prev_label

    last = max(range(C), key=lambda j: dp[m - 1][j])
    best_score = dp[m - 1][last]

    y = [0] * m
    y[m - 1] = last
    for s in range(m - 1, 0, -1):
        y[s - 1] = back[s][y[s]]

    return DecodeResult(labels=y, score=best_score)

@dataclass
class OCRRow:
    row_index: int
    letter: Optional[str]
    word_id: int
    position: int
    x: List[float]


@dataclass
class WordSample:
    x_seq: List[List[float]]
    y_seq: List[int]  # 0-indexed labels


def parse_ocr_file(path: str | Path, require_labels: bool, d: int = 128) -> List[OCRRow]:
    rows: List[OCRRow] = []
    for idx, line in enumerate(Path(path).read_text().splitlines()):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5 + d:
            raise ValueError(f"Malformed OCR row (expected at least {5+d} tokens): {line}")

        letter_raw = parts[1].lower()
        letter = letter_raw if letter_raw in ALPHABET else None
        if require_labels and letter is None:
            raise ValueError(f"Label required but got {parts[1]!r} in row: {line}")

        word_id = int(parts[3])
        position = int(parts[4])
        x = [float(v) for v in parts[5 : 5 + d]]

        rows.append(OCRRow(row_index=idx, letter=letter, word_id=word_id, position=position, x=x))

    return rows


def rows_to_word_samples(rows: Sequence[OCRRow]) -> List[WordSample]:
    grouped: Dict[int, List[OCRRow]] = {}
    for r in rows:
        grouped.setdefault(r.word_id, []).append(r)

    words: List[WordSample] = []
    for _, group in sorted(grouped.items(), key=lambda kv: kv[0]):
        group_sorted = sorted(group, key=lambda r: r.position)
        x_seq = [r.x for r in group_sorted]
        y_seq = [ALPHABET.index(r.letter) for r in group_sorted if r.letter is not None]
        if len(y_seq) != len(x_seq):
            raise ValueError("Missing labels while building training word samples")
        words.append(WordSample(x_seq=x_seq, y_seq=y_seq))

    return words


def pack_params(w: Sequence[Sequence[float]], t: Sequence[Sequence[float]]) -> List[float]:
    vals: List[float] = []
    for row in w:
        vals.extend(row)
    for row in t:
        vals.extend(row)
    return vals


def unpack_params(theta: Sequence[float], c: int = 26, d: int = 128) -> Tuple[List[List[float]], List[List[float]]]:
    expected = c * d + c * c
    if len(theta) != expected:
        raise ValueError(f"Expected parameter length {expected}, got {len(theta)}")

    w_flat = theta[: c * d]
    t_flat = theta[c * d :]

    w = [list(w_flat[i * d : (i + 1) * d]) for i in range(c)]
    t = [list(t_flat[i * c : (i + 1) * c]) for i in range(c)]
    return w, t

###################
def objective_and_gradient(theta: Sequence[float], words, c_reg: float) -> Tuple[float, List[float]]:
    """
    Matches lab Eq.(6):
        min  -(C/n) * sum_t log p(y^t|X^t) + 1/2 sum_c ||w_c||^2 + 1/2 sum_{i,j} T_ij^2
    """
    W, T = unpack_params(theta)
    n = len(words)
    c = len(W)
    d = len(W[0])

    # accumulate UN-SCALED sums over dataset first
    total_nll = 0.0
    gradW = [[0.0] * d for _ in range(c)]     # gradient of sum_t NLL_t
    gradT = [[0.0] * c for _ in range(c)]     # gradient of sum_t NLL_t

    for sample in words:
        x_seq = sample.x_seq
        y_seq = sample.y_seq   # 0-index labels
        m = len(x_seq)

        node_marg, edge_marg, log_z = forward_backward_log_marginals(x_seq, W, T)

        # score(y, x) = sum_s <w_{y_s}, x_s> + sum_s T_{y_s, y_{s+1}}
        score = 0.0
        for s in range(m):
            score += dot(W[y_seq[s]], x_seq[s])
        for s in range(m - 1):
            score += T[y_seq[s]][y_seq[s + 1]]

        # NLL_t = logZ - score
        total_nll += (log_z - score)

        # grad of NLL wrt W: expected - empirical
        for s in range(m):
            xs = x_seq[s]

            # add expected contributions for all classes
            for cls in range(c):
                coeff = node_marg[s][cls]
                if coeff != 0.0:
                    for k in range(d):
                        gradW[cls][k] += coeff * xs[k]

            # subtract empirical contribution for the true class
            y_true = y_seq[s]
            for k in range(d):
                gradW[y_true][k] -= xs[k]

        # grad of NLL wrt T: expected - empirical
        for s in range(m - 1):
            # add expected (full matrix for this position)
            for i in range(c):
                row = edge_marg[s][i]
                for j in range(c):
                    gradT[i][j] += row[j]

            # subtract 1 on the true transition
            yi = y_seq[s]
            yj = y_seq[s + 1]
            gradT[yi][yj] -= 1.0

    # Now apply the lab’s scaling: (C/n) * sum_t NLL_t
    scale = c_reg / float(n)
    total_data = total_nll * scale
    for cls in range(c):
        for k in range(d):
            gradW[cls][k] *= scale
    for i in range(c):
        for j in range(c):
            gradT[i][j] *= scale

    # Regularizer: + 1/2 ||W||^2 + 1/2 ||T||^2  (NO c_reg here)
    reg = 0.0
    for cls in range(c):
        reg += sum(v * v for v in W[cls])
    for i in range(c):
        reg += sum(v * v for v in T[i])
    reg *= 0.5

    obj = total_data + reg

    # Add gradients of regularizer: +W and +T  (NO c_reg here)
    for cls in range(c):
        for k in range(d):
            gradW[cls][k] += W[cls][k]
    for i in range(c):
        for j in range(c):
            gradT[i][j] += T[i][j]

    return obj, pack_params(gradW, gradT)


def write_solution(path: str | Path, theta: Sequence[float]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(f"{v:.12f}" for v in theta) + "\n")

###############################
def train_with_fmin_tnc(
    train_words: Sequence[WordSample],
    c_reg: float = 1000.0,
    maxfun: int = 200,
) -> Tuple[List[float], int, int]:
    from scipy.optimize import fmin_tnc

    dim = 26 * 128 + 26 * 26
    theta0 = [0.0] * dim

    xopt, nfeval, rc = fmin_tnc(
        func=lambda th, *args: objective_and_gradient(th, train_words, c_reg),
        x0=theta0,
        approx_grad=False,
        maxfun=maxfun,
        messages=0,
    )
    return list(xopt), int(nfeval), int(rc)


def predict_test_rows(test_rows: Sequence[OCRRow], theta: Sequence[float]) -> List[int]:
    w, t = unpack_params(theta)

    grouped: Dict[int, List[OCRRow]] = {}
    for r in test_rows:
        grouped.setdefault(r.word_id, []).append(r)

    pred_by_row: Dict[int, int] = {}

    for _, group in sorted(grouped.items(), key=lambda kv: kv[0]):
        group_sorted = sorted(group, key=lambda r: r.position)
        x_seq = [r.x for r in group_sorted]

        res = decode_viterbi(x_seq, w, t)
        for r, y0 in zip(group_sorted, res.labels):
            pred_by_row[r.row_index] = y0 + 1  # write as 1..26

    return [pred_by_row[r.row_index] for r in sorted(test_rows, key=lambda r: r.row_index)]


def write_predictions(path: str | Path, preds: Sequence[int]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(str(p) for p in preds) + "\n")

'''
import random
def create_dummy_data(num_words=3, max_word_length=5, d=128):
    """Generates a small list of WordSamples with random 128D features."""
    random.seed(100) # For reproducible testing
    words = []
    
    for _ in range(num_words):
        word_length = random.randint(2, max_word_length)
        x_seq = []
        y_seq = []
        
        for _ in range(word_length):
            # Random 128-dimensional feature vector
            x_seq.append([random.uniform(-0.5, 0.5) for _ in range(d)])
            # Random label between 0 and 25 (for 26 letters in ALPHABET)
            y_seq.append(random.randint(0, 25))
            
        words.append(WordSample(x_seq=x_seq, y_seq=y_seq))
        
    return words

def run_small_test():
    print("Generating small sample training set...")
    small_train_words = create_dummy_data(num_words=2, max_word_length=4)
    
    # smaller maxfun for a quick test
    test_maxfun = 15 
    test_c_reg = 1000.0
    
    print(f"Starting fmin_tnc optimization (maxfun={test_maxfun})...")
    theta_opt, nfeval, rc = train_with_fmin_tnc(
        small_train_words, 
        c_reg=test_c_reg, 
        maxfun=test_maxfun
    )
    
    print("\n--- Test Results ---")
    print(f"Return Code (rc): {rc} (0, 1, or 2 indicate success/convergence)")
    print(f"Number of function evaluations: {nfeval}")
    
    final_obj, _ = objective_and_gradient(theta_opt, small_train_words, c_reg=test_c_reg)
    print(f"Final objective value: {final_obj:.4f}")
    
    print(f"Output theta length: {len(theta_opt)} (Expected: 4004)")


run_small_test()
'''
train = "data/train.txt"
test = "data/test.txt"
c_reg = 1000.0
maxfun = 200
solution = "result/solution.txt"
prediction = "result/prediction.txt"

train_rows = parse_ocr_file(train, require_labels=True)
train_words = rows_to_word_samples(train_rows)

theta_opt, nfeval, rc = train_with_fmin_tnc(train_words, c_reg=c_reg, maxfun=maxfun)
write_solution(solution, theta_opt)

test_rows = parse_ocr_file(test, require_labels=False)
preds = predict_test_rows(test_rows, theta_opt)
write_predictions(prediction, preds)

obj, _ = objective_and_gradient(theta_opt, train_words, c_reg=c_reg)
print(f"Optimization done. rc={rc}, nfeval={nfeval}")
print(f"Final objective value (Eq.6): {obj}")
print(f"Wrote learned parameters to: {solution}")
print(f"Wrote test predictions to: {prediction}")



