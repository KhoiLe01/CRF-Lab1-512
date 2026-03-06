#!/usr/bin/env python
# coding: utf-8

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from random import Random
from typing import List, Sequence, Tuple
from pathlib import Path


@dataclass
class DecodeResult:
    labels: List[int]
    score: float


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def score_sequence(y: Sequence[int], X: List[List[float]], W: List[List[float]], T: List[List[float]]) -> float:
    """Compute CRF MAP objective for a full label sequence y.

    y uses 0-based labels in [0, C-1].
    X shape: (m, d), W shape: (C, d), T shape: (C, C).
    """
    m = len(y)
    node = sum(dot(W[y[s]], X[s]) for s in range(m))
    edge = sum(T[y[s]][y[s + 1]] for s in range(m - 1))
    return node + edge


def decode_bruteforce(X: List[List[float]], W: List[List[float]], T: List[List[float]]) -> DecodeResult:
    """O(|Y|^m) decoder by exhaustive enumeration (for tiny test cases)."""
    m = len(X)
    C = len(W)

    best_y: Tuple[int, ...] | None = None
    best_score = float("-inf")

    for y in product(range(C), repeat=m):
        cur = score_sequence(y, X, W, T)
        if cur > best_score:
            best_score = cur
            best_y = y

    assert best_y is not None
    return DecodeResult(labels=list(best_y), score=best_score)


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


def parse_decode_input(
    path: str | Path,
    m: int = 100,
    C: int = 26,
    d: int = 128,
    transpose_t: bool = False,
) -> Tuple[List[List[float]], List[List[float]], List[List[float]]]:
    """Parse data/decode_input.txt style flat vector into X, W, and T.

    Expected layout (all in one column/flat list):
      x_1..x_m, w_1..w_C, then T entries of a CxC matrix.
    """
    values = [float(tok) for tok in Path(path).read_text().split()]

    x_count = m * d
    w_count = C * d
    t_count = C * C
    expected = x_count + w_count + t_count

    if len(values) != expected:
        raise ValueError(f"Expected {expected} values, got {len(values)} from {path}")

    xs = values[:x_count]
    ws = values[x_count : x_count + w_count]
    ts = values[x_count + w_count :]

    X = [xs[s * d : (s + 1) * d] for s in range(m)]
    W = [ws[c * d : (c + 1) * d] for c in range(C)]
    T = [ts[r * C : (r + 1) * C] for r in range(C)]

    if transpose_t:
        T = [[T[r][c] for r in range(C)] for c in range(C)]

    return X, W, T


def write_labels_one_indexed(path: str | Path, labels_zero_indexed: Sequence[int]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(str(y + 1) for y in labels_zero_indexed) + "\n")


def decode_file(
    input_path: str | Path = "data/decode_input.txt",
    output_path: str | Path = "result/decode_output.txt",
    transpose_t: bool = False,
) -> DecodeResult:
    """Decode assignment test input and write 1..26 labels to output file."""
    X, W, T = parse_decode_input(input_path, transpose_t=transpose_t)
    result = decode_viterbi(X, W, T)
    write_labels_one_indexed(output_path, result.labels)
    return result


result = decode_file("data/decode_input.txt", "result/decode_output.txt", transpose_t=False)
print(f"Decoded {len(result.labels)} labels.")
print(f"Objective score: {result.score}")

