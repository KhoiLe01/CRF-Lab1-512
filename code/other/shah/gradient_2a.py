from typing import List, Sequence

def gradient_w_c_single_word(
    x_seq: Sequence[Sequence[float]],
    y_seq: Sequence[int],
    p_y_equals_c_given_x: Sequence[float],
    c: int,
) -> List[float]:
    """Compute ∇_{w_c} log p(y|X) for one word.

    Implements Eq. (4):
      grad_w_c = sum_s ( I[y_s == c] - p(y_s == c | X) ) * x_s

    Notes:
    - y_seq is assumed 1-indexed labels (1..|Y|), to match assignment notation.
    - c should also be 1-indexed.
    - p_y_equals_c_given_x[s] is p(y_s = c | X) for position s.
    """
    m = len(x_seq)
    if len(y_seq) != m or len(p_y_equals_c_given_x) != m:
        raise ValueError("x_seq, y_seq, and p_y_equals_c_given_x must have equal length m")

    d = len(x_seq[0])
    if any(len(x) != d for x in x_seq):
        raise ValueError("All x_s vectors must have the same dimension")

    grad = [0.0] * d

    for s in range(m):
        indicator = 1.0 if y_seq[s] == c else 0.0
        coeff = indicator - p_y_equals_c_given_x[s]
        for k in range(d):
            grad[k] += coeff * x_seq[s][k]

    return grad


def average_gradient_w_c_dataset(
    x_words: Sequence[Sequence[Sequence[float]]],
    y_words: Sequence[Sequence[int]],
    p_words: Sequence[Sequence[float]],
    c: int,
) -> List[float]:
    """Average ∇_{w_c} log p(y^t|X^t) over dataset t=1..n for fixed class c."""
    n = len(x_words)
    if len(y_words) != n or len(p_words) != n:
        raise ValueError("x_words, y_words, p_words must have the same number of words")

    d = len(x_words[0][0])
    acc = [0.0] * d

    for t in range(n):
        g = gradient_w_c_single_word(x_words[t], y_words[t], p_words[t], c)
        for k in range(d):
            acc[k] += g[k]

    return [v / n for v in acc]


if __name__ == "__main__":
    # Tiny sanity check with hand-computable numbers
    x = [[1.0, 2.0], [3.0, 4.0]]
    y = [1, 2]
    c = 1
    p = [0.25, 0.10]  # p(y_1=1|X), p(y_2=1|X)

    # Expected:
    # s=1: (1 - 0.25)*[1,2] = [0.75,1.5]
    # s=2: (0 - 0.10)*[3,4] = [-0.3,-0.4]
    # total = [0.45,1.1]
    g = gradient_w_c_single_word(x, y, p, c)
    print("gradient:", g)
