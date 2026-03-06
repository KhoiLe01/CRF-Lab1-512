# returns objective value
def objective_value_function(X, W, T, y):
    m = len(X)
    
    # optional checks:
    # - len(y) = m
    # - W has 26 rows
    # - T is 26x26

    # node_score = sum_s dot(W[y[s]], X[s])
    node_score = 0.0
    for s in range(m):
        # current label at position s
        # get right W row
        # use dot with X[s]
        pass

    # edge_score = sum_s T[y[s], y[s+1]]
    edge_score = 0.0
    for s in range(m - 1):
        # add transition from y[s] to y[s+1]
        pass

    return node_score + edge_score


def main():
    print("Hello from crf-lab1-512!")


if __name__ == "__main__":
    main()
