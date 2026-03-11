import numpy as np

def viterbi_decoder(X, W, T):
    """
    X: Word image features, shape (m, 128)
    W: Node weights (letter-wise), shape (26, 128)
    T: Transition weights, shape (26, 26)
    """
    m = X.shape[0]  # Number of letters
    num_labels = 26
    
    node_potentials = np.dot(X, W.T) 
    
    L = np.zeros((m, num_labels))
    
    # Use to track best labels
    backpointers = np.zeros((m, num_labels), dtype=int)
    
    L[0, :] = 1
    
    # Compute best score
    for s in range(1, m):
        prev_node = node_potentials[s-1, :][:, np.newaxis]
        prev_L = L[s-1, :][:, np.newaxis]
        
        scores = prev_node + T + prev_L
        
        L[s, :] = np.max(scores, axis=0)
        backpointers[s, :] = np.argmax(scores, axis=0)
        
    # print(L)

    y_star = np.zeros(m, dtype=int)
    
    # Last letter
    final_scores = node_potentials[m-1, :] + L[m-1, :]
    y_star[m-1] = np.argmax(final_scores)
    max_objective_value = np.max(final_scores)
    
    # Trace back
    for s in range(m-2, -1, -1):
        y_star[s] = backpointers[s+1, y_star[s+1]]

    return y_star + 1, max_objective_value

def bruteforce(X, W, T):
    m = X.shape[0]
    num_labels = 26
    best_score = -np.inf
    best_sequence = None
    
    # Generate all possible sequences of length m (26^m possibilities).
    for i in range(26**m):
        sequence = []
        score = 0
        temp = i
        for s in range(m):
            label = temp % 26
            temp //= 26
            sequence.append(label)
            score += np.dot(W[label], X[s]) # Node potential
            if s > 0:
                score += T[sequence[-2], label] # Transition potential
        
        if score > best_score:
            best_score = score
            best_sequence = sequence
            
    return np.array(best_sequence) + 1, best_score

if __name__ == "__main__":
    with open('data/decode_input.txt', 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        X = []
        start = 0
        for i in range(0, 100):
            X.append([np.float64(x) for x in lines[128*i:128*i+128]])
            start += 128
        W = []
        newstart = start
        for i in range(0, 26):
            W.append([np.float64(x) for x in lines[start+128*i:start+128*i+128]])
            newstart += 128
        print(W[15][20])
        start = newstart
        T = np.zeros((26, 26), dtype=np.float64)
        for i in range(0, 26):
            for j in range(0, 26):
                T[i][j] = np.float64(lines[start + 26*i + j])
        print(len(T), len(T[0]))
        result, max_objective_value = viterbi_decoder(np.array(X, dtype=np.float64), np.array(W, dtype=np.float64), np.array(T, dtype=np.float64))
        print(result)
        print(max_objective_value)
        
        result_bf, max_objective_value_bf = bruteforce(np.array(X[:3], dtype=np.float64), np.array(W, dtype=np.float64), np.array(T, dtype=np.float64))
        print(result_bf)
        print(max_objective_value_bf)
    f.close()