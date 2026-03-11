import numpy as np

def parse_ocr_file(filepath):
    words = []
    cur_y = []

    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            letter = parts[1]
            next_id = int(parts[2])

            cur_y.append(ord(letter) - ord("a"))

            if next_id == -1:
                words.append(np.array(cur_y, dtype=int))
                cur_y = []

    assert len(cur_y) == 0
    return words

def load_predictions(filepath):
    preds = np.loadtxt(filepath, dtype=int).reshape(-1)
    return preds - 1   # convert 1..26 to 0..25

def main():
    true_words = parse_ocr_file("../../data/test.txt")
    preds = load_predictions("result/prediction.txt")

    true_all = np.concatenate(true_words)
    assert len(preds) == len(true_all)

    letter_error = np.mean(preds != true_all)

    word_errors = []
    offset = 0
    for y_true in true_words:
        m = len(y_true)
        y_pred = preds[offset:offset + m]
        word_errors.append(np.any(y_pred != y_true))
        offset += m

    word_error = np.mean(word_errors)

    print("letter_error =", letter_error)
    print("word_error =", word_error)

if __name__ == "__main__":
    main()