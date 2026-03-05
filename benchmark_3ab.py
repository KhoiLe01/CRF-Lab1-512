#!/usr/bin/env python
# coding: utf-8
"""
- CRF reads data/train.txt + data/test.txt (OCR rows with 128-d float features).
- SVM-MC reads data/train_libsvm.txt + data/test_libsvm.txt (LibSVM sparse format).
- SVM-Struct-SVMHMM still requires svm_hmm_learn/svm_hmm_classify binaries (per lab).
"""

import argparse
import csv
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Iterable


ALPHABET = "abcdefghijklmnopqrstuvwxyz"


@dataclass
class OCRRow:
    row_index: int
    letter: Optional[str]
    word_id: int
    position: int
    x: List[float]


def parse_ocr_file(path: str | Path, require_labels: bool, d: int = 128) -> List[OCRRow]:
    rows: List[OCRRow] = []
    for idx, line in enumerate(Path(path).read_text().splitlines()):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5 + d:
            raise ValueError(f"Malformed row in {path}: expected at least {5+d} tokens")

        letter_raw = parts[1].lower()
        letter = letter_raw if letter_raw in ALPHABET else None
        if require_labels and letter is None:
            raise ValueError(f"Missing valid label in row: {line}")

        rows.append(
            OCRRow(
                row_index=idx,
                letter=letter,
                word_id=int(parts[3]),
                position=int(parts[4]),
                x=[float(v) for v in parts[5 : 5 + d]],
            )
        )
    return rows


def compute_letter_word_accuracy(test_rows: Sequence[OCRRow], pred_labels_1idx: Sequence[int]) -> Tuple[float, float]:
    if len(test_rows) != len(pred_labels_1idx):
        raise ValueError(f"Prediction length ({len(pred_labels_1idx)}) must match number of test rows ({len(test_rows)})")

    total = len(test_rows)
    correct = 0

    by_word_truth: Dict[int, List[Tuple[int, int]]] = {}
    by_word_pred: Dict[int, List[Tuple[int, int]]] = {}

    for row, p in zip(test_rows, pred_labels_1idx):
        if row.letter is None:
            raise ValueError("Word accuracy requires labels in test rows")
        t = ALPHABET.index(row.letter) + 1
        if t == p:
            correct += 1

        by_word_truth.setdefault(row.word_id, []).append((row.position, t))
        by_word_pred.setdefault(row.word_id, []).append((row.position, p))

    letter_acc = correct / total if total else 0.0

    word_correct = 0
    for wid in by_word_truth:
        truth = [v for _, v in sorted(by_word_truth[wid], key=lambda z: z[0])]
        pred = [v for _, v in sorted(by_word_pred[wid], key=lambda z: z[0])]
        if truth == pred:
            word_correct += 1

    word_acc = word_correct / len(by_word_truth) if by_word_truth else 0.0
    return letter_acc, word_acc


def predict_crf_rows(train_rows: Sequence[OCRRow], test_rows: Sequence[OCRRow], c_value: float, maxfun: int) -> List[int]:
    from optimizer import rows_to_word_samples  # local import to keep coupling explicit

    train_words = rows_to_word_samples(train_rows)
    theta, _, _ = train_with_fmin_tnc(train_words, c_reg=c_value, maxfun=maxfun)
    w, t = unpack_params(theta)

    grouped: Dict[int, List[OCRRow]] = {}
    for r in test_rows:
        grouped.setdefault(r.word_id, []).append(r)

    pred_by_row: Dict[int, int] = {}
    for _, group in sorted(grouped.items(), key=lambda kv: kv[0]):
        ordered = sorted(group, key=lambda r: r.position)
        x_seq = [r.x for r in ordered]
        res = decode_viterbi(x_seq, w, t)
        for r, y0 in zip(ordered, res.labels):
            pred_by_row[r.row_index] = y0 + 1  # decode_viterbi returns 0-based labels

    return [pred_by_row[r.row_index] for r in sorted(test_rows, key=lambda r: r.row_index)]


# ---------- External command helpers (for SVMhmm baseline) ----------

def run_external_command(command: str) -> None:
    proc = subprocess.run(command, shell=True, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {command}\n{proc.stdout}\n{proc.stderr}")


def parse_prediction_file(path: str | Path) -> List[int]:
    return [int(x.strip()) for x in Path(path).read_text().splitlines() if x.strip()]


def load_libsvm_file(path: str | Path) -> Tuple[List[int], List[Dict[int, float]]]:
    """
    LibSVM-format loader:
      <label> <idx>:<val> <idx>:<val> ...
    Returns:
      y: list[int]
      X: list[dict[int,float]] with 1-based indices
    """
    y: List[int] = []
    X: List[Dict[int, float]] = []
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        y.append(int(float(parts[0])))
        feats: Dict[int, float] = {}
        for tok in parts[1:]:
            if ":" not in tok:
                continue
            k, v = tok.split(":", 1)
            feats[int(k)] = float(v)
        X.append(feats)
    return y, X


def predict_svm_mc_python_from_libsvm(
    train_libsvm_path: str | Path,
    test_libsvm_path: str | Path,
    c_value: float,
) -> List[int]:
    """
    Train/predict multi-class linear SVM on individual letters
    Returns predictions aligned with the order of lines in test_libsvm_path, in labels 1..26.
    """
    y_train, X_train = load_libsvm_file(train_libsvm_path)
    y_test, X_test = load_libsvm_file(test_libsvm_path)

    try:
        from liblinear.liblinearutil import train as ll_train, predict as ll_predict
    except Exception:
        from liblinearutil import train as ll_train, predict as ll_predict  # type: ignore

    model = ll_train(y_train, X_train, f"-c {c_value} -q")
    p_labels, _, _ = ll_predict(y_test, X_test, model, "-q")
    
    return [int(v) for v in p_labels]


def maybe_plot_curves(out_png: Path, title: str, ylabel: str, c_values: Sequence[float], y_values: Sequence[float]) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    plt.figure(figsize=(6, 4))
    plt.plot(c_values, y_values, marker="o")
    plt.xscale("log")
    plt.xlabel("C")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    return True


def write_observation(path: Path, model: str, c_values: Sequence[float], letter: Sequence[float], word: Sequence[float]) -> None:
    best_letter_idx = max(range(len(letter)), key=lambda i: letter[i])
    best_word_idx = max(range(len(word)), key=lambda i: word[i])

    text = (
        f"Model: {model}\n"
        f"Best letter-accuracy C: {c_values[best_letter_idx]} (acc={letter[best_letter_idx]:.4f})\n"
        f"Best word-accuracy C: {c_values[best_word_idx]} (acc={word[best_word_idx]:.4f})\n"
        "Typical pattern: very small C underfits, while very large C may overfit.\n"
        "Word accuracy is usually lower and more sensitive than letter accuracy because one wrong letter makes the entire word wrong.\n"
    )
    path.write_text(text)


def evaluate(
    model: str,
    c_values: Sequence[float],
    train_rows: Sequence[OCRRow],
    test_rows: Sequence[OCRRow],
    out_dir: Path,
    crf_maxfun: int,
    svm_mc_train_libsvm: Optional[Path],
    svm_mc_test_libsvm: Optional[Path],
    train_struct_path: Path,
    test_struct_path: Path,
    svm_hmm_dir: Path,
) -> None:
    results_letter: List[Tuple[float, float]] = []
    results_word: List[Tuple[float, float]] = []
    
    for c in c_values:
        if model == "CRF":                        
            from decoder import decode_viterbi
            from optimizer import train_with_fmin_tnc, unpack_params
            yhat = predict_crf_rows(train_rows, test_rows, c, maxfun=crf_maxfun)

        elif model == "SVM_MC":            
            print(svm_hmm_dir)
            if not svm_mc_train_libsvm or not svm_mc_test_libsvm:
                raise RuntimeError("SVM_MC python mode requires --svm-mc-train-libsvm and --svm-mc-test-libsvm")
            yhat = predict_svm_mc_python_from_libsvm(svm_mc_train_libsvm, svm_mc_test_libsvm, c)

            # sanity: ensure length matches test_rows for accuracy computation
            if len(yhat) != len(test_rows):
                raise RuntimeError(
                    f"SVM_MC predictions length ({len(yhat)}) != test rows ({len(test_rows)}). "
                    "Make sure test_libsvm corresponds line-by-line to data/test.txt."
                )

        elif model == "SVM_Struct_SVMHMM":
            learn_exe = svm_hmm_dir / "svm_hmm_learn.exe"
            classify_exe = svm_hmm_dir / "svm_hmm_classify.exe"

            model_path = out_dir / f"svm_struct_model_C{c}.txt"
            pred_path  = out_dir / "svm_struct_prediction.txt"

            cmd_learn = f'"{learn_exe}" -c {c} "{train_struct_path}" "{model_path}"'
            cmd_pred  = f'"{classify_exe}" "{test_struct_path}" "{model_path}" "{pred_path}"'

            print(cmd_learn)
            print(cmd_pred)

            run_external_command(cmd_learn)
            run_external_command(cmd_pred)

            yhat = parse_prediction_file(pred_path)

        else:
            raise ValueError(f"Unknown model: {model}")

        letter_acc, word_acc = compute_letter_word_accuracy(test_rows, yhat)
        results_letter.append((c, letter_acc))
        results_word.append((c, word_acc))

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / f"{model.lower()}_metrics.csv"
    with metrics_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["C", "letter_acc", "word_acc"])
        for (c, la), (_, wa) in zip(results_letter, results_word):
            writer.writerow([c, la, wa])

    c_arr = [c for c, _ in results_letter]
    letter_arr = [v for _, v in results_letter]
    word_arr = [v for _, v in results_word]

    maybe_plot_curves(out_dir / f"{model.lower()}_letter_acc.png", f"{model} letter accuracy", "letter accuracy", c_arr, letter_arr)
    maybe_plot_curves(out_dir / f"{model.lower()}_word_acc.png", f"{model} word accuracy", "word accuracy", c_arr, word_arr)

    write_observation(out_dir / f"{model.lower()}_observation.txt", model, c_arr, letter_arr, word_arr)


import argparse
from pathlib import Path

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Benchmark CRF/SVM models for Q3(a)/(b).")
    p.add_argument("--train", default="data/train.txt")
    p.add_argument("--test", default="data/test.txt")

    p.add_argument("--train-struct", dest="train_struct", default="data/train_struct.txt")
    p.add_argument("--test-struct",  dest="test_struct",  default="data/test_struct.txt")

    p.add_argument("--svm-mc-train-libsvm", "--train-libsvm",
                   dest="svm_mc_train_libsvm", default="data/train_libsvm.txt")
    p.add_argument("--svm-mc-test-libsvm", "--test-libsvm",
                   dest="svm_mc_test_libsvm", default="data/test_libsvm.txt")

    p.add_argument("--svm-hmm-dir", default="svm_hmm")

    p.add_argument("--models", default="CRF,SVM_MC,SVM_Struct_SVMHMM")
    p.add_argument("--c-values", dest="c_values", default="1,10,100,1000")
    p.add_argument("--out-dir", default="result/benchmark")
    p.add_argument("--crf-maxfun", type=int, default=200)
    p.add_argument("--self-test", action="store_true")
    return p

def main() -> None:
    args = build_arg_parser().parse_args()

    c_values = [float(x.strip()) for x in args.c_values.split(",") if x.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    train_rows = parse_ocr_file(args.train, require_labels=True)
    test_rows  = parse_ocr_file(args.test,  require_labels=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_struct_path=Path(args.train_struct)
    test_struct_path=Path(args.test_struct)
    svm_hmm_dir=Path(args.svm_hmm_dir)
    
    for model in models:
        evaluate(
            model=model,
            c_values=c_values,
            train_rows=train_rows,
            test_rows=test_rows,
            out_dir=out_dir,
            crf_maxfun=args.crf_maxfun,
            svm_mc_train_libsvm=Path(args.svm_mc_train_libsvm),
            svm_mc_test_libsvm=Path(args.svm_mc_test_libsvm),
            train_struct_path=train_struct_path,
            test_struct_path=test_struct_path,
            svm_hmm_dir=svm_hmm_dir
        )

    print(f"Benchmark finished. Results saved in: {out_dir}")

if __name__ == "__main__":
    main()

'''    
'''