from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SHAH_DIR = ROOT / "code" / "shah"
RESULTS_DIR = ROOT / "code" / "angelo" / "result"
TRAIN_PATH = ROOT / "data" / "train.txt"
TEST_PATH = ROOT / "data" / "test.txt"
TRANSFORM_PATH = ROOT / "data" / "transform.txt"


if str(SHAH_DIR) not in sys.path:
    sys.path.insert(0, str(SHAH_DIR))

from benchmark_3ab import ALPHABET, compute_letter_word_accuracy, parse_ocr_file


def load_q2b_module():
    module_path = ROOT / "code" / "angelo" / "2b.py"
    spec = importlib.util.spec_from_file_location("angelo_q2b", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load CRF module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


Q2B = load_q2b_module()


@dataclass(frozen=True)
class TransformOp:
    kind: str
    word_id: int
    values: Tuple[float, ...]


def parse_transform_file(path: Path) -> List[TransformOp]:
    ops: List[TransformOp] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        kind = parts[0]
        if kind == "r" and len(parts) == 3:
            ops.append(TransformOp(kind="r", word_id=int(parts[1]), values=(float(parts[2]),)))
        elif kind == "t" and len(parts) == 4:
            ops.append(TransformOp(kind="t", word_id=int(parts[1]), values=(float(parts[2]), float(parts[3]))))
        else:
            raise ValueError(f"Malformed transform line: {line}")
    return ops


def vector_to_image(vec: Sequence[float]) -> np.ndarray:
    return np.asarray(vec, dtype=np.float32).reshape((8, 16), order="F")


def image_to_vector(img: np.ndarray) -> List[float]:
    return img.reshape(128, order="F").astype(np.float64).tolist()


def apply_transform_to_vector(vec: Sequence[float], op: TransformOp) -> List[float]:
    img = vector_to_image(vec)
    height, width = img.shape
    if op.kind == "r":
        degree = op.values[0]
        matrix = cv2.getRotationMatrix2D(center=(width / 2.0, height / 2.0), angle=degree, scale=1.0)
    elif op.kind == "t":
        dx, dy = op.values
        matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    else:
        raise ValueError(f"Unsupported transform kind: {op.kind}")

    warped = cv2.warpAffine(
        img,
        matrix,
        dsize=(width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    return image_to_vector(warped)


def clone_rows(rows) -> List:
    return [
        type(row)(
            row_index=row.row_index,
            letter=row.letter,
            word_id=row.word_id,
            position=row.position,
            x=list(row.x),
        )
        for row in rows
    ]


def apply_training_distortions(train_rows: Sequence, ops: Sequence[TransformOp]) -> List:
    transformed = clone_rows(train_rows)
    op_by_word: Dict[int, TransformOp] = {op.word_id: op for op in ops}
    for row in transformed:
        op = op_by_word.get(row.word_id)
        if op is not None:
            row.x = apply_transform_to_vector(row.x, op)
    return transformed


def rows_to_q2_words(rows: Sequence) -> List[dict]:
    grouped: Dict[int, List] = {}
    for row in rows:
        grouped.setdefault(row.word_id, []).append(row)

    words: List[dict] = []
    for _, group in sorted(grouped.items(), key=lambda item: item[0]):
        ordered = sorted(group, key=lambda row: row.position)
        words.append(
            {
                "X": np.asarray([row.x for row in ordered], dtype=np.float64),
                "y": np.asarray([ALPHABET.index(row.letter) for row in ordered], dtype=int),
            }
        )
    return words


def train_crf_and_predict(train_rows: Sequence, test_rows: Sequence, c_value: float, maxfun: int) -> List[int]:
    train_words = rows_to_q2_words(train_rows)
    theta0 = np.zeros(26 * 128 + 26 * 26, dtype=np.float64)
    theta_opt, _, _ = Q2B.fmin_tnc(
        func=lambda th: Q2B.objective_and_gradient(th, train_words, C=c_value),
        x0=theta0,
        bounds=None,
        messages=0,
        maxfun=maxfun,
    )
    w_opt, t_opt = Q2B.unpack_params(theta_opt)

    grouped: Dict[int, List] = {}
    for row in test_rows:
        grouped.setdefault(row.word_id, []).append(row)

    pred_by_row: Dict[int, int] = {}
    for _, group in sorted(grouped.items(), key=lambda item: item[0]):
        ordered = sorted(group, key=lambda row: row.position)
        x_seq = np.asarray([row.x for row in ordered], dtype=np.float64)
        labels, _ = Q2B.max_sum_decoder(x_seq, w_opt, t_opt)
        for row, y0 in zip(ordered, labels):
            pred_by_row[row.row_index] = y0 + 1

    return [pred_by_row[row.row_index] for row in sorted(test_rows, key=lambda row: row.row_index)]


def rows_to_svm_dataset(rows: Sequence) -> Tuple[List[int], List[Dict[int, float]]]:
    labels: List[int] = []
    features: List[Dict[int, float]] = []
    for row in rows:
        if row.letter is None:
            raise ValueError("SVM-MC training requires labels")
        labels.append(ALPHABET.index(row.letter) + 1)
        feats = {idx + 1: float(value) for idx, value in enumerate(row.x) if value != 0.0}
        features.append(feats)
    return labels, features


def train_svm_mc_and_predict(train_rows: Sequence, test_rows: Sequence, c_value: float) -> List[int]:
    try:
        from liblinear.liblinearutil import predict as ll_predict, train as ll_train
    except Exception:
        from liblinearutil import predict as ll_predict, train as ll_train  # type: ignore

    y_train, x_train = rows_to_svm_dataset(train_rows)
    y_test, x_test = rows_to_svm_dataset(test_rows)
    model = ll_train(y_train, x_train, f"-c {c_value} -q")
    preds, _, _ = ll_predict(y_test, x_test, model, "-q")
    return [int(value) for value in preds]


def plot_accuracy_curves(
    path: Path,
    title: str,
    ylabel: str,
    x_values: Sequence[int],
    crf_values: Sequence[float],
    svm_values: Sequence[float],
) -> None:
    plt.figure(figsize=(7, 4.5))
    plt.plot(x_values, crf_values, marker="o", linewidth=2, label="CRF")
    plt.plot(x_values, svm_values, marker="s", linewidth=2, label="SVM-MC")
    plt.xlabel("Number of applied transformations")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def write_summary(
    path: Path,
    transform_counts: Sequence[int],
    metrics: Sequence[dict],
    crf_c: float,
    svm_c_letter: float,
    svm_c_word: float,
    crf_maxfun: int,
) -> None:
    lines = [
        "# Question 5 Summary",
        "",
        f"- Chosen CRF C value: `{crf_c}`",
        f"- Chosen SVM-MC C value for Q5a: `{svm_c_letter}`",
        f"- Chosen SVM-MC C value for Q5b: `{svm_c_word}`",
        f"- CRF training maxfun used for this run: `{crf_maxfun}`",
        "",
        "## Accuracy Table",
        "",
        "| Transforms | CRF letter acc | SVM-MC letter acc | CRF word acc | SVM-MC word acc |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for x, row in zip(transform_counts, metrics):
        lines.append(
            f"| {x} | {row['crf_letter']:.6f} | {row['svm_letter']:.6f} | {row['crf_word']:.6f} | {row['svm_word']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Observations",
            "",
            "- CRF reuses sequence structure and is expected to retain higher word-wise accuracy as more training words are distorted.",
            "- SVM-MC classifies letters independently, so distortion in the training images tends to hurt word-wise accuracy more sharply.",
            "- The exact robustness trend is captured numerically in the table above and visually in the saved plots.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def print_repo_summary(crf_c: float, svm_c_letter: float, svm_c_word: float) -> None:
    print("Repository summary for Q5", flush=True)
    print("- CRF code reused from code/angelo/2b.py for training and decoding.", flush=True)
    print("- Benchmark helpers reused from code/shah/benchmark_3ab.py for OCR parsing and accuracy computation.", flush=True)
    print("- SVM-MC benchmark path in code/shah/benchmark_3ab.py uses LibLinear on LibSVM-style letter-level features.", flush=True)
    print(f"- Chosen C values: CRF={crf_c}, SVM-MC(Q5a)={svm_c_letter}, SVM-MC(Q5b)={svm_c_word}.", flush=True)
    print("- Plan: distort training data only, retrain both models for each requested x, evaluate on clean test data, then save plots and a markdown summary.", flush=True)


def run_experiment(
    transform_counts: Sequence[int],
    crf_c: float,
    svm_c_letter: float,
    svm_c_word: float,
    crf_maxfun: int,
) -> None:
    train_rows = parse_ocr_file(TRAIN_PATH, require_labels=True)
    test_rows = parse_ocr_file(TEST_PATH, require_labels=True)
    transform_ops = parse_transform_file(TRANSFORM_PATH)

    if max(transform_counts) > len(transform_ops):
        raise ValueError(f"Requested up to {max(transform_counts)} transforms, but only found {len(transform_ops)}")

    print_repo_summary(crf_c=crf_c, svm_c_letter=svm_c_letter, svm_c_word=svm_c_word)

    metrics: List[dict] = []
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for x in transform_counts:
        print(f"Starting x={x} with {x} applied training transformations", flush=True)
        distorted_train_rows = apply_training_distortions(train_rows, transform_ops[:x])
        print(f"Training CRF for x={x} with C={crf_c}", flush=True)
        crf_preds = train_crf_and_predict(distorted_train_rows, test_rows, c_value=crf_c, maxfun=crf_maxfun)
        print(f"Training SVM-MC for Q5a at x={x} with C={svm_c_letter}", flush=True)
        svm_preds_letter = train_svm_mc_and_predict(distorted_train_rows, test_rows, c_value=svm_c_letter)
        if svm_c_word == svm_c_letter:
            svm_preds_word = svm_preds_letter
        else:
            print(f"Training SVM-MC for Q5b at x={x} with C={svm_c_word}", flush=True)
            svm_preds_word = train_svm_mc_and_predict(distorted_train_rows, test_rows, c_value=svm_c_word)

        crf_letter, crf_word = compute_letter_word_accuracy(test_rows, crf_preds)
        svm_letter, _ = compute_letter_word_accuracy(test_rows, svm_preds_letter)
        _, svm_word = compute_letter_word_accuracy(test_rows, svm_preds_word)

        row = {
            "x": x,
            "crf_letter": crf_letter,
            "crf_word": crf_word,
            "svm_letter": svm_letter,
            "svm_word": svm_word,
        }
        metrics.append(row)
        print(
            f"x={x}: "
            f"CRF(letter={crf_letter:.4f}, word={crf_word:.4f}) "
            f"SVM-MC(letter={svm_letter:.4f}, word={svm_word:.4f})"
        , flush=True)

        completed_x = [row["x"] for row in metrics]
        plot_accuracy_curves(
            RESULTS_DIR / "q5_letter_accuracy.png",
            title="Q5a: Letter-wise Accuracy vs Training Distortion",
            ylabel="Letter-wise test accuracy",
            x_values=completed_x,
            crf_values=[row["crf_letter"] for row in metrics],
            svm_values=[row["svm_letter"] for row in metrics],
        )
        plot_accuracy_curves(
            RESULTS_DIR / "q5_word_accuracy.png",
            title="Q5b: Word-wise Accuracy vs Training Distortion",
            ylabel="Word-wise test accuracy",
            x_values=completed_x,
            crf_values=[row["crf_word"] for row in metrics],
            svm_values=[row["svm_word"] for row in metrics],
        )
        write_summary(
            RESULTS_DIR / "q5_summary.md",
            transform_counts=completed_x,
            metrics=metrics,
            crf_c=crf_c,
            svm_c_letter=svm_c_letter,
            svm_c_word=svm_c_word,
            crf_maxfun=crf_maxfun,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Question 5 robustness experiment.")
    parser.add_argument("--crf-c", type=float, default=1000.0)
    parser.add_argument("--svm-c-letter", type=float, default=1.0)
    parser.add_argument("--svm-c-word", type=float, default=1.0)
    parser.add_argument("--crf-maxfun", type=int, default=200)
    parser.add_argument("--transform-counts", default="0,500,1000,1500,2000")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    transform_counts = [int(value.strip()) for value in args.transform_counts.split(",") if value.strip()]
    run_experiment(
        transform_counts=transform_counts,
        crf_c=args.crf_c,
        svm_c_letter=args.svm_c_letter,
        svm_c_word=args.svm_c_word,
        crf_maxfun=args.crf_maxfun,
    )


if __name__ == "__main__":
    main()
