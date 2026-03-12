#!/usr/bin/env python
"""Convert SVMhmm-style OCR data (train_struct.txt / test_struct.txt) to plain LibSVM format.

Input line format (per data/fields_struct.txt):
  <label> qid:<word_id> <feat_id>:<value> <feat_id>:<value> ...

LibSVM classification format does NOT accept the qid field, so we drop it and keep only
  <label> <feat_id>:<value> ...

Optionally, we could also add the constant bias feature described in fields_struct.txt (feature 129 = 1).
"""

from __future__ import annotations

import argparse
from pathlib import Path


def convert_struct_to_libsvm(in_path: str | Path, out_path: str | Path, *, add_bias: bool = True, bias_index: int = 129) -> None:
    in_path = Path(in_path)
    out_path = Path(out_path)

    out_lines: list[str] = []

    for raw in in_path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 1:
            continue

        label = parts[0]
        feats: list[tuple[int, float]] = []

        for tok in parts[1:]:
            if tok.startswith("qid:"):
                continue
            if ":" not in tok:
                continue
            k, v = tok.split(":", 1)
            try:
                idx = int(k)
                val = float(v)
            except ValueError:
                # skip malformed tokens
                continue
            feats.append((idx, val))

        if add_bias:
            feats.append((bias_index, 1.0))

        feats.sort(key=lambda kv: kv[0])
        feat_str = " ".join(f"{i}:{v:g}" for i, v in feats)
        out_lines.append(f"{label} {feat_str}".rstrip())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert train_struct/test_struct to LibSVM format (drop qid; optional bias feature).")
    ap.add_argument("--in", dest="in_path", required=True, help="Input path (e.g., data/train_struct.txt)")
    ap.add_argument("--out", dest="out_path", required=True, help="Output path (e.g., data/train_libsvm.txt)")
    ap.add_argument("--no-bias", action="store_true", help="Do NOT add the constant 129:1 feature")
    ap.add_argument("--bias-index", type=int, default=129, help="Bias feature index to add (default: 129)")
    args = ap.parse_args()

    convert_struct_to_libsvm(args.in_path, args.out_path, add_bias=(not args.no_bias), bias_index=args.bias_index)


if __name__ == "__main__":
    main()
