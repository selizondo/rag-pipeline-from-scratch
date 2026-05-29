"""
seed_corpus.py — Download AI/ML Q&A datasets and write them as markdown
files ready for ingest.py to chunk and embed.

Sources:
  1. Shlok307/Interview_questions (HuggingFace) — 1,843 AI-domain Q&A pairs
  2. mjphayes/machine_learning_questions (HuggingFace) — 636 ML interview Q&A
  3. lorenzoscaturchio/ml-interview-qa (Kaggle) — 502 ML Q&A with difficulty + company tags

Requirements:
  pip install datasets kaggle
  Kaggle credentials: ~/.kaggle/kaggle.json

Usage:
    python seed_corpus.py                   # write to ./corpus/
    python seed_corpus.py --out ./corpus
    python seed_corpus.py --skip-kaggle     # HuggingFace only
"""

import argparse
import csv
import os
import shutil
import tempfile
import textwrap

# ---------------------------------------------------------------------------
# HuggingFace: Shlok307/Interview_questions (AI domain only)
# ---------------------------------------------------------------------------

def fetch_shlok307() -> list[dict]:
    from datasets import load_dataset
    print("Downloading Shlok307/Interview_questions...")
    ds = load_dataset("Shlok307/Interview_questions", split="train")
    rows = []
    for row in ds:
        if row.get("domain", "").strip().lower() != "ai":
            continue
        q = (row.get("question") or "").strip()
        a = (row.get("answer") or "").strip()
        if q and len(a) >= 40:
            rows.append({"question": q, "answer": a})
    print(f"  {len(rows)} AI-domain Q&A pairs")
    return rows


def write_shlok307(rows: list[dict], out_dir: str):
    fpath = os.path.join(out_dir, "ai_interview_qa_shlok.md")
    lines = [
        "# AI/ML Interview Questions & Answers\n",
        "_Source: Shlok307/Interview_questions (HuggingFace, AI domain)_\n\n---\n",
    ]
    for i, row in enumerate(rows, 1):
        lines.append(f"## Q{i}: {row['question']}\n")
        lines.append(textwrap.fill(row["answer"], width=100) + "\n\n---\n")
    with open(fpath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Wrote {fpath} ({len(rows)} Q&A)")


# ---------------------------------------------------------------------------
# HuggingFace: mjphayes/machine_learning_questions
# ---------------------------------------------------------------------------

_ML_KEYWORDS = {
    "machine learning", "deep learning", "neural network", "model", "training",
    "overfitting", "regularization", "gradient", "backprop", "transformer",
    "attention", "embedding", "classification", "regression", "clustering",
    "random forest", "svm", "feature", "loss function", "optimizer", "epoch",
    "batch", "cross-validation", "bias", "variance", "precision", "recall",
    "f1", "roc", "llm", "rag", "fine-tun", "lora", "bert", "gpt",
    "reinforcement", "convolutional", "recurrent", "lstm", "encoder", "decoder",
    "tokeniz", "natural language", "nlp", "data augmentation", "dropout",
    "activation", "softmax", "relu", "normalization", "hyperparameter",
}


def fetch_mjphayes() -> list[dict]:
    from datasets import load_dataset
    print("Downloading mjphayes/machine_learning_questions...")
    ds = load_dataset("mjphayes/machine_learning_questions", split="train+test")
    rows = []
    for row in ds:
        q = (row.get("question") or "").strip()
        a = (row.get("answer") or "").strip()
        if not q or len(a) < 40:
            continue
        combined = (q + " " + a).lower()
        if not any(kw in combined for kw in _ML_KEYWORDS):
            continue
        rows.append({"question": q, "answer": a})
    print(f"  {len(rows)} ML-filtered Q&A pairs")
    return rows


def write_mjphayes(rows: list[dict], out_dir: str):
    fpath = os.path.join(out_dir, "ml_interview_qa_mjphayes.md")
    lines = [
        "# Machine Learning Interview Questions & Answers\n",
        "_Source: mjphayes/machine_learning_questions (HuggingFace)_\n\n---\n",
    ]
    for i, row in enumerate(rows, 1):
        lines.append(f"## Q{i}: {row['question']}\n")
        lines.append(textwrap.fill(row["answer"], width=100) + "\n\n---\n")
    with open(fpath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Wrote {fpath} ({len(rows)} Q&A)")


# ---------------------------------------------------------------------------
# Kaggle: lorenzoscaturchio/ml-interview-qa
# ---------------------------------------------------------------------------

KAGGLE_DATASET = "lorenzoscaturchio/ml-interview-qa"
KAGGLE_CSV = "ml_interview_questions.csv"


def fetch_kaggle() -> list[dict]:
    try:
        from kaggle import api
        api.authenticate()
    except ImportError:
        print("  SKIP: kaggle package not installed (pip install kaggle)")
        return []
    except Exception as e:
        print(f"  SKIP: Kaggle auth failed — {e}")
        return []

    print(f"Downloading {KAGGLE_DATASET} from Kaggle...")

    tmpdir = tempfile.mkdtemp()
    try:
        api.dataset_download_files(KAGGLE_DATASET, path=tmpdir, unzip=True, quiet=True)
        csv_path = os.path.join(tmpdir, KAGGLE_CSV)
        if not os.path.exists(csv_path):
            # Find any CSV in the download
            csvs = [f for f in os.listdir(tmpdir) if f.endswith(".csv")]
            if not csvs:
                print("  SKIP: no CSV found in Kaggle download")
                return []
            csv_path = os.path.join(tmpdir, csvs[0])

        rows = []
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                q = (row.get("question") or "").strip()
                a = (row.get("answer") or "").strip()
                category = (row.get("category") or "").strip()
                difficulty = (row.get("difficulty") or "").strip()
                tags = (row.get("topic_tags") or "").strip()
                if q and len(a) >= 40:
                    rows.append({
                        "question": q,
                        "answer": a,
                        "category": category,
                        "difficulty": difficulty,
                        "tags": tags,
                    })
        print(f"  {len(rows)} Q&A pairs from Kaggle")
        return rows
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def write_kaggle(rows: list[dict], out_dir: str):
    fpath = os.path.join(out_dir, "ml_interview_qa_kaggle.md")
    lines = [
        "# ML Interview Questions & Answers (with Difficulty & Company Tags)\n",
        "_Source: lorenzoscaturchio/ml-interview-qa (Kaggle)_\n\n---\n",
    ]
    for i, row in enumerate(rows, 1):
        meta = f"**Category:** {row['category']}  |  **Difficulty:** {row['difficulty']}"
        if row["tags"]:
            meta += f"  |  **Topics:** {row['tags'].replace('|', ', ')}"
        lines.append(f"## Q{i}: {row['question']}\n")
        lines.append(f"{meta}\n")
        lines.append(textwrap.fill(row["answer"], width=100) + "\n\n---\n")
    with open(fpath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Wrote {fpath} ({len(rows)} Q&A)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Seed RAG corpus from HuggingFace + Kaggle ML datasets"
    )
    parser.add_argument("--out", default="./corpus",
                        help="Output directory for .md files (default: ./corpus)")
    parser.add_argument("--skip-kaggle", action="store_true",
                        help="Skip Kaggle download (no credentials needed)")
    parser.add_argument("--skip-hf", action="store_true",
                        help="Skip HuggingFace downloads")
    args = parser.parse_args()

    try:
        import datasets  # noqa: F401
    except ImportError:
        print("ERROR: pip install datasets")
        return

    os.makedirs(args.out, exist_ok=True)

    if not args.skip_hf:
        write_shlok307(fetch_shlok307(), args.out)
        write_mjphayes(fetch_mjphayes(), args.out)

    if not args.skip_kaggle:
        kaggle_rows = fetch_kaggle()
        if kaggle_rows:
            write_kaggle(kaggle_rows, args.out)
    else:
        print("Skipping Kaggle (--skip-kaggle)")

    md_files = [f for f in os.listdir(args.out) if f.endswith(".md")]
    total_size = sum(
        os.path.getsize(os.path.join(args.out, f)) for f in md_files
    )
    print(f"\nCorpus ready: {len(md_files)} .md files  |  {total_size // 1024} KB  →  {args.out}/")
    print("\nNext — ingest into Chroma:")
    print(f"  python ingest.py --corpus {args.out} --chunk-size 256 --overlap 32")


if __name__ == "__main__":
    main()
