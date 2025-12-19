import random
from pathlib import Path

RAW_DIR = Path("../data/acter/hrtfl/without_named_entities")
OUT_DIR = Path("../data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def read_file(path):
    sentences = []
    tokens, labels = [], []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            if tokens:
                sentences.append((tokens, labels))
                tokens, labels = [], []
        else:
            tok, lab = line.split("\t")
            tokens.append(tok)
            labels.append(lab)

    if tokens:
        sentences.append((tokens, labels))

    return sentences


def write_tsv(path, documents):
    with open(path, "w", encoding="utf-8") as f:
        for doc in documents:
            for tokens, labels in doc:
                for t, l in zip(tokens, labels):
                    f.write(f"{t}\t{l}\n")
                f.write("\n")


if __name__ == "__main__":
    files = sorted(RAW_DIR.glob("*.tsv"))

    random.seed(42)
    random.shuffle(files)

    n = len(files)
    train_files = files[:int(0.7*n)]
    val_files   = files[int(0.7*n):int(0.85*n)]
    test_files  = files[int(0.85*n):]

    train_docs = [read_file(f) for f in train_files]
    val_docs   = [read_file(f) for f in val_files]
    test_docs  = [read_file(f) for f in test_files]

    write_tsv(OUT_DIR/"train.tsv", train_docs)
    write_tsv(OUT_DIR/"val.tsv", val_docs)
    write_tsv(OUT_DIR/"test.tsv", test_docs)
