from datasets import Dataset

def load_tsv(path):
    data = []
    tokens, labels = [], []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    data.append({
                        "tokens": tokens,
                        "labels": labels
                    })
                    tokens, labels = [], []
            else:
                t, l = line.split("\t")
                tokens.append(t)
                labels.append(l)

    if tokens:
        data.append({"tokens": tokens, "labels": labels})

    return Dataset.from_list(data)
