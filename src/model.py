from transformers import AutoModelForTokenClassification
def load_model():
    return AutoModelForTokenClassification.from_pretrained(
        "dmis-lab/biobert-base-cased-v1.1",
        num_labels=2,
        # O: out of term, I: inside a term
        id2label={0: "O", 1: "I"},
        label2id={"O": 0, "I": 1}
    )
