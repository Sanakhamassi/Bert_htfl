import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# =========================
# CONFIG
# =========================
MODEL_DIR = "/Users/mandamac1/Desktop/FineTuneBert/model_checkpoints/biobert_htfl/checkpoint-618"
MAX_LENGTH = 512

# =========================
# LOAD MODEL + TOKENIZER
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)

model.eval()

# =========================
# TERM EXTRACTION LOGIC
# =========================
def extract_terms(word_preds):
    terms = []
    current = []

    for word, label in word_preds:
        if label == "I":
            current.append(word)
        else:
            if current:
                terms.append(" ".join(current))
                current = []

    if current:
        terms.append(" ".join(current))

    return terms


def predict_terms(text):
    words = text.split()

    encoding = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH
    )

    with torch.no_grad():
        outputs = model(**encoding)

    preds = torch.argmax(outputs.logits, dim=-1)[0].tolist()
    word_ids = encoding.word_ids()

    word_preds = []
    prev_word_id = None

    for pred, word_id in zip(preds, word_ids):
        if word_id is None:
            continue
        if word_id != prev_word_id:
            label = model.config.id2label[pred]
            word_preds.append((words[word_id], label))
        prev_word_id = word_id

    return extract_terms(word_preds)


# =========================
# RUN EXAMPLE
# =========================
if __name__ == "__main__":
    text = (
        "Heart failure (HF), as defined by the American College of Cardiology (ACC) and the American Heart Association (AHA), is a complex clinical syndrome that results from any structural or functional impairment of ventricular filling or ejection of blood. HF is a common disorder worldwide with a high morbidity and mortality rate. With an estimated prevalence of 26 million people worldwide, CHF contributes to increased healthcare costs, reduces functional capacity, and significantly affects quality of life. Accurately diagnosing and effectively treating the disease is essential to prevent recurrent hospitalizations, decrease morbidity and mortality, and enhance patient outcomes.[1] "
"The etiology of HF is variable and extensive. Ischemic heart disease is the leading cause of HF. The general management of HF aims to relieve systemic and pulmonary congestion and stabilize hemodynamic status, regardless of the cause. The treatment of HF requires a multifaceted approach involving patient education, optimal medication administration, and decreasing acute exacerbations. Per the recent ACC/AHA guidelines for HF 2022, patients with HF are classified based on left ventricle ejection fraction (LVEF), whereas clinical and laboratory parameters are integrated to stage patients. The New York Heart Association (NYHA) classification stratifies and defines the functional capacity and severity of HF symptoms. This system is subjectively determined by clinicians and is widely used in clinical practice to direct therapy. Management of patients depends on the classification and staging of the disease"
    )

    terms = predict_terms(text)

    print("\nInput text:")
    print(text)

    print("\nExtracted terms:")
    for t in terms:
        print(f"- {t}")
