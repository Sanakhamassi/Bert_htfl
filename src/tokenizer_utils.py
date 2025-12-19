from transformers import AutoTokenizer

MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"
# load the tokenizer related to biobert model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

label2id = {"O": 0, "I": 1}
id2label = {0: "O", 1: "I"}
# max sentence length
MAX_LENGTH = 512
# preprocess the data and tokenize it
def tokenize_and_align_labels(examples):
    tokenized = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True, # truncate terms if it exceeds 512
        max_length=MAX_LENGTH,   
        padding=False            # let DataCollator handle padding
    )

    labels_batch = []

    for i in range(len(examples["tokens"])):
        word_ids = tokenized.word_ids(batch_index=i)
        prev_word_id = None
        label_ids = []

        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != prev_word_id:
                label_ids.append(label2id[examples["labels"][i][word_id]])
            else:
                label_ids.append(-100)

            prev_word_id = word_id

        labels_batch.append(label_ids)

    tokenized["labels"] = labels_batch
    return tokenized
