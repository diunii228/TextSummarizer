import pandas as pd
from datasets import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
import evaluate
from src.utils import load_model_and_tokenizer

tokenizer, model = load_model_and_tokenizer("VietAI/vit5-base")

df = pd.read_csv('/Users/ddlyy/Documents/TextSummarizer/data/bio_medicine.csv', encoding='utf-8')
df = df.dropna(subset=['Document', 'Summary'])
print("Số dòng dữ liệu:", len(df))
print("Các cột:", df.columns.tolist())

print("5 mẫu dữ liệu đầu:")
print(df[['Document', 'Summary']].head(5))

df["doc_len"] = df["Document"].apply(lambda x: len(x.split()))
df["sum_len"] = df["Summary"].apply(lambda x: len(x.split()))

print("Thống kê độ dài:")
print(df[["doc_len", "sum_len"]].describe())

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(df["doc_len"], bins=50, kde=True, color="skyblue")
plt.title("Phân bố độ dài Document")

plt.subplot(1, 2, 2)
sns.histplot(df["sum_len"], bins=50, kde=True, color="salmon")
plt.title("Phân bố độ dài Summary")
plt.tight_layout()
plt.show()
dataset = df[['Document', 'Summary']].rename(columns={'Document': 'input_text', 'Summary': 'target_text'})

hf_dataset = Dataset.from_pandas(dataset)
def preprocess_data(batch):
    inputs = tokenizer(
        batch["input_text"],
        max_length=512,
        truncation=True,
        padding="max_length"
    )
    targets = tokenizer(
        batch["target_text"],
        max_length=128,
        truncation=True,
        padding="max_length"
    )
    inputs["labels"] = targets["input_ids"]
    return inputs

tokenized_dataset = hf_dataset.map(
    preprocess_data,
    batched=True,
    remove_columns=["input_text", "target_text"]
)
split_dataset = tokenized_dataset.train_test_split(test_size=0.1)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

def compute_metrics(eval_preds, tokenizer):
    preds, labels = eval_preds
    labels = [[(l if l != -100 else tokenizer.pad_token_id) for l in label] for label in labels]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels)

    return {
        "rouge1": result["rouge1"] * 100,
        "rouge2": result["rouge2"] * 100,
        "rougeL": result["rougeL"] * 100
    }

training_args = Seq2SeqTrainingArguments(
    output_dir="./vit5-vietnamese-summarization",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=15,
    predict_with_generate=True,
    logging_dir="./logs",
    remove_unused_columns=False)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=split_dataset['train'],
    eval_dataset=split_dataset['test'],
    data_collator=data_collator,
    compute_metrics=compute_metrics
)
trainer.train()

metrics = trainer.evaluate()

from transformers import Seq2SeqTrainingArguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./vit5-vietnamese-summarization-1",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=2,
    weight_decay=0.01,
    save_total_limit=1,
    num_train_epochs=5,
    predict_with_generate=True,
    fp16=False,
    logging_dir="./logs",
    remove_unused_columns=False,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=split_dataset['train'],
    eval_dataset=split_dataset['test'],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)
trainer.train()

metrics = trainer.evaluate()