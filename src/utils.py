import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

def load_model_and_tokenizer(model_path):
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    return tokenizer, model

def generate_summary(text, model, tokenizer, device='cpu'):
    model.to(device)
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(model.device)
    summary_ids = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=128,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=3,        
        repetition_penalty=2.0,       
        length_penalty=1.1            
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
import evaluate

rouge = evaluate.load("rouge")

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
