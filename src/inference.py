from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import load_model_and_tokenizer, generate_summary
import torch
import os
import gdown

def download_model_if_not_exists():
    model_path = Path("models/model.pt")
    if not model_path.exists():
        print("Model not found. Downloading...")
        file_id = "1qjknYW12UpdS3ALhrpUFi971wmqWl8Yg"
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, str(model_path), quiet=False)
    else:
        print("Model already exists.")

download_model_if_not_exists()
def summarize_text_input(model, tokenizer, device):
    print("Nhập văn bản cần tóm tắt (gõ 'exit' để thoát):")
    while True:
        text = input("\n Văn bản: ")
        if text.lower() == "exit":
            break
        if len(text.strip()) == 0:
            print("Vui lòng nhập văn bản hợp lệ.")
            continue
        summary = generate_summary(text, model, tokenizer, device=device)
        print("\nTóm tắt:\n" + summary)

def summarize_file(model, tokenizer, device):
    file_path = input("\nNhập đường dẫn tới file .txt: ").strip()
    if not os.path.isfile(file_path):
        print("File không tồn tại.")
        return
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    if len(text.strip()) == 0:
        print("File trống.")
        return
    summary = generate_summary(text, model, tokenizer, device=device)
    print("\nTóm tắt:\n" + summary)

def main():
    print("Vietnamese Text Summarization\n")
    print("Chọn chế độ:")
    print("1. Nhập văn bản trực tiếp")
    print("2. Nhập từ file .txt")
    choice = input(" Nhập lựa chọn (1 hoặc 2): ").strip()

    model_name = "/Users/ddlyy/Documents/GitHub/TextSummarizer/saved_models/vit5_summarizer"  
    tokenizer, model = load_model_and_tokenizer(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if choice == "1":
        summarize_text_input(model, tokenizer, device)
    elif choice == "2":
        summarize_file(model, tokenizer, device)
    else:
        print("Lựa chọn không hợp lệ.")

if __name__ == "__main__":
    main()

