<<<<<<< HEAD
# TextSummarizer
 A NLP project for Vietnamese text summarization
=======

## 📌 Overview
This project builds a **Vietnamese text summarization system** using the **ViT5-base** model, fine-tuned on over 10,000 news and biomedical documents.  
The system can generate concise and accurate summaries for Vietnamese text, optimized for both performance and user experience.

## 🚀 Features
- **NLP Pipeline**: Data preprocessing (tokenization, labeling, normalization, batching).
- **Model**: ViT5-base fine-tuning using Hugging Face Transformers.
- **Evaluation**: Achieved high ROUGE scores, outperforming several public fine-tuned models.
- **Deployment**: Web-based demo for instant summarization.

## 📂 Dataset
- **Source**:
  - [Vietnews](https://huggingface.co/datasets/vietnews)
  - `bio_medicine.csv` from [Hugging Face Vietnamese Summarization dataset](https://huggingface.co/datasets/vietnamese-summarization)
- **Size**: 10,653 records
- **Structure**:
  - `Document`: Original Vietnamese text
  - `Summary`: Reference summary
  - `Dataset`: Source category

## 🛠 Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/vietnamese-text-summarization.git
cd vietnamese-text-summarization

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

