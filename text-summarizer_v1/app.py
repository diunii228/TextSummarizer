from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import os
import sys
import torch

# Thêm src vào đường dẫn
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.utils import generate_summary, load_model_and_tokenizer

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask App
app = Flask(__name__)
CORS(app)

# Load model
try:
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "saved_models", "vit5_summarizer"))
    tokenizer, model = load_model_and_tokenizer(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    logger.info(f"Model loaded on {device} from {model_path}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    tokenizer, model, device = None, None, "cpu"

@app.route("/summarize", methods=["POST"])
def summarize_api():
    if not model or not tokenizer:
        return jsonify({"error": "Model not available"}), 503

    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    text = data["text"].strip()
    if len(text) < 50:
        return jsonify({"error": "Text too short for summarization (min 50 chars)"}), 400

    # summary_type = data.get("summary_type", "medium")
    # max_length, min_length = {
    #     "short": (50, 20),
    #     "detailed": (200, 100),
    #     "medium": (100, 50)
    # }.get(summary_type, (100, 50))

    try:
        summary_text = generate_summary(
            text,
            model=model,
            tokenizer=tokenizer,
            device=device,
            # max_length=max_length,
            # min_length=min_length
        )
        return jsonify({
            "summary": summary_text,
            "original_length": len(text.split()),
            "summary_length": len(summary_text.split()),
            "compression_ratio": round(len(summary_text.split()) / len(text.split()), 2)
            # "summary_type": summary_type
        })
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        return jsonify({"error": "Internal server error"}), 500
    
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "device": device
    })

if __name__ == "__main__":
    # Cổng có bảo mật, ko thể kiểm tra
    # port = int(os.environ.get("PORT", 6000))
    # app.run(host="0.0.0.0", port=port)
    # Nên chuyển sang cổng 5001 để kiểm tra api
    port = int(os.environ.get("PORT", 5001))
    app.run(host="localhost", port=port)