require("dotenv").config();
const axios = require("axios");

const AI_SERVICE_URL = process.env.AI_SERVICE_URL;
// const AI_SERVICE_URL = "http://localhost:6000";

async function summarizeText(text, summaryType = "medium") {
  try {
    console.log(`Gọi AI Service tại ${AI_SERVICE_URL}/summarize`);

    const { data } = await axios.post(
      `${AI_SERVICE_URL}/summarize`,
      {
        text: text,
        // summary_type: summaryType,
      },
      {
        timeout: 30000,
        headers: { "Content-Type": "application/json" },
      }
    );

    console.log(`AI summarization OK. Compression: ${data.compression_ratio}`);

    return {
      summary: data.summary,
      compression_ratio: data.compression_ratio,
      original_length: data.original_length,
      summary_length: data.summary_length,
      // summary_type: data.summary_type,
    };
  } catch (error) {
    console.error("Lỗi gọi AI Service:", error.message);

    // fallback
    const sentences = text.split(/[.!?]+/).filter((s) => s.trim() !== "");

    let summaryLength;
    switch (summaryType) {
      case "short":
        summaryLength = Math.min(2, sentences.length);
        break;
      case "detailed":
        summaryLength = Math.min(5, sentences.length);
        break;
      default:
        summaryLength = Math.min(3, sentences.length);
    }

    const summary = sentences.slice(0, summaryLength).join(". ") + ".";

    return {
      summary,
      compression_ratio: Math.round((summary.length / text.length) * 100) / 100,
      original_length: text.split(" ").length,
      summary_length: summary.split(" ").length,
      // summary_type: summaryType,
      fallback: true,
    };
  }
}

module.exports = summarizeText;