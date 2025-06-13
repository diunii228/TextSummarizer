const pool = require("../db");
const summarizeText = require("../utils/summarizer");

// Validate input văn bản
const validateInput = (text) => {
  if (!text || text.trim() === "") return "Văn bản không được để trống";
  if (text.length < 10) return "Văn bản quá ngắn để tóm tắt (tối thiểu 10 ký tự)";
  if (text.length > 10000) return "Văn bản quá dài (tối đa 10,000 ký tự)";
  return null;
};

exports.summarize = async (req, res) => {
  // const { original_text, summary_type = "medium" } = req.body;
  const { original_text } = req.body;
  const user_id = req.user.id;
  const errorMsg = validateInput(original_text);
  if (errorMsg) {
    return res.status(400).json({ message: errorMsg });
  }

  try {
    // console.log(`User ${user_id} đang yêu cầu tóm tắt (${summary_type}) – Length: ${original_text.length}`);

    // const result = await summarizeText(original_text, summary_type);
    const result = await summarizeText(original_text);

    // const [dbResult] = await pool.query(
    //   `INSERT INTO summaries 
    //     (user_id, original_text, summary_text, summary_type, compression_ratio) 
    //    VALUES (?, ?, ?, ?, ?)`,
    //   [user_id, original_text, result.summary, summary_type, result.compression_ratio]
    // );

    const [dbResult] = await pool.query(
      `INSERT INTO summaries 
        (user_id, original_text, summary_text) 
       VALUES (?, ?, ?)`,
      [user_id, original_text, result.summary]
    );

    res.status(200).json({
      id: dbResult.insertId,
      user_id,
      original_text,
      summary_text: result.summary,
      // summary_type,
      compression_ratio: result.compression_ratio,
      original_length: result.original_length,
      summary_length: result.summary_length,
      created_at: new Date(),
      ai_powered: !result.fallback,
    });

  } catch (error) {
    console.error("Lỗi tóm tắt:", error);
    res.status(500).json({
      message: "Đã xảy ra lỗi khi tóm tắt.",
      error: process.env.NODE_ENV === "development" ? error.message : undefined,
    });
  }
};

exports.getSummaryStats = async (req, res) => {
  const user_id = req.user.id;

  try {
    const [stats] = await pool.query(
      `SELECT 
        COUNT(*) as total_summaries,
        SUM(LENGTH(original_text)) as total_original_chars,
        SUM(LENGTH(summary_text)) as total_summary_chars,
        AVG(compression_ratio) as avg_compression_ratio,
        COUNT(CASE WHEN summary_type = 'short' THEN 1 END) as short_summaries,
        COUNT(CASE WHEN summary_type = 'medium' THEN 1 END) as medium_summaries,
        COUNT(CASE WHEN summary_type = 'detailed' THEN 1 END) as detailed_summaries
      FROM summaries 
      WHERE user_id = ?`,
      [user_id]
    );

    res.json(stats[0] || {
      total_summaries: 0,
      total_original_chars: 0,
      total_summary_chars: 0,
      avg_compression_ratio: 0,
      short_summaries: 0,
      medium_summaries: 0,
      detailed_summaries: 0,
    });

  } catch (error) {
    console.error("Error getting stats:", error);
    res.status(500).json({ message: "Lỗi server khi lấy thống kê" });
  }
};