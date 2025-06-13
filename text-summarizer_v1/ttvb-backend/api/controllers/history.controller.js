const pool = require('../db');

exports.getHistory = async (req, res) => {
  try {
    const userId = req.user.id;
    const [rows] = await pool.query(
      `SELECT id, original_text, summary_text, created_at 
       FROM summaries 
       WHERE user_id = ? 
       ORDER BY created_at DESC`,
      [userId]
    );
    res.json(rows);
  } catch (error) {
    console.error('Lỗi lấy lịch sử:', error);
    res.status(500).json({ message: 'Lỗi server khi lấy lịch sử' });
  }
};

exports.saveHistory = async (req, res) => {
  try {
    const userId = req.user.id;
    const { original_text, summary_text } = req.body;

    const [result] = await pool.query(
      `INSERT INTO summaries (user_id, original_text, summary_text) 
       VALUES (?, ?, ?)`,
      [userId, original_text, summary_text]
    );

    res.status(201).json({ message: 'Lưu tóm tắt thành công', id: result.insertId });
  } catch (error) {
    console.error('Lỗi lưu tóm tắt:', error);
    res.status(500).json({ message: 'Lỗi server khi lưu tóm tắt' });
  }
};

exports.deleteHistory = async (req, res) => {
  try {
    const userId = req.user.id;
    const historyId = req.params.id;

    const [result] = await pool.query(
      `DELETE FROM summaries 
       WHERE id = ? AND user_id = ?`,
      [historyId, userId]
    );

    if (result.affectedRows === 0) {
      return res.status(404).json({ message: 'Không tìm thấy lịch sử hoặc không có quyền xóa' });
    }

    res.json({ message: 'Xóa thành công' });
  } catch (error) {
    console.error('Lỗi xóa lịch sử:', error);
    res.status(500).json({ message: 'Lỗi server khi xóa lịch sử' });
  }
};