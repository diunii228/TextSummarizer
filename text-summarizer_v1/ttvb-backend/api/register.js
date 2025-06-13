const express = require('express');
const bcrypt = require('bcryptjs');
const cors = require('cors');

const app = express();
app.use(cors());
app.use(express.json());

const users = [];

app.post('/api/register', async (req, res) => {
  const { email, username, password } = req.body;

  try {
    const [existing] = await pool.query('SELECT * FROM accounts WHERE email = ?', [email]);
    if (existing.length > 0) {
      return res.status(400).json({ message: 'Email đã được sử dụng' });
    }

    const passwordHash = bcrypt.hashSync(password, 8);
    await pool.query('INSERT INTO accounts (username, email, password) VALUES (?, ?, ?)', [username, email, passwordHash]);

    res.status(201).json({ message: 'Đăng ký thành công' });
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Lỗi server' });
  }
});

const PORT = 5000;
app.listen(PORT, () => console.log(`Server chạy cổng ${PORT}`));
