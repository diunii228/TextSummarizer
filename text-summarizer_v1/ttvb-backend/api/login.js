const express = require('express');
const jwt = require('jsonwebtoken');
const bcrypt = require('bcryptjs');
const cors = require('cors');

const app = express();
app.use(cors());
app.use(express.json());
require("dotenv").config();
const JWT_SECRET = process.env.JWT_SECRET; 

const users = [
  {
    id: 1,
    email: 'user@example.com',
    passwordHash: bcrypt.hashSync('123456', 8),
    name: 'Nguyen Van A',
  },
];

app.post('/api/register', (req, res) => {
  const { email, password } = req.body;
  if (users.find(u => u.email === email)) {
    return res.status(400).json({ message: 'Email đã tồn tại' });
  }
  const passwordHash = bcrypt.hashSync(password, 8);
  const newUser = { id: users.length + 1, email, passwordHash, name: email };
  users.push(newUser);
  res.json({ message: 'Đăng ký thành công' });
});

app.post('/api/login', async (req, res) => {
  const { email, password } = req.body;

  try {
    const [rows] = await pool.query('SELECT * FROM accounts WHERE email = ?', [email]);
    if (rows.length === 0) {
      return res.status(401).json({ message: 'Email hoặc mật khẩu không đúng' });
    }
    const user = rows[0];
    const isPasswordValid = bcrypt.compareSync(password, user.password);
    if (!isPasswordValid) {
      return res.status(401).json({ message: 'Email hoặc mật khẩu không đúng' });
    }

    const token = jwt.sign({ id: user.id, email: user.email }, JWT_SECRET, { expiresIn: '1h' });

    res.json({
      token,
      user: {
        id: user.id,
        email: user.email,
        name: user.username,
      },
    });
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Lỗi server' });
  }
});

const PORT = 5000;
app.listen(PORT, () => console.log(`Server chạy cổng ${PORT}`));
