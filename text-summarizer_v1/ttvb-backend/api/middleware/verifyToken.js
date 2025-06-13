const jwt = require('jsonwebtoken');
require("dotenv").config();
const JWT_SECRET = process.env.JWT_SECRET; 

function verifyToken(req, res, next) {
  const authHeader = req.headers['authorization'];
  if (!authHeader) return res.status(401).json({ message: 'Token không hợp lệ' });

  const token = authHeader.split(' ')[1];
  if (!token) return res.status(401).json({ message: 'Token không hợp lệ' });

  jwt.verify(token, JWT_SECRET, (err, user) => {
    if (err) return res.status(403).json({ message: 'Token hết hạn hoặc không hợp lệ' });
    req.user = user;
    next();
  });
}

module.exports = verifyToken;
