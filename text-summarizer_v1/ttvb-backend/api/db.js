const mysql = require("mysql2/promise");
require("dotenv").config();

const pool = mysql.createPool({
  host: process.env.DB_HOST || "localhost",
  port: process.env.DB_PORT || 3306,
  user: process.env.DB_USER || "root",
  password: process.env.DB_PASSWORD || "",
  database: process.env.DB_NAME || "summary_accounts",
  waitForConnections: true,
  connectionLimit: 10,
  queueLimit: 0,
});

async function testConnection() {
  try {
    const connection = await pool.getConnection();
    console.log("Kết nối database thành công!");
    connection.release();
  } catch (error) {
    console.error("Kết nối database thất bại:", error.message);
    setTimeout(testConnection, 5000);
  }
}

testConnection();

module.exports = pool;