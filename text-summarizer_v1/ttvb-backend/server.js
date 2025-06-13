const express = require("express");
const cors = require("cors");
const path = require("path");
require("dotenv").config();

const app = express();
const PORT = process.env.PORT || 5000;

// Middleware
app.use(cors({
  origin: "http://localhost:3000",
  credentials: true,
}));
app.use(express.json());

// API routes
const authRoutes = require("./api/routes/auth.routes");
const summaryRoutes = require("./api/routes/summary.routes");
const historyRoutes = require("./api/routes/history.routes");

app.use("/api/auth", authRoutes);
app.use("/api", summaryRoutes);
app.use("/api", historyRoutes);

// Health check
app.get("/health", (req, res) => {
  res.json({ status: "OK", message: "Server đang chạy" });
});

// Serve React build (chỉ dùng khi đã build production frontend)
const frontendPath = path.join(__dirname, "ttvb-frontend", "build");
if (process.env.NODE_ENV === "production" && require("fs").existsSync(frontendPath)) {
  app.use(express.static(frontendPath));

  app.get("*", (req, res) => {
    res.sendFile(path.join(frontendPath, "index.html"));
  });
}

// Global error handler
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({ message: "Lỗi middleware!", error: err.message });
});

// Start server
app.listen(PORT, () => {
  console.log(`Backend đang chạy tại: http://localhost:${PORT}`);
});