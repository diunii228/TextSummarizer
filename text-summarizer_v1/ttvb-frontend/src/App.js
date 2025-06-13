import { HashRouter as Router, Routes, Route } from "react-router-dom"
import Home from "./pages/home"
import LoginPage from "./pages/LoginPage"
import RegisterPage from "./pages/RegisterPage"
import ForgotPassword from "./pages/FogotPassword"
import HistoryPage from "./pages/HistoryPage"

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/login" element={<LoginPage />} />
        <Route path="/register" element={<RegisterPage />} />
        <Route path="/forgotpassword" element={<ForgotPassword />} />
        <Route path="/history" element={<HistoryPage />} />
      </Routes>
    </Router>
  )
}

export default App