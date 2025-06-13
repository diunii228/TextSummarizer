"use client"

import { useState } from "react"
import { Link, useNavigate } from "react-router-dom"
import apiService from "../services/api"
import "./LoginPage.css"

const LoginPage = () => {
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState("")
  const navigate = useNavigate()

  const handleLogin = async (e) => {
    e.preventDefault()
    setIsLoading(true)
    setError("")

    try {
      const response = await apiService.login(email, password)
      console.log("Login successful:", response)

      // Redirect to home page
      navigate("/")
      window.location.reload() // Refresh to update header
    } catch (error) {
      console.error("Login error:", error)
      setError(error.message || "Đăng nhập thất bại")
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="login-container">
      <form className="login-box" onSubmit={handleLogin}>
        <h2 className="login-title">ĐĂNG NHẬP</h2>

        {error && <div className="error-message">{error}</div>}

        <label>Email</label>
        <input
          type="email"
          placeholder="Nhập email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
          disabled={isLoading}
        />

        <label>Mật khẩu</label>
        <input
          type="password"
          placeholder="Nhập mật khẩu"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
          disabled={isLoading}
        />

        <button type="submit" className="login-button" disabled={isLoading}>
          {isLoading ? "Đang đăng nhập..." : "Đăng nhập"}
        </button>

        <div className="login-links">
          <Link to="/forgotpassword">Quên mật khẩu</Link>
          <Link to="/register">Chưa có tài khoản?</Link>
        </div>
      </form>
    </div>
  )
}

export default LoginPage