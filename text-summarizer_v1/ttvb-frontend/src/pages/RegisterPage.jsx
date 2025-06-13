"use client"

import { useState } from "react"
import { Link, useNavigate } from "react-router-dom"
import apiService from "../services/api"
import "./RegisterPage.css"

const RegisterPage = () => {
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [confirmPassword, setConfirmPassword] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState("")
  const [success, setSuccess] = useState("")
  const navigate = useNavigate()

  const handleRegister = async (e) => {
    e.preventDefault()

    if (password !== confirmPassword) {
      setError("Mật khẩu không khớp!")
      return
    }

    if (password.length < 6) {
      setError("Mật khẩu phải có ít nhất 6 ký tự!")
      return
    }

    setIsLoading(true)
    setError("")
    setSuccess("")

    try {
      await apiService.register(email, password)
      setSuccess("Đăng ký thành công! Đang chuyển hướng...")

      // Auto login after successful registration
      setTimeout(async () => {
        try {
          await apiService.login(email, password)
          navigate("/")
          window.location.reload()
        } catch (loginError) {
          navigate("/login")
        }
      }, 2000)
    } catch (error) {
      console.error("Registration error:", error)
      setError(error.message || "Đăng ký thất bại")
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="register-container">
      <form className="register-box" onSubmit={handleRegister}>
        <h2 className="register-title">ĐĂNG KÝ</h2>

        {error && <div className="error-message">{error}</div>}
        {success && <div className="success-message">{success}</div>}

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
          placeholder="Nhập mật khẩu (ít nhất 6 ký tự)"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
          disabled={isLoading}
          minLength={6}
        />

        <label>Xác nhận mật khẩu</label>
        <input
          type="password"
          placeholder="Nhập lại mật khẩu"
          value={confirmPassword}
          onChange={(e) => setConfirmPassword(e.target.value)}
          required
          disabled={isLoading}
        />

        <button type="submit" className="register-button" disabled={isLoading}>
          {isLoading ? "Đang đăng ký..." : "Đăng ký"}
        </button>

        <div className="register-links">
          <Link to="/login">Đã có tài khoản</Link>
        </div>
      </form>
    </div>
  )
}

export default RegisterPage