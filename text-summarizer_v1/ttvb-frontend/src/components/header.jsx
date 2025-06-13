"use client"

import { useState, useEffect } from "react"
import { useNavigate } from "react-router-dom"
import apiService from "../services/api"
import "./header.css"
import summzy from "../assets/images/summzy.png";
import botAI from "../assets/images/botAI.jpg";

function Header() {
  const [isOpen, setIsOpen] = useState(false)
  const [user, setUser] = useState(null)
  const navigate = useNavigate()

  useEffect(() => {
    const currentUser = apiService.getCurrentUser()
    setUser(currentUser)
  }, [])

  const toggleDropdown = () => {
    setIsOpen(!isOpen)
  }

  const handleLogin = () => {
    setIsOpen(false)
    navigate("/login")
  }

  const handleLogout = () => {
    apiService.logout()
    setUser(null)
    setIsOpen(false)
    navigate("/")
    window.location.reload()
  }

  const handleHistory = () => {
    setIsOpen(false)
    navigate("/history")
  }

  return (
    <div className="header-container">
      <div className="logo">
        <img src={summzy} alt="summzy-logo" className="summzy"/>
      </div>
      <div className="user-info">
        <span className="user-name">{user ? `Xin chào, ${user.name || user.email}` : "Khách"}</span>
        <div className="account" onClick={toggleDropdown}>
          {user ? (
            <img src={botAI} alt="user-avatar" className="avatar" />
          ) : (
            <div className="avatar">👤</div>
          )}
          {isOpen && (
            <div className="dropdown-menu">
              {!user ? (
                <div className="dropdown-item" onClick={handleLogin}>
                  Đăng nhập
                </div>
              ) : (
                <>
                  <div className="dropdown-item" onClick={handleHistory}>
                    Lịch sử
                  </div>
                  <div className="dropdown-item" onClick={handleLogout}>
                    Đăng xuất
                  </div>
                </>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default Header