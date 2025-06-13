"use client"

import { useState, useEffect } from "react"
import { useNavigate } from "react-router-dom"
import apiService from "../services/api"
import Header from "../components/header"
import "./HistoryPage.css"

const HistoryPage = () => {
  const [history, setHistory] = useState([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState("")
  const navigate = useNavigate()

  useEffect(() => {
    if (!apiService.isAuthenticated()) {
      navigate("/login")
      return
    }

    loadHistory()
  }, [navigate])

  const loadHistory = async () => {
    try {
      setIsLoading(true)
      const data = await apiService.getHistory()
      setHistory(data)
    } catch (error) {
      console.error("Error loading history:", error)
      setError("Không thể tải lịch sử")
    } finally {
      setIsLoading(false)
    }
  }

  const handleDelete = async (id) => {
    if (!window.confirm("Bạn có chắc muốn xóa mục này?")) {
      return
    }

    try {
      await apiService.deleteHistory(id)
      setHistory(history.filter((item) => item.id !== id))
    } catch (error) {
      console.error("Error deleting history:", error)
      alert("Không thể xóa mục này")
    }
  }

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleString("vi-VN")
  }

  if (isLoading) {
    return (
      <>
        <Header />
        <div className="history-container">
          <div className="loading">Đang tải lịch sử...</div>
        </div>
      </>
    )
  }

  return (
    <>
      <Header />
      <div className="history-container">
        <h1>Lịch sử tóm tắt</h1>

        {error && <div className="error-message">{error}</div>}

        {history.length === 0 ? (
          <div className="empty-history">
            <p>Chưa có lịch sử tóm tắt nào</p>
            <button onClick={() => navigate("/")} className="back-button">
              Về trang chủ
            </button>
          </div>
        ) : (
          <div className="history-list">
            {history.map((item) => (
              <div key={item.id} className="history-item">
                <div className="history-header">
                  <span className="history-date">{formatDate(item.created_at)}</span>
                  <button onClick={() => handleDelete(item.id)} className="delete-button">
                    🗑️
                  </button>
                </div>

                <div className="history-content">
                  <div className="original-text">
                    <h4>Văn bản gốc:</h4>
                    <p>{item.original_text}</p>
                  </div>

                  <div className="summary-text">
                    <h4>Tóm tắt:</h4>
                    <p>{item.summary_text}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </>
  )
}

export default HistoryPage