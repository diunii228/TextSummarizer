"use client"

import { useState } from "react"
import apiService from "../services/api"
import "./content-box.css"
import docIcon from "../assets/images/docIcon.png"

function ContentBox() {
  const [text, setText] = useState("")
  const [summary, setSummary] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState("")
  const [summaryStats, setSummaryStats] = useState(null)

  const handleFileUpload = (event) => {
    const file = event.target.files[0]
    if (file && file.type === "text/plain") {
      const reader = new FileReader()
      reader.onload = (e) => {
        setText(e.target.result)
        setSummary("")
        setSummaryStats(null)
        setError("")
      }
      reader.readAsText(file)
    } else {
      setError("Chỉ hỗ trợ file .txt")
    }
  }

  const handleSummarize = async () => {
    if (!text.trim()) {
      setError("Vui lòng nhập nội dung cần tóm tắt")
      return
    }

    if (text.length < 50) {
      setError("Văn bản quá ngắn để tóm tắt (tối thiểu 50 ký tự)")
      return
    }

    setIsLoading(true)
    setError("")
    setSummaryStats(null)

    try {
      if (!apiService.isAuthenticated()) {
        const sentences = text.match(/[^.!?]+[.!?]?/g)
        if (!sentences) {
          setSummary("Không có nội dung hợp lệ để tóm tắt.")
          return
        }

        const summaryText = sentences.slice(0).join(" ")
        setSummary(summaryText)
        setSummaryStats({
          ai_powered: false,
          compression_ratio: Math.round((summaryText.length / text.length) * 100) / 100,
        })
      } else {
        const response = await apiService.summarizeText(text)
        setSummary(response.summary_text)
        setSummaryStats({
          ai_powered: response.ai_powered,
          compression_ratio: response.compression_ratio,
          original_length: response.original_length,
          summary_length: response.summary_length,
        })
      }
    } catch (error) {
      console.error("Summarization error:", error)
      setError(error.message || "Có lỗi xảy ra khi tóm tắt văn bản")

      const sentences = text.match(/[^.!?]+[.!?]?/g)
      if (sentences) {
        const summaryText = sentences.slice(0, 3).join(" ")
        setSummary(summaryText)
        setSummaryStats({
          ai_powered: false,
          compression_ratio: Math.round((summaryText.length / text.length) * 100) / 100,
        })
        setError("Đã sử dụng tóm tắt cơ bản (không kết nối được AI service)")
      }
    } finally {
      setIsLoading(false)
    }
  }

  const wordCount = text.trim() === "" ? 0 : text.trim().split(/\s+/).length

  return (
    <div className="container">
      <div className="input-section">
        <textarea
          className="text-input"
          placeholder="Nhập nội dung tại đây..."
          value={text}
          onChange={(e) => {
            setText(e.target.value)
            setError("")
          }}
          disabled={isLoading}
        />

        <div className="controls-section">
          <label className="upload-label">
            <img src={docIcon} alt="doc-icon" className="docIcon" />
            <span className="upload-text">Upload file (.txt)</span>
            <input type="file" accept=".txt" onChange={handleFileUpload} hidden disabled={isLoading} />
          </label>
        </div>

        {error && <div className="error-message">{error}</div>}

        <button className="summarize-button" onClick={handleSummarize} disabled={isLoading || !text.trim()}>
          {isLoading ? "Đang tóm tắt..." : "Tóm tắt"}
        </button>

        {!apiService.isAuthenticated() && (
          <div className="guest-notice">Đăng nhập để sử dụng AI BART model và lưu lịch sử</div>
        )}
      </div>

      <div className="output-section">
        <div className="output-content">
          {isLoading ? (
            <div className="loading">
              <div className="loading-spinner"></div>
              <p>Đang xử lý với AI model...</p>
            </div>
          ) : (
            summary
          )}
        </div>

        <div className="output-stats">
          <div className="word-count">{wordCount} từ</div>
          {summaryStats && (
            <div className="summary-info">
              <span className={`ai-badge ${summaryStats.ai_powered ? "ai-powered" : "basic"}`}>
                {/* {summaryStats.ai_powered ? "Ai model" : "Cơ bản"} */}
              </span>
              <span className="compression-ratio">Tỷ lệ nén: {summaryStats.compression_ratio}</span>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default ContentBox