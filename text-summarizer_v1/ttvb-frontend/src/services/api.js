const API_BASE_URL = process.env.REACT_APP_API_BASE_URL;
// const API_BASE_URL = 'http://localhost:5000';
console.log("API base URL:", API_BASE_URL);

class ApiService {
  async makeRequest(url, options = {}) {
    const token = localStorage.getItem("token")

    const config = {
      headers: {
        "Content-Type": "application/json",
        ...(token && { Authorization: `Bearer ${token}` }),
        ...options.headers,
      },
      ...options,
    }

    try {
      const response = await fetch(`${API_BASE_URL}${url}`, config)
      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.message || "Có lỗi xảy ra")
      }

      return data
    } catch (error) {
      console.error("Lỗi API:", error)
      throw error
    }
  }

  // Authentication APIs
  async register(email, password) {
    return this.makeRequest("/auth/register", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    })
  }

  async login(email, password) {
    const data = await this.makeRequest("/auth/login", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    })

    if (data.token) {
      localStorage.setItem("token", data.token)
      localStorage.setItem("user", JSON.stringify(data.user))
    }

    return data
  }

  // Summarization API các loại tóm tắt api: http://localhost:5000/api/summarize
  async summarizeText(originalText, summaryType = "medium") {
    return this.makeRequest("/summarize", {
      method: "POST",
      body: JSON.stringify({
        original_text: originalText,
        // summary_type: summaryType,
      }),
    })
  }

  // Get trạng thái user
  async getUserStats() {
    return this.makeRequest("/stats")
  }

  // API lịch sử
  async getHistory() {
    return this.makeRequest("/history")
  }

  async saveHistory(originalText, summaryText) {
    return this.makeRequest("/history", {
      method: "POST",
      body: JSON.stringify({
        original_text: originalText,
        summary_text: summaryText,
      }),
    })
  }

  async deleteHistory(id) {
    return this.makeRequest(`/history/${id}`, {
      method: "DELETE",
    })
  }

  // Kiểm tra AI Service health check
  async checkBackendHealth() {
    // return this.makeRequest("/api/health")
    return fetch("http://localhost:5000/health").then((res) => res.json())
  }

  async checkAIServer() {
    return fetch("http://localhost:6000/health").then(res => res.json());
  }

  // Utility methods
  logout() {
    localStorage.removeItem("token")
    localStorage.removeItem("user")
  }

  getCurrentUser() {
    const user = localStorage.getItem("user")
    return user ? JSON.parse(user) : null
  }

  isAuthenticated() {
    return !!localStorage.getItem("token")
  }
}

const apiService = new ApiService();
export default apiService;