// ============================================================================
// FINAL CONFIG (NO AUTO-DETECTION)
// ============================================================================

// 🔥 CHANGE THIS ONLY WHEN NEEDED
const USE_LOCAL = false; // ✅ set true only for local testing

export const BACKEND_URL = USE_LOCAL
  ? "http://127.0.0.1:8000"
  : "https://interview-assist-1.onrender.com";

console.log("🔥 BACKEND_URL:", BACKEND_URL);

// WebSocket URL
export const getWebSocketUrl = (path) => {
  if (USE_LOCAL) {
    return `ws://127.0.0.1:8000${path}`;
  } else {
    return `wss://interview-assist-1.onrender.com${path}`;
  }
};