
// ============================================================================
// FILE: utils/config.js
// ============================================================================

// Production URLs
// export const BACKEND_URL = "https://verve-ai-ukec.onrender.com";
// export const WS_URL = "wss://verve-ai-ukec.onrender.com";

// Development URLs
export const BACKEND_URL = "http://0.0.0.0:8000";
export const WS_URL = "wss://0.0.0.0:8000";

export const getWebSocketUrl = (path) => {
  const isDevelopment =
    window.location.hostname === "localhost" ||
    window.location.hostname === "127.0.0.1";

  if (isDevelopment) {
    return `ws://127.0.0.1:8000${path}`;
  } else {
    return `${WS_URL}${path}`;
  }
};