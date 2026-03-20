export class ReconnectingWebSocket {
  constructor(url, onMessage, onStatusChange, maxRetries = 5) {
    this.url = url;
    this.onMessage = onMessage;
    this.onStatusChange = onStatusChange;
    this.maxRetries = maxRetries;
    this.retryCount = 0;
    this.ws = null;
    this.shouldReconnect = true;
    this.reconnectTimeout = null;
    this.pingInterval = null;
  }

  connect() {
    return new Promise((resolve, reject) => {
      try {
        console.log(`🔗 Connecting to: ${this.url}`);
        this.ws = new WebSocket(this.url);

        this.ws.onopen = () => {
          console.log(`✅ Connected: ${this.url}`);
          this.retryCount = 0;
          this.onStatusChange?.("connected");

          // ✅ CRITICAL: Send immediate handshake to prevent Render timeout
          // Use setTimeout to ensure WebSocket is fully ready
          setTimeout(() => {
            this.send({
              type: "client_ready",
              timestamp: Date.now(),
            });
            console.log("📤 Sent immediate handshake to:", this.url);
          }, 100);

          this.startPingPong();
          resolve(this.ws);
        };

        this.ws.onmessage = (event) => {
          const data = JSON.parse(event.data);

          // Handle server keepalive ping
          if (data.type === "keepalive") {
            console.log("🏓 Received server keepalive");
            return;
          }

          // Handle server acknowledgment
          if (data.type === "server_ack") {
            console.log("✅ Server acknowledged handshake");
            return;
          }

          // Handle connection established
          if (data.type === "connection_established") {
            console.log("✅ Connection established:", data.message);
            return;
          }

          this.onMessage?.(event);
        };

        this.ws.onerror = (error) => {
          console.error(`❌ WebSocket error: ${this.url}`, error);
          reject(error);
        };

        this.ws.onclose = (event) => {
          console.log(
            `🔌 Connection closed: ${this.url}`,
            event.code,
            event.reason
          );
          this.stopPingPong();
          this.onStatusChange?.("disconnected");

          if (this.shouldReconnect && this.retryCount < this.maxRetries) {
            this.scheduleReconnect();
          }
        };
      } catch (error) {
        reject(error);
      }
    });
  }

  startPingPong() {
    // ✅ Send ping every 30 seconds (Render has 60s timeout)
    this.pingInterval = setInterval(() => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.send({ type: "ping" });
        console.log("🏓 Sent client ping");
      }
    }, 30000);
  }

  stopPingPong() {
    if (this.pingInterval) {
      clearInterval(this.pingInterval);
      this.pingInterval = null;
    }
  }

  scheduleReconnect() {
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
    }

    const delay = Math.min(1000 * Math.pow(2, this.retryCount), 16000);
    this.retryCount++;

    console.log(
      `🔄 Reconnecting in ${delay}ms (attempt ${this.retryCount}/${this.maxRetries})`
    );

    this.onStatusChange?.("reconnecting");

    this.reconnectTimeout = setTimeout(() => {
      console.log(`🔄 Attempting reconnection...`);
      this.connect().catch(() => {
        // Reconnection will be scheduled again by onclose handler
      });
    }, delay);
  }

  send(data) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
      return true;
    }
    return false;
  }

  close() {
    this.shouldReconnect = false;
    this.stopPingPong();

    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }

    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }
}