import React from "react";
import { X } from "lucide-react";

export default function SettingsModal({ settings, backendUrl, onClose }) {
  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-md flex items-center justify-center z-50 p-4">
      <div className="bg-[#16171C] rounded-2xl border border-[#23242B] shadow-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">

        {/* Header */}
        <div className="sticky top-0 bg-[#16171C]/90 backdrop-blur-md border-b border-[#23242B] px-6 py-4 flex items-center justify-between z-20">
          <h2 className="text-2xl font-semibold text-white tracking-wide">Settings</h2>
          <button
            onClick={onClose}
            className="p-2 rounded-xl bg-[#1D1E24] hover:bg-[#25262D] text-gray-300 hover:text-white transition-all"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-8">

          {/* Configuration */}
          <div>
            <h3 className="text-lg font-bold text-purple-300 mb-4 tracking-wide">
              Configuration
            </h3>

            <div className="space-y-3 text-sm">
              {[
                ["Style", settings.selectedResponseStyleId || settings.responseStyle],
                ["Language", settings.audioLanguage],
                ["Pause Time", `${settings.pauseInterval}s`],
                ["Coding", settings.programmingLanguage],
                ["Model", settings.defaultModel],
                ["Backend", backendUrl],
              ].map(([label, value], i) => (
                <div
                  key={i}
                  className="flex justify-between py-2 border-b border-[#1F2025]"
                >
                  <span className="text-gray-500">{label}:</span>
                  <span className="text-gray-200 font-medium text-right max-w-xs break-words">
                    {value}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Optimization Box */}
          <div className="bg-gradient-to-br from-[#162015] to-[#0E1510] border border-green-700/40 rounded-xl p-5 shadow-inner">
            <h4 className="text-green-400 font-semibold mb-3 text-sm flex items-center gap-2">
              <span>💡</span> Render Optimization
            </h4>

            <ul className="space-y-2 text-sm text-gray-300">
              {[
                "30s client keepalive (Render 60s timeout)",
                "30s server keepalive (dual protection)",
                "Immediate WebSocket handshake",
                "Auto-reconnection with exponential backoff",
              ].map((text, i) => (
                <li key={i} className="flex items-start gap-2">
                  <span className="text-green-500 mt-0.5">•</span>
                  <span>{text}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* Footer */}
        <div className="border-t border-[#23242B] px-6 py-4 bg-[#16171C]/80 backdrop-blur-md">
          <button
            onClick={onClose}
            className="w-full py-3 rounded-xl bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 text-white font-semibold transition-all shadow-md"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
