import React from "react";

export default function StatusBar({
  deepgramStatus,
  qaStatus,
  isGenerating,
  isPaused,
}) {
  return (
    <div className="bg-[#111111] border-t border-[#1F1F1F] px-5 py-2.5">
      <div className="flex items-center justify-between text-[12px]">

        {/* LEFT STATUS INDICATORS */}
        <div className="flex items-center gap-6">

          {/* DEEPGRAM */}
          <div className="flex items-center gap-2">
            <div
              className={`w-2 h-2 rounded-full ${
                deepgramStatus.includes("Recording")
                  ? "bg-green-500"
                  : deepgramStatus.includes("Reconnecting")
                  ? "bg-yellow-400 animate-pulse"
                  : "bg-gray-600"
              }`}
            />
            <span className="text-gray-400">
              Deepgram: <span className="text-gray-300">{deepgramStatus}</span>
            </span>
          </div>

          {/* Q&A */}
          <div className="flex items-center gap-2">
            <div
              className={`w-2 h-2 rounded-full ${
                qaStatus.includes("Active")
                  ? "bg-purple-400"
                  : qaStatus.includes("Reconnecting")
                  ? "bg-yellow-400 animate-pulse"
                  : "bg-gray-600"
              }`}
            />
            <span className="text-gray-400">
              Q&A: <span className="text-gray-300">{qaStatus}</span>
            </span>
          </div>
        </div>

        {/* RIGHT SIDE: PROCESSING / PAUSED */}
        <div className="flex items-center gap-4">

          {isGenerating && (
            <div className="flex items-center gap-2">
              <div className="w-1 h-4 bg-[#222]" />
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-purple-400 rounded-full animate-pulse" />
                <span className="text-purple-300 text-[11px]">Processing…</span>
              </div>
            </div>
          )}

          {isPaused && (
            <div className="flex items-center gap-2">
              <div className="w-1 h-4 bg-[#222]" />
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-yellow-400 rounded-full" />
                <span className="text-yellow-400 text-[11px]">Paused</span>
              </div>
            </div>
          )}

        </div>
      </div>
    </div>
  );
}
