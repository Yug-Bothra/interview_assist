import React from "react";
import { X, Volume2, AlertCircle } from "lucide-react";

export default function TabAudioModal({
  onSelect,
  onClose,
  errorMessage,
}) {
  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-gray-900 rounded-2xl border border-gray-700 max-w-md w-full shadow-2xl">
        {/* Header */}
        <div className="border-b border-gray-700 px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-green-500/20 rounded-lg flex items-center justify-center">
              <Volume2 className="w-5 h-5 text-green-400" />
            </div>
            <h2 className="text-lg font-semibold text-white">
              Enable Interviewer Audio
            </h2>
          </div>
          <button 
            onClick={onClose} 
            className="text-gray-400 hover:text-gray-200 transition-colors p-1 hover:bg-gray-800 rounded"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-5">
          {/* Instructions Card */}
          <div className="bg-gradient-to-br from-blue-900/20 to-purple-900/20 border border-blue-500/30 rounded-xl p-5">
            <p className="text-blue-300 text-sm font-semibold mb-4">
              📋 Follow these steps:
            </p>
            <ol className="space-y-3 text-sm text-gray-300">
              <li className="flex items-start gap-3">
                <span className="flex-shrink-0 w-6 h-6 bg-blue-500/30 text-blue-400 rounded-full flex items-center justify-center text-xs font-bold">
                  1
                </span>
                <span>Click "Select Tab Audio" below</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="flex-shrink-0 w-6 h-6 bg-blue-500/30 text-blue-400 rounded-full flex items-center justify-center text-xs font-bold">
                  2
                </span>
                <span>Choose your interview tab (Zoom/Meet/Teams)</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="flex-shrink-0 w-6 h-6 bg-blue-500/30 text-blue-400 rounded-full flex items-center justify-center text-xs font-bold">
                  3
                </span>
                <span>Enable "Share tab audio" checkbox</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="flex-shrink-0 w-6 h-6 bg-blue-500/30 text-blue-400 rounded-full flex items-center justify-center text-xs font-bold">
                  4
                </span>
                <span>Click "Share" to confirm</span>
              </li>
            </ol>
          </div>

          {/* Error Message */}
          {errorMessage && (
            <div className="bg-red-900/20 border border-red-500/30 rounded-xl p-4 flex items-start gap-3 animate-pulse">
              <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-red-300 text-sm font-medium mb-1">Error</p>
                <p className="text-red-200 text-sm">{errorMessage}</p>
              </div>
            </div>
          )}

          {/* Action Button */}
          <button
            onClick={onSelect}
            className="w-full bg-gradient-to-r from-green-600 to-green-500 hover:from-green-700 hover:to-green-600 text-white py-3.5 px-6 rounded-lg font-semibold transition-all shadow-lg hover:shadow-green-500/20 flex items-center justify-center gap-2"
          >
            <Volume2 className="w-5 h-5" />
            Select Tab Audio
          </button>

          {/* Info Text */}
          <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-3 flex items-start gap-2">
            <div className="w-1.5 h-1.5 bg-green-500 rounded-full mt-2 flex-shrink-0 animate-pulse" />
            <p className="text-xs text-gray-400 leading-relaxed">
              Your microphone is already active. This step captures the interviewer's voice from your browser tab.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}