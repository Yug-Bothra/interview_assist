// C:\Users\yugbo\Desktop\interview_assistent\frontend\src\components\InterviewAssistent\PersonaDetailsModal.jsx

import React from 'react';
import { X, FileText, Download } from 'lucide-react';

export default function PersonaDetailsModal({ persona, onClose, onSelect, showSelectButton = true }) {
  if (!persona) return null;

  const getCompanyInitial = (companyName) => {
    return companyName?.charAt(0).toUpperCase() || '?';
  };

  const handleSelect = () => {
    if (onSelect) {
      onSelect(persona);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-4">
      <div className="bg-[#14141b] rounded-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto border border-gray-800">
        {/* Header */}
        <div className="sticky top-0 bg-[#14141b] border-b border-gray-800 p-6">
          <div className="flex items-start justify-between">
            <div className="flex items-start gap-4">
              <div
                className={`w-16 h-16 rounded-xl flex items-center justify-center text-2xl text-white font-bold ${
                  persona.is_sample
                    ? 'bg-gradient-to-br from-purple-600/30 to-pink-600/30'
                    : 'bg-gradient-to-br from-purple-600 to-pink-600'
                }`}
              >
                {getCompanyInitial(persona.company_name)}
              </div>
              <div>
                <h2 className="text-white text-2xl font-bold">{persona.position}</h2>
                <p className="text-gray-400">@{persona.company_name}</p>
                {persona.is_sample && (
                  <span className="inline-block mt-2 px-2 py-1 bg-purple-500/20 rounded text-xs text-purple-400">
                    Sample Persona
                  </span>
                )}
              </div>
            </div>
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-white transition-colors"
            >
              <X className="w-6 h-6" />
            </button>
          </div>

          {/* Resume Section */}
          {persona.resume_filename && (
            <div className="mt-4 flex items-center gap-3 bg-[#1a1a24] border border-gray-700 rounded-xl p-3">
              <FileText className="w-5 h-5 text-gray-400" />
              <div className="flex-1">
                <p className="text-white text-sm font-medium">
                  Resume: {persona.resume_filename}
                </p>
              </div>
              {persona.resume_url && (
                <a
                  href={persona.resume_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-1 text-purple-400 hover:text-purple-300 text-sm font-medium transition-colors"
                >
                  <Download className="w-4 h-4" />
                  View
                </a>
              )}
            </div>
          )}
        </div>

        {/* Content */}
        <div className="p-6 space-y-6">
          <div>
            <h3 className="text-white font-semibold mb-2">About the company</h3>
            <p className="text-gray-400 text-sm leading-relaxed">
              {persona.company_description}
            </p>
          </div>

          <div>
            <h3 className="text-white font-semibold mb-2">About the role</h3>
            <p className="text-gray-400 text-sm leading-relaxed">
              {persona.job_description}
            </p>
          </div>

          {/* Action Buttons */}
          <div className="flex gap-3">
            <button
              onClick={onClose}
              className="flex-1 bg-gray-700 hover:bg-gray-600 text-white font-semibold py-3 rounded-xl transition-all"
            >
              Close
            </button>
            {showSelectButton && (
              <button
                onClick={handleSelect}
                className="flex-1 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 text-white font-semibold py-3 rounded-xl transition-all"
              >
                Select
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}