import React from 'react';
import { X, FileText, Download } from 'lucide-react';

export default function PersonaDetailsModal({ persona, onClose, onSelect, showSelectButton = true }) {
  if (!persona) return null;

  const getCompanyInitial = (companyName) =>
    companyName?.charAt(0).toUpperCase() || '?';

  const handleSelect = () => {
    if (onSelect) {
      // Pass full persona object to parent
      onSelect(persona);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/90 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-zinc-900 rounded-xl max-w-2xl w-full max-h-[90vh] overflow-y-auto border border-zinc-800">

        {/* Header */}
        <div className="sticky top-0 bg-zinc-900 border-b border-zinc-800 p-6">
          <div className="flex items-start justify-between">
            <div className="flex items-start gap-4">
              <div
                className={`w-16 h-16 rounded-xl flex items-center justify-center text-2xl font-bold ${
                  persona.is_sample
                    ? "bg-zinc-800 text-gray-400"
                    : "bg-white text-black"
                }`}
              >
                {getCompanyInitial(persona.company_name)}
              </div>

              <div>
                <h2 className="text-white text-2xl font-bold">{persona.position}</h2>
                <p className="text-gray-500">@{persona.company_name}</p>

                {persona.is_sample && (
                  <span className="inline-block mt-2 px-2.5 py-1 bg-zinc-800 rounded text-xs text-gray-400 font-medium">
                    Sample Persona
                  </span>
                )}
              </div>
            </div>

            <button
              onClick={onClose}
              className="text-gray-500 hover:text-white transition-colors"
            >
              <X className="w-6 h-6" />
            </button>
          </div>

          {/* Resume Info */}
          {persona.resume_filename && (
            <div className="mt-4 flex items-center gap-3 bg-black border border-zinc-800 rounded-lg p-3">
              <FileText className="w-5 h-5 text-gray-500" />

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
                  className="flex items-center gap-1 text-gray-400 hover:text-white text-sm font-medium transition-colors"
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

          {/* Company */}
          <div>
            <h3 className="text-white font-semibold mb-2">About the company</h3>
            <p className="text-gray-400 text-sm leading-relaxed">
              {persona.company_description || "No company description provided."}
            </p>
          </div>

          {/* Role */}
          <div>
            <h3 className="text-white font-semibold mb-2">About the role</h3>
            <p className="text-gray-400 text-sm leading-relaxed">
              {persona.job_description || "No job description provided."}
            </p>
          </div>

          {/* Buttons */}
          <div className="flex gap-3">
            <button
              onClick={onClose}
              className="flex-1 bg-zinc-800 hover:bg-zinc-700 text-white font-medium py-3 rounded-lg transition-colors"
            >
              Close
            </button>

            {showSelectButton && (
              <button
                onClick={handleSelect}
                className="flex-1 bg-white hover:bg-gray-100 text-black font-medium py-3 rounded-lg transition-colors"
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
