// C:\Users\yugbo\Desktop\interview_assistent\frontend\src\components\InterviewAssistent\LaunchChecklist.jsx

import React from 'react';
import { Check, FileText, Building2, Briefcase, Globe2 } from 'lucide-react';

export default function LaunchChecklist({ persona, personaData, domain }) {
  const checklistItems = [
    {
      icon: Building2,
      title: 'Persona Selected',
      description: persona || 'No persona selected',
      status: !!persona,
    },
    {
      icon: Globe2,
      title: 'Interview Domain',
      description: domain || 'No domain selected',
      status: !!domain,
    },
    {
      icon: FileText,
      title: 'Resume',
      description: personaData?.resume_filename || 'No resume attached',
      status: !!personaData?.resume_filename,
    },
  ];

  return (
    <div className="bg-[#14141b] rounded-2xl p-8 border border-gray-800">
      <div className="mb-8">
        <h2 className="text-white text-xl font-semibold mb-2">Pre-flight Checklist</h2>
        <p className="text-gray-400 text-sm">
          Review your selections before launching the interview assistant
        </p>
      </div>

      {/* Persona Details Card */}
      {personaData && (
        <div className="bg-gradient-to-br from-purple-900/20 to-pink-900/20 border border-purple-700/50 rounded-xl p-6 mb-6">
          <div className="flex items-start gap-4">
            <div className="w-16 h-16 bg-gradient-to-br from-purple-600 to-pink-600 rounded-xl flex items-center justify-center text-2xl text-white font-bold flex-shrink-0">
              {personaData.company_name?.charAt(0).toUpperCase()}
            </div>
            <div className="flex-1 min-w-0">
              <h3 className="text-white font-bold text-lg mb-1">{personaData.position}</h3>
              <p className="text-purple-400 text-sm mb-3">@{personaData.company_name}</p>
              <p className="text-gray-400 text-sm line-clamp-2">
                {personaData.company_description}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Checklist Items */}
      <div className="space-y-4 mb-8">
        {checklistItems.map((item, index) => (
          <div
            key={index}
            className={`bg-[#1a1a24] border ${
              item.status ? 'border-green-700/50' : 'border-gray-700'
            } rounded-xl p-4 transition-all`}
          >
            <div className="flex items-start gap-4">
              <div
                className={`w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0 ${
                  item.status ? 'bg-green-500/20' : 'bg-gray-700'
                }`}
              >
                {item.status ? (
                  <Check className="w-5 h-5 text-green-400" />
                ) : (
                  <item.icon className="w-5 h-5 text-gray-400" />
                )}
              </div>
              <div className="flex-1 min-w-0">
                <h4 className="text-white font-medium mb-1">{item.title}</h4>
                <p className={`text-sm ${item.status ? 'text-gray-300' : 'text-gray-500'} truncate`}>
                  {item.description}
                </p>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* System Status */}
      <div className="bg-[#1a1a24] border border-gray-700 rounded-xl p-4 mb-6">
        <div className="flex items-center gap-3 mb-4">
          <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
          <h4 className="text-white font-medium">System Status</h4>
        </div>
        <div className="space-y-2 text-sm">
          <div className="flex items-center justify-between">
            <span className="text-gray-400">AI Assistant</span>
            <span className="text-green-400">Ready</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-gray-400">Voice Recognition</span>
            <span className="text-green-400">Active</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-gray-400">Context Engine</span>
            <span className="text-green-400">Loaded</span>
          </div>
        </div>
      </div>

      {/* Ready Status */}
      <div className="bg-gradient-to-r from-green-500/10 to-emerald-500/10 border border-green-500/30 rounded-xl p-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-green-500/20 rounded-full flex items-center justify-center">
            <Check className="w-6 h-6 text-green-400" />
          </div>
          <div>
            <h4 className="text-green-400 font-semibold">All Systems Go!</h4>
            <p className="text-gray-400 text-sm">Your interview assistant is ready to launch</p>
          </div>
        </div>
      </div>

      {/* Tips */}
      <div className="mt-6 bg-blue-500/10 border border-blue-500/30 rounded-xl p-4">
        <h4 className="text-blue-400 font-medium mb-2">💡 Pro Tips</h4>
        <ul className="text-gray-400 text-sm space-y-1">
          <li>• Test your microphone before starting</li>
          <li>• Ensure you're in a quiet environment</li>
          <li>• Keep your resume details handy</li>
        </ul>
      </div>
    </div>
  );
}