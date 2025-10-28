import React from 'react';
import { Sparkles, MessageSquare, Users, Briefcase, HelpCircle, User } from 'lucide-react';
import { useAuth } from '../Auth/AuthContext';


export default function Sidebar({ currentPage, onNavigate }) {
  const { user } = useAuth();

  // Get user's initials for avatar
  const getInitials = () => {
    if (user?.user_metadata?.full_name) {
      const names = user.user_metadata.full_name.split(' ');
      if (names.length >= 2) {
        return `${names[0][0]}${names[names.length - 1][0]}`.toUpperCase();
      }
      return names[0][0].toUpperCase();
    } else if (user?.email) {
      return user.email[0].toUpperCase();
    }
    return 'U';
  };

  // Get user's full name or email
  const getUserName = () => {
    if (user?.user_metadata?.full_name) {
      return user.user_metadata.full_name;
    } else if (user?.email) {
      return user.email;
    }
    return 'User';
  };

  return (
    <aside className="w-[280px] bg-[#14141b] border-r border-gray-800 flex flex-col">
      {/* Logo */}
      <div className="p-6 border-b border-gray-800">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 bg-gray-900 rounded flex items-center justify-center">
            <svg
              className="w-5 h-5 text-white"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M13 10V3L4 14h7v7l9-11h-7z"
              />
            </svg>
          </div>
          <span className="text-white text-xl font-bold">InterviewAI</span>
        </div>
      </div>

      {/* User Profile */}
      <div className="p-4 border-b border-gray-800">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-gradient-to-br from-amber-600 to-amber-700 rounded-full flex items-center justify-center">
            <span className="text-white font-semibold text-sm">{getInitials()}</span>
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-white text-sm font-medium truncate">{getUserName()}</p>
            <p className="text-gray-400 text-xs truncate">Manage Account (Free Plan)</p>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4 overflow-y-auto">
        {/* Interview Section */}
        <div className="mb-6">
          <p className="text-gray-500 text-xs font-semibold uppercase mb-2 px-2">Interview</p>
          <button
            onClick={() => onNavigate('launchpad')}
            className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg transition-colors ${
              currentPage === 'launchpad'
                ? 'bg-gradient-to-r from-blue-600/20 to-purple-600/20 text-white border border-blue-500/30'
                : 'text-gray-400 hover:text-white hover:bg-gray-800/50'
            }`}
          >
            <Sparkles className="w-5 h-5" />
            <span className="text-sm font-medium">AI Interview Copilot</span>
          </button>
          <button
            onClick={() => onNavigate('ai-mock')}
            className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg transition-colors mt-1 ${
              currentPage === 'ai-mock'
                ? 'bg-gradient-to-r from-blue-600/20 to-purple-600/20 text-white border border-blue-500/30'
                : 'text-gray-400 hover:text-white hover:bg-gray-800/50'
            }`}
          >
            <MessageSquare className="w-5 h-5" />
            <span className="text-sm font-medium">AI Mock</span>
          </button>
          <button
            onClick={() => onNavigate('personas')}
            className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg transition-colors mt-1 ${
              currentPage === 'personas'
                ? 'bg-gradient-to-r from-blue-600/20 to-purple-600/20 text-white border border-blue-500/30'
                : 'text-gray-400 hover:text-white hover:bg-gray-800/50'
            }`}
          >
            <Users className="w-5 h-5" />
            <span className="text-sm font-medium">Personas</span>
          </button>
        </div>

        {/* Prepare Section */}
        <div className="mb-6">
          <p className="text-gray-500 text-xs font-semibold uppercase mb-2 px-2">Prepare</p>
          <button
            onClick={() => onNavigate('jobs')}
            className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg transition-colors ${
              currentPage === 'jobs'
                ? 'bg-gradient-to-r from-blue-600/20 to-purple-600/20 text-white border border-blue-500/30'
                : 'text-gray-400 hover:text-white hover:bg-gray-800/50'
            }`}
          >
            <Briefcase className="w-5 h-5" />
            <span className="text-sm font-medium">Jobs</span>
            <span className="ml-auto text-xs bg-purple-500/20 text-purple-300 px-2 py-0.5 rounded">New</span>
          </button>
          <button
            onClick={() => onNavigate('questions')}
            className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg transition-colors mt-1 ${
              currentPage === 'questions'
                ? 'bg-gradient-to-r from-blue-600/20 to-purple-600/20 text-white border border-blue-500/30'
                : 'text-gray-400 hover:text-white hover:bg-gray-800/50'
            }`}
          >
            <HelpCircle className="w-5 h-5" />
            <span className="text-sm font-medium">Questions</span>
          </button>
          <button
            onClick={() => onNavigate('train')}
            className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg transition-colors mt-1 ${
              currentPage === 'train'
                ? 'bg-gradient-to-r from-blue-600/20 to-purple-600/20 text-white border border-blue-500/30'
                : 'text-gray-400 hover:text-white hover:bg-gray-800/50'
            }`}
          >
            <User className="w-5 h-5" />
            <span className="text-sm font-medium">Train Copilot</span>
          </button>
        </div>
      </nav>

      {/* Desktop App Promotion */}
      <div className="p-4 border-t border-gray-800">
        <div className="bg-gradient-to-br from-purple-900/40 to-pink-900/40 rounded-xl p-4 border border-purple-500/20">
          <h3 className="text-white font-semibold text-sm mb-2">Try Our Desktop App</h3>
          <p className="text-gray-400 text-xs mb-4 leading-relaxed">
            Get Stealth Mode and other exclusive features on your desktop
          </p>
          <div className="flex items-center justify-between mb-3">
            <span className="text-gray-500 text-xs">2/2</span>
          </div>
          <button className="w-full bg-gray-800 hover:bg-gray-700 text-white text-sm font-medium py-2.5 rounded-lg transition-colors">
            Download
          </button>
        </div>
      </div>
    </aside>
  );
}