import React from "react";
import {
  Sparkles,
  MessageSquare,
  Users,
  Briefcase,
  HelpCircle,
  User,
} from "lucide-react";
import { useAuth } from "../Auth/AuthContext";

export default function Sidebar({ currentPage, onNavigate }) {
  const { user } = useAuth();

  // Name & initials
  const getInitials = () => {
    if (user?.user_metadata?.full_name) {
      const parts = user.user_metadata.full_name.trim().split(" ");
      return parts.length >= 2
        ? (parts[0][0] + parts[parts.length - 1][0]).toUpperCase()
        : parts[0][0].toUpperCase();
    }
    return user?.email?.[0]?.toUpperCase() || "U";
  };

  const getUserName = () => {
    if (user?.user_metadata?.full_name) return user.user_metadata.full_name;
    if (user?.email) return user.email;
    return "User";
  };

  // Reusable button
  const NavButton = ({ icon: Icon, label, page }) => {
    const isActive = currentPage === page;

    return (
      <button
        onClick={() => onNavigate(page)}
        className={`w-full flex items-center gap-3 px-4 py-2.5 rounded-lg text-sm font-medium transition-all
          ${
            isActive
              ? "bg-[#1D1E22] text-white border border-purple-500/30"
              : "text-gray-400 hover:text-white hover:bg-[#1A1B1F]"
          }`}
      >
        <Icon className={`w-5 h-5 ${isActive ? "text-purple-300" : "text-gray-500"}`} />
        {label}
      </button>
    );
  };

  return (
    <aside className="w-[260px] h-screen bg-[#141518] border-r border-[#1F2024] flex flex-col sticky top-0">
      
      {/* Header Logo */}
      <div className="px-5 py-5 border-b border-[#1F2024]">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 bg-[#1D1E22] rounded-lg flex items-center justify-center">
            <Sparkles className="w-5 h-5 text-purple-300" />
          </div>
          <span className="text-[17px] font-semibold text-white">InterviewAI</span>
        </div>
      </div>

      {/* User Profile */}
      <div className="px-5 py-4 border-b border-[#1F2024]">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-[#1D1E22] flex items-center justify-center">
            <span className="text-white font-semibold text-sm">{getInitials()}</span>
          </div>
          <div>
            <p className="text-white text-sm font-medium leading-tight truncate">
              {getUserName()}
            </p>
            <p className="text-gray-500 text-[11px] leading-tight">
              Free Plan • Manage
            </p>
          </div>
        </div>
      </div>

      {/* Navigation Sections */}
      <nav className="flex-1 px-4 py-4 overflow-y-auto">
        
        {/* Interview Section */}
        <div className="mb-6">
          <p className="text-gray-500 text-[11px] font-semibold uppercase mb-2 px-1">
            INTERVIEW
          </p>

          <NavButton icon={Sparkles} label="AI Interview Copilot" page="launchpad" />
          <NavButton icon={MessageSquare} label="AI Mock" page="ai-mock" />
          <NavButton icon={Users} label="Personas" page="personas" />
        </div>

        {/* Prepare Section */}
        <div>
          <p className="text-gray-500 text-[11px] font-semibold uppercase mb-2 px-1">
            PREPARE
          </p>

          <NavButton icon={Briefcase} label="Jobs" page="jobs" />
          <NavButton icon={HelpCircle} label="Questions" page="questions" />
          <NavButton icon={User} label="Train Copilot" page="train" />
        </div>
      </nav>
    </aside>
  );
}
