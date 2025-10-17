// frontend/src/components/Landing/Navbar.jsx

import React, { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../Auth/AuthContext";


// Features Dropdown Component (keeping same as before)
const FeaturesDropdown = ({ isOpen, onClose }) => {
  const dropdownRef = useRef(null);

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        onClose();
      }
    };
    if (isOpen) {
      document.addEventListener("mousedown", handleClickOutside);
    }
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <div ref={dropdownRef} className="absolute top-full left-1/2 transform -translate-x-1/2 mt-2 w-[600px] bg-white rounded-xl shadow-2xl border border-gray-100 p-6">
      <div className="grid grid-cols-2 gap-6">
        <div>
          <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-4">INTERVIEW COPILOT</h3>
          <div className="space-y-3">
            <div className="flex items-start gap-3 p-2 rounded-lg hover:bg-gray-50 cursor-pointer transition-colors">
              <div className="w-8 h-8 bg-indigo-100 rounded flex items-center justify-center flex-shrink-0">
                <svg className="w-4 h-4 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
                </svg>
              </div>
              <div>
                <h4 className="font-semibold text-gray-900 text-sm">AI Mock Interview</h4>
                <p className="text-xs text-gray-500 mt-0.5">Practice with AI recruiter</p>
              </div>
            </div>
          </div>
        </div>
        <div>
          <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-4">FREE TOOLS</h3>
          <div className="space-y-3">
            <div className="flex items-start gap-3 p-2 rounded-lg hover:bg-gray-50 cursor-pointer transition-colors">
              <div className="w-8 h-8 bg-purple-100 rounded flex items-center justify-center flex-shrink-0">
                <svg className="w-4 h-4 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
              </div>
              <div>
                <h4 className="font-semibold text-gray-900 text-sm">AI Resume Builder</h4>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// Resources Dropdown Component (simplified version)
const ResourcesDropdown = ({ isOpen, onClose }) => {
  const dropdownRef = useRef(null);

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        onClose();
      }
    };
    if (isOpen) {
      document.addEventListener("mousedown", handleClickOutside);
    }
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <div ref={dropdownRef} className="absolute top-full left-1/2 transform -translate-x-1/2 mt-2 w-[400px] bg-white rounded-xl shadow-2xl border border-gray-100 p-6">
      <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-4">RESOURCES</h3>
      <div className="space-y-3">
        <div className="flex items-start gap-3 p-2 rounded-lg hover:bg-gray-50 cursor-pointer transition-colors">
          <div className="w-8 h-8 bg-indigo-100 rounded flex items-center justify-center flex-shrink-0">
            <svg className="w-4 h-4 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
            </svg>
          </div>
          <div>
            <h4 className="font-semibold text-gray-900 text-sm">Blog</h4>
            <p className="text-xs text-gray-500 mt-0.5">Latest industry news</p>
          </div>
        </div>
      </div>
    </div>
  );
};

// User Menu Dropdown
const UserMenu = ({ user, onSignOut }) => {
  const [isOpen, setIsOpen] = useState(false);
  const menuRef = useRef(null);
  const navigate = useNavigate();

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (menuRef.current && !menuRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    };
    if (isOpen) {
      document.addEventListener("mousedown", handleClickOutside);
    }
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [isOpen]);

  const firstLetter = user?.email?.[0]?.toUpperCase() || 'U';

  return (
    <div className="relative" ref={menuRef}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-11 h-11 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-full flex items-center justify-center text-white font-bold cursor-pointer shadow-lg hover:shadow-xl transform hover:scale-110 transition-all duration-300"
      >
        {firstLetter}
      </button>

      {isOpen && (
        <div className="absolute right-0 mt-2 w-56 bg-white rounded-xl shadow-2xl border border-gray-100 py-2">
          <div className="px-4 py-3 border-b border-gray-100">
            <p className="text-sm font-medium text-gray-900 truncate">{user?.email}</p>
          </div>
          <button
            onClick={() => {
              navigate('/interview');
              setIsOpen(false);
            }}
            className="w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-50 transition-colors"
          >
            Dashboard
          </button>
          <button
            onClick={() => {
              navigate('/personas');
              setIsOpen(false);
            }}
            className="w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-50 transition-colors"
          >
            Manage Personas
          </button>
          <div className="border-t border-gray-100 mt-2 pt-2">
            <button
              onClick={onSignOut}
              className="w-full text-left px-4 py-2 text-sm text-red-600 hover:bg-red-50 transition-colors"
            >
              Sign Out
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

// Main Navbar Component
export default function Navbar() {
  const { user, signOut } = useAuth();
  const navigate = useNavigate();
  const [scrolled, setScrolled] = useState(false);
  const [featuresOpen, setFeaturesOpen] = useState(false);
  const [resourcesOpen, setResourcesOpen] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 20);
    };
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const handleSignOut = async () => {
    await signOut();
    navigate('/');
  };

  return (
    <nav className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${scrolled ? "bg-white shadow-md" : "bg-white/80 backdrop-blur-md"}`}>
      <div className="max-w-7xl mx-auto px-6 py-4">
        <div className="flex justify-between items-center">
          <div className="flex items-center gap-2 cursor-pointer" onClick={() => navigate("/")}>
            <div className="w-8 h-8 bg-gray-900 rounded flex items-center justify-center">
              <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            </div>
            <span className="text-xl font-bold text-gray-900">InterviewAI</span>
          </div>

          <div className="hidden md:flex items-center gap-8">
            <span
              onClick={() => navigate("/interview")}
              className="text-indigo-600 font-medium hover:text-indigo-700 cursor-pointer"
            >
              Interview Copilot
            </span>
            <span
              onClick={() => navigate("/pricing")}
              className="text-gray-700 font-medium hover:text-gray-900 cursor-pointer"
            >
              Pricing
            </span>
            
            <div className="relative">
              <span
                onClick={() => {
                  setFeaturesOpen(!featuresOpen);
                  setResourcesOpen(false);
                }}
                className="text-gray-700 font-medium hover:text-gray-900 cursor-pointer flex items-center gap-1"
              >
                Features
                <svg className={`w-4 h-4 transition-transform ${featuresOpen ? "rotate-180" : ""}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </span>
              <FeaturesDropdown isOpen={featuresOpen} onClose={() => setFeaturesOpen(false)} />
            </div>

            <div className="relative">
              <span
                onClick={() => {
                  setResourcesOpen(!resourcesOpen);
                  setFeaturesOpen(false);
                }}
                className="text-gray-700 font-medium hover:text-gray-900 cursor-pointer flex items-center gap-1"
              >
                Resources
                <svg className={`w-4 h-4 transition-transform ${resourcesOpen ? "rotate-180" : ""}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </span>
              <ResourcesDropdown isOpen={resourcesOpen} onClose={() => setResourcesOpen(false)} />
            </div>
          </div>

          {!user ? (
            <button
              onClick={() => navigate("/sign-in")}
              className="px-6 py-2.5 bg-indigo-600 text-white rounded-full font-semibold hover:bg-indigo-700 transition-colors"
            >
              Start for free
            </button>
          ) : (
            <UserMenu user={user} onSignOut={handleSignOut} />
          )}
        </div>
      </div>
    </nav>
  );
}