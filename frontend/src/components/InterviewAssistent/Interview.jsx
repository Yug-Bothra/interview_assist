import React, { useState } from 'react';
import { useAuth } from '../Auth/AuthContext';

import { useNavigate } from 'react-router-dom';
import Sidebar from './Sidebar';
import CopilotLaunchpad from './CopilotLaunchpad';
import PersonasPage from './PersonasPage';

export default function Interview() {
  const { user, loading } = useAuth();
  const navigate = useNavigate();
  const [currentPage, setCurrentPage] = useState('launchpad');

  // Redirect to sign-in if not authenticated
  React.useEffect(() => {
    if (!loading && !user) {
      navigate('/sign-in');
    }
  }, [user, loading, navigate]);

  // Show loading state while checking authentication
  if (loading) {
    return (
      <div className="min-h-screen bg-[#0a0a0f] flex items-center justify-center">
        <div className="text-white text-lg">Loading...</div>
      </div>
    );
  }

  // Don't render content if not signed in (will redirect)
  if (!user) {
    return null;
  }

  return (
    <div className="min-h-screen bg-[#0a0a0f] flex">
      <Sidebar currentPage={currentPage} onNavigate={setCurrentPage} />
      <main className="flex-1 overflow-auto">
        {currentPage === 'launchpad' && <CopilotLaunchpad />}
        {currentPage === 'personas' && <PersonasPage />}
        {currentPage === 'ai-mock' && <div className="p-8 text-gray-400">AI Mock - Coming Soon</div>}
        {currentPage === 'jobs' && <div className="p-8 text-gray-400">Jobs - Coming Soon</div>}
        {currentPage === 'questions' && <div className="p-8 text-gray-400">Questions - Coming Soon</div>}
        {currentPage === 'train' && <div className="p-8 text-gray-400">Train Copilot - Coming Soon</div>}
      </main>
    </div>
  );
}