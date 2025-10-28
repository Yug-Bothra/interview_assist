// frontend/src/components/InterviewAssistent/PersonaSelection.jsx

import React, { useState, useEffect } from 'react';
import { Search, Plus, Loader2, AlertCircle } from 'lucide-react';
import { useAuth } from "../Auth/AuthContext";
  
import { getUserPersonas } from '../../database/personaService';
import PersonaDetailsModal from './PersonaDetailsModal';

export default function PersonaSelection({ onSelect }) {
  const { user } = useAuth();
  const [personas, setPersonas] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedPersona, setSelectedPersona] = useState(null);
  const [showModal, setShowModal] = useState(false);

  useEffect(() => {
    if (user) {
      fetchPersonas();
    }
  }, [user]);

  const fetchPersonas = async () => {
    try {
      setLoading(true);
      setError(null);

      console.log('Fetching personas for user:', user.id);

      const result = await getUserPersonas();

      if (result.success) {
        console.log('Personas loaded:', result.data);
        setPersonas(result.data);

        if (result.data.length === 0) {
          setError('No personas found. Create your first persona to get started.');
        }
      } else {
        console.error('Failed to fetch personas:', result.error);
        setError(result.error || 'Failed to load personas');
      }
    } catch (error) {
      console.error('Error in fetchPersonas:', error);
      setError('An unexpected error occurred while loading personas');
    } finally {
      setLoading(false);
    }
  };

  const filteredPersonas = personas.filter(persona => {
    const searchLower = searchTerm.toLowerCase();
    return (
      persona.company_name?.toLowerCase().includes(searchLower) ||
      persona.position?.toLowerCase().includes(searchLower)
    );
  });

  const handlePersonaClick = (persona) => {
    setSelectedPersona(persona);
    setShowModal(true);
  };

  const handleSelectPersona = (persona) => {
    setShowModal(false);
    onSelect(persona.id, persona);
  };

  const getCompanyInitial = (companyName) => {
    return companyName?.charAt(0).toUpperCase() || '?';
  };

  if (loading) {
    return (
      <div className="bg-[#14141b] rounded-2xl p-8 border border-gray-800">
        <div className="flex flex-col items-center justify-center py-12 gap-4">
          <Loader2 className="w-10 h-10 text-purple-500 animate-spin" />
          <p className="text-gray-400">Loading personas...</p>
        </div>
      </div>
    );
  }

  if (error && personas.length === 0) {
    return (
      <div className="bg-[#14141b] rounded-2xl p-8 border border-gray-800">
        <div className="flex flex-col items-center justify-center py-12 gap-4">
          <AlertCircle className="w-12 h-12 text-yellow-500" />
          <p className="text-gray-400 text-center">{error}</p>
          <button
            onClick={() => window.location.href = '/personas'}
            className="mt-4 inline-flex items-center gap-2 bg-purple-600 hover:bg-purple-500 text-white font-semibold py-2 px-6 rounded-lg transition-all"
          >
            <Plus className="w-4 h-4" />
            Create Your First Persona
          </button>
        </div>
      </div>
    );
  }

  return (
    <>
      <div className="bg-[#14141b] rounded-2xl p-6 border border-gray-800">
        {/* Header */}
        <div className="mb-6">
          <h2 className="text-white text-xl font-bold mb-2">Select a Persona</h2>
          <p className="text-gray-400 text-sm">
            Choose from your custom personas or pre-configured samples
          </p>
        </div>

        {/* Search Bar */}
        <div className="relative mb-6">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
          <input
            type="text"
            placeholder="Search by company or position..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full bg-[#1a1a24] border border-gray-700 rounded-xl pl-10 pr-4 py-3 text-white placeholder-gray-500 focus:outline-none focus:border-purple-500 transition-colors"
          />
        </div>

        {/* Personas Grid */}
        <div className="space-y-3 max-h-[600px] overflow-y-auto pr-2 custom-scrollbar">
          {filteredPersonas.length === 0 ? (
            <div className="text-center py-12">
              <p className="text-gray-400 mb-4">
                {searchTerm ? 'No personas match your search' : 'No personas found'}
              </p>
              {!searchTerm && (
                <button
                  onClick={() => window.location.href = '/personas'}
                  className="inline-flex items-center gap-2 bg-purple-600 hover:bg-purple-500 text-white font-semibold py-2 px-4 rounded-lg transition-all"
                >
                  <Plus className="w-4 h-4" />
                  Create Your First Persona
                </button>
              )}
            </div>
          ) : (
            filteredPersonas.map((persona) => (
              <button
                key={persona.id}
                onClick={() => handlePersonaClick(persona)}
                className="w-full bg-[#1a1a24] hover:bg-gray-900/50 border border-gray-700 hover:border-purple-500/50 rounded-xl p-4 transition-all text-left group"
              >
                <div className="flex items-start gap-4">
                  <div
                    className={`w-12 h-12 rounded-lg flex items-center justify-center text-lg text-white font-bold flex-shrink-0 ${
                      persona.is_sample
                        ? 'bg-gradient-to-br from-purple-600/30 to-pink-600/30'
                        : 'bg-gradient-to-br from-purple-600 to-pink-600'
                    }`}
                  >
                    {getCompanyInitial(persona.company_name)}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <h3 className="text-white font-semibold truncate">
                        {persona.position}
                      </h3>
                      {persona.is_sample && (
                        <span className="inline-block px-2 py-0.5 bg-purple-500/20 rounded text-xs text-purple-400 flex-shrink-0">
                          Sample
                        </span>
                      )}
                    </div>
                    <p className="text-gray-400 text-sm truncate">
                      @{persona.company_name}
                    </p>
                    {persona.company_description && (
                      <p className="text-gray-500 text-xs mt-1 line-clamp-2">
                        {persona.company_description}
                      </p>
                    )}
                    {persona.resume_filename && (
                      <p className="text-green-400 text-xs mt-1 flex items-center gap-1">
                        <span className="w-1.5 h-1.5 bg-green-400 rounded-full"></span>
                        Resume attached
                      </p>
                    )}
                  </div>
                </div>
              </button>
            ))
          )}
        </div>

        {/* Action Button */}
        <div className="mt-6 pt-4 border-t border-gray-800">
          <button
            onClick={() => window.location.href = '/personas'}
            className="w-full bg-gray-800 hover:bg-gray-700 text-white font-semibold py-3 rounded-xl transition-all flex items-center justify-center gap-2"
          >
            <Plus className="w-4 h-4" />
            Manage Personas
          </button>
        </div>
      </div>

      {showModal && selectedPersona && (
        <PersonaDetailsModal
          persona={selectedPersona}
          onClose={() => setShowModal(false)}
          onSelect={handleSelectPersona}
          showSelectButton={true}
        />
      )}

      {/* Scrollbar styling */}
      <style>
        {`
          .custom-scrollbar::-webkit-scrollbar {
            width: 6px;
          }
          .custom-scrollbar::-webkit-scrollbar-track {
            background: #1a1a24;
            border-radius: 10px;
          }
          .custom-scrollbar::-webkit-scrollbar-thumb {
            background: #4a4a5a;
            border-radius: 10px;
          }
          .custom-scrollbar::-webkit-scrollbar-thumb:hover {
            background: #5a5a6a;
          }
        `}
      </style>
    </>
  );
}