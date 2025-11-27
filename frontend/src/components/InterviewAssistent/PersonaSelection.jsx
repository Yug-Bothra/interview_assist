import React, { useState, useEffect } from 'react';
import { Search, Plus, Loader2, AlertCircle } from 'lucide-react';
import { useAuth } from "../Auth/AuthContext";

import PersonaDetailsModal from './PersonaDetailsModal';

// GLOBAL CONTEXT
import { useAppData } from "../../context/AppDataContext";

export default function PersonaSelection({ onSelect }) {
  const { user } = useAuth();

  // Global data
  const { personas, loading, error } = useAppData();

  // Local UI
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedPersona, setSelectedPersona] = useState(null);
  const [showModal, setShowModal] = useState(false);

  // Filter personas
  const filteredPersonas = (personas || []).filter((persona) => {
    const term = searchTerm.toLowerCase();
    return (
      persona.company_name?.toLowerCase().includes(term) ||
      persona.position?.toLowerCase().includes(term)
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

  const getCompanyInitial = (companyName) =>
    companyName?.charAt(0).toUpperCase() || '?';

  // -----------------------------
  // LOADING STATE
  // -----------------------------
  if (loading) {
    return (
      <div className="bg-zinc-900 rounded-xl p-8 border border-zinc-800">
        <div className="flex flex-col items-center justify-center py-12 gap-4">
          <Loader2 className="w-10 h-10 text-white animate-spin" />
          <p className="text-gray-500">Loading personas...</p>
        </div>
      </div>
    );
  }

  // -----------------------------
  // ERROR / EMPTY STATE
  // -----------------------------
  if (!loading && error && personas.length === 0) {
    return (
      <div className="bg-zinc-900 rounded-xl p-8 border border-zinc-800">
        <div className="flex flex-col items-center justify-center py-12 gap-4">
          <AlertCircle className="w-12 h-12 text-yellow-500" />
          <p className="text-gray-500 text-center">{error}</p>

          <button
            onClick={() => window.location.href = '/personas'}
            className="mt-4 inline-flex items-center gap-2 bg-white hover:bg-gray-100 text-black font-medium py-2 px-6 rounded-lg transition-colors"
          >
            <Plus className="w-4 h-4" />
            Create Your First Persona
          </button>
        </div>
      </div>
    );
  }

  // -----------------------------
  // MAIN UI
  // -----------------------------
  return (
    <>
      <div className="bg-zinc-900 rounded-xl p-6 border border-zinc-800">
        {/* Header */}
        <div className="mb-6">
          <h2 className="text-white text-xl font-bold mb-2">Select a Persona</h2>
          <p className="text-gray-500 text-sm">
            Choose from your custom personas or pre-configured samples
          </p>
        </div>

        {/* Search */}
        <div className="relative mb-6">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-600 w-5 h-5" />
          <input
            type="text"
            placeholder="Search by company or position..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full bg-black border border-zinc-800 rounded-lg pl-10 pr-4 py-2.5 text-white text-sm placeholder-gray-600 focus:outline-none focus:border-zinc-700 transition-colors"
          />
        </div>

        {/* Personas List */}
        <div className="space-y-2 max-h-[600px] overflow-y-auto pr-2 custom-scrollbar">
          {filteredPersonas.length === 0 ? (
            <div className="text-center py-12">
              <p className="text-gray-500 mb-4">
                {searchTerm
                  ? "No personas match your search"
                  : "No personas found"}
              </p>

              {!searchTerm && (
                <button
                  onClick={() => window.location.href = '/personas'}
                  className="inline-flex items-center gap-2 bg-white hover:bg-gray-100 text-black font-medium py-2 px-4 rounded-lg transition-colors"
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
                className="w-full bg-black hover:bg-zinc-800 border border-zinc-800 hover:border-zinc-700 rounded-lg p-4 transition-all text-left group"
              >
                <div className="flex items-start gap-4">
                  <div
                    className={`w-12 h-12 rounded-lg flex items-center justify-center text-lg font-bold flex-shrink-0 ${
                      persona.is_sample
                        ? 'bg-zinc-800 text-gray-400'
                        : 'bg-white text-black'
                    }`}
                  >
                    {getCompanyInitial(persona.company_name)}
                  </div>

                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <h3 className="text-white font-medium truncate">
                        {persona.position}
                      </h3>

                      {persona.is_sample && (
                        <span className="inline-block px-2 py-0.5 bg-zinc-800 rounded text-xs text-gray-400 flex-shrink-0 font-medium">
                          Sample
                        </span>
                      )}
                    </div>

                    <p className="text-gray-500 text-sm truncate">
                      @{persona.company_name}
                    </p>

                    {persona.company_description && (
                      <p className="text-gray-600 text-xs mt-1 line-clamp-2">
                        {persona.company_description}
                      </p>
                    )}

                    {persona.resume_filename && (
                      <p className="text-emerald-400 text-xs mt-1 flex items-center gap-1">
                        <span className="w-1.5 h-1.5 bg-emerald-400 rounded-full"></span>
                        Resume attached
                      </p>
                    )}
                  </div>
                </div>
              </button>
            ))
          )}
        </div>

        {/* Manage button */}
        <div className="mt-6 pt-4 border-t border-zinc-800">
          <button
            onClick={() => window.location.href = '/personas'}
            className="w-full bg-zinc-800 hover:bg-zinc-700 text-white font-medium py-3 rounded-lg transition-colors flex items-center justify-center gap-2"
          >
            <Plus className="w-4 h-4" />
            Manage Personas
          </button>
        </div>
      </div>

      {/* Modal */}
      {showModal && selectedPersona && (
        <PersonaDetailsModal
          persona={selectedPersona}
          onClose={() => setShowModal(false)}
          onSelect={handleSelectPersona}
          showSelectButton={true}
        />
      )}

      {/* Scrollbar CSS */}
      <style>{`
        .custom-scrollbar::-webkit-scrollbar { width: 6px; }
        .custom-scrollbar::-webkit-scrollbar-track { background: #000; border-radius: 10px; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: #3f3f46; border-radius: 10px; }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover { background: #52525b; }
      `}</style>
    </>
  );
}
