import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Settings, HelpCircle, Rocket, CheckCircle } from 'lucide-react';
import PersonaSelection from './PersonaSelection';
import InterviewDomainSelection from './InterviewDomainSelection';
import LaunchChecklist from './LaunchChecklist';
import SettingsModal from './SettingsModal';

export default function CopilotLaunchpad() {
  const navigate = useNavigate();
  const [step, setStep] = useState(1);
  const [selectedPersona, setSelectedPersona] = useState(null);
  const [selectedPersonaData, setSelectedPersonaData] = useState(null);
  const [selectedDomain, setSelectedDomain] = useState(null);
  const [showSettings, setShowSettings] = useState(false);
  const [processingResume, setProcessingResume] = useState(false);

  const handlePersonaSelect = async (personaId, personaData) => {
    setSelectedPersona(personaId);
    setSelectedPersonaData(personaData);
    
    if (personaData.resume_url && !personaData.resume_text) {
      setProcessingResume(true);
      try {
        const response = await fetch(`http://127.0.0.1:8000/api/persona/${personaId}`);
        const result = await response.json();
        
        if (result.resume_text) {
          console.log('Resume processed successfully');
          setSelectedPersonaData(result);
        } else {
          console.warn('Resume processing incomplete');
        }
      } catch (error) {
        console.error('Error processing resume:', error);
      } finally {
        setProcessingResume(false);
      }
    }
    
    setStep(2);
  };

  const handleDomainSelect = (domain) => {
    setSelectedDomain(domain);
    setStep(3);
  };

  const handleLaunch = () => {
    if (!selectedPersona || !selectedDomain) {
      alert('Please complete all steps before launching');
      return;
    }

    sessionStorage.setItem('launchAuthorized', 'true');
    navigate('/interview-assist', {
      state: {
        personaId: selectedPersona,
        personaData: selectedPersonaData,
        domain: selectedDomain
      },
    });
  };

  const handleCreatePersona = () => {
    navigate('/personas');
  };

  const handleEditStep = (stepNumber) => {
    if (stepNumber <= step) {
      setStep(stepNumber);
      if (stepNumber === 1) {
        setSelectedPersona(null);
        setSelectedPersonaData(null);
        setSelectedDomain(null);
      } else if (stepNumber === 2) {
        setSelectedDomain(null);
      }
    }
  };

  return (
    <div className="min-h-screen bg-[#0a0a0f] p-8">
      <div className="max-w-7xl mx-auto">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left Panel */}
          <div className="bg-[#14141b] rounded-2xl p-8 border border-gray-800">
            {/* Header */}
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-3">
                <h1 className="text-white text-2xl font-bold">Copilot Launchpad</h1>
                <button className="text-gray-400 hover:text-white transition-colors">
                  <HelpCircle className="w-5 h-5" />
                </button>
              </div>
              <button
                onClick={() => setShowSettings(true)}
                className="text-gray-400 hover:text-white transition-colors p-2 hover:bg-gray-800/50 rounded-lg"
              >
                <Settings className="w-5 h-5" />
              </button>
            </div>

            {/* Step 1 */}
            <div className={`mb-6 ${step > 1 ? 'opacity-60' : ''}`}>
              <div className="flex items-start gap-4">
                <div
                  className={`flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center ${
                    step > 1 ? 'bg-purple-500' : 'bg-gray-800 border-2 border-purple-500'
                  }`}
                >
                  {step > 1 ? (
                    <CheckCircle className="w-5 h-5 text-white" />
                  ) : (
                    <span className="text-white font-semibold">1</span>
                  )}
                </div>
                <div className="flex-1">
                  <div className="flex items-center justify-between">
                    <h3 className="text-white font-semibold mb-1">Select Persona</h3>
                    {step > 1 && (
                      <button
                        onClick={() => handleEditStep(1)}
                        className="text-purple-400 hover:text-purple-300 text-sm"
                      >
                        Edit
                      </button>
                    )}
                  </div>
                  <p className="text-gray-400 text-sm">
                    Choose a persona that describes the company and role of your interview
                  </p>
                  {step === 1 && (
                    <button
                      onClick={handleCreatePersona}
                      className="mt-4 bg-purple-600 hover:bg-purple-500 text-white font-semibold py-2 px-4 rounded-lg transition-all"
                    >
                      + Create Your Own Persona
                    </button>
                  )}
                  {step > 1 && selectedPersonaData && (
                    <div className="mt-3 bg-gray-800/50 rounded-lg p-3 border border-gray-700">
                      <p className="text-white text-sm font-medium">
                        {selectedPersonaData.position} @ {selectedPersonaData.company_name}
                      </p>
                      {selectedPersonaData.company_description && (
                        <p className="text-gray-400 text-xs mt-1 line-clamp-2">
                          {selectedPersonaData.company_description.substring(0, 80)}...
                        </p>
                      )}
                      {selectedPersonaData.resume_text && (
                        <p className="text-green-400 text-xs mt-1">✓ Resume context loaded</p>
                      )}
                      {processingResume && (
                        <p className="text-yellow-400 text-xs mt-1">⏳ Processing resume...</p>
                      )}
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Step 2 */}
            <div className={`mb-6 ${step < 2 ? 'opacity-40' : step > 2 ? 'opacity-60' : ''}`}>
              <div className="flex items-start gap-4">
                <div
                  className={`flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center ${
                    step > 2
                      ? 'bg-purple-500'
                      : step === 2
                      ? 'bg-gray-800 border-2 border-purple-500'
                      : 'bg-gray-800'
                  }`}
                >
                  {step > 2 ? (
                    <CheckCircle className="w-5 h-5 text-white" />
                  ) : (
                    <span className="text-white font-semibold">2</span>
                  )}
                </div>
                <div className="flex-1">
                  <div className="flex items-center justify-between">
                    <h3 className="text-white font-semibold mb-1">Select Interview Domain</h3>
                    {step > 2 && (
                      <button
                        onClick={() => handleEditStep(2)}
                        className="text-purple-400 hover:text-purple-300 text-sm"
                      >
                        Edit
                      </button>
                    )}
                  </div>
                  <p className="text-gray-400 text-sm">
                    Choose a domain for industry-specific responses
                  </p>
                  {step > 2 && selectedDomain && (
                    <div className="mt-3 bg-gray-800/50 rounded-lg p-3 border border-gray-700">
                      <p className="text-white text-sm font-medium">{selectedDomain}</p>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Step 3 */}
            <div className={`mb-8 ${step < 3 ? 'opacity-40' : ''}`}>
              <div className="flex items-start gap-4">
                <div
                  className={`flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center ${
                    step === 3 ? 'bg-gray-800 border-2 border-purple-500' : 'bg-gray-800'
                  }`}
                >
                  <span className="text-white font-semibold">3</span>
                </div>
                <div className="flex-1">
                  <h3 className="text-white font-semibold mb-1">Confirm Launch Settings</h3>
                  <p className="text-gray-400 text-sm">Review selections and system settings</p>
                  {step === 3 && (
                    <div className="mt-3 bg-green-500/10 rounded-lg p-3 border border-green-500/30">
                      <p className="text-green-400 text-sm font-medium">✓ Ready to launch</p>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Launch Button */}
            <button
              onClick={handleLaunch}
              disabled={step < 3 || !selectedPersona || !selectedDomain || processingResume}
              className="w-full bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 disabled:from-gray-700 disabled:to-gray-700 text-white font-semibold py-4 rounded-xl transition-all disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              <Rocket className="w-5 h-5" />
              {processingResume 
                ? 'Processing resume...' 
                : step < 3 
                ? 'Complete all steps to launch' 
                : 'Launch Interview Copilot'}
            </button>
          </div>

          {/* Right Panel */}
          <div>
            {step === 1 && <PersonaSelection onSelect={handlePersonaSelect} />}
            {step === 2 && <InterviewDomainSelection onSelect={handleDomainSelect} />}
            {step === 3 && (
              <LaunchChecklist
                persona={selectedPersona}
                personaData={selectedPersonaData}
                domain={selectedDomain}
              />
            )}
          </div>
        </div>
      </div>

      {/* Settings Modal */}
      {showSettings && (
        <SettingsModal onClose={() => setShowSettings(false)} />
      )}
    </div>
  );
}