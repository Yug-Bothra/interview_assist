import React, { useState, useEffect } from 'react';
import { Search, Plus, X, Upload, FileText, Trash2, Loader2, AlertCircle, RefreshCw } from 'lucide-react';
import { getUserPersonas, createPersona, deletePersona, uploadResume, searchPersonas } from '../../database/personaService';
import { useAuth } from '../Auth/AuthContext';


export default function PersonasPage() {
  const { user, loading: authLoading } = useAuth();
  const [personas, setPersonas] = useState([]);
  const [filteredPersonas, setFilteredPersonas] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [showDetailsModal, setShowDetailsModal] = useState(false);
  const [selectedPersona, setSelectedPersona] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Form state
  const [formData, setFormData] = useState({
    companyName: '',
    companyDescription: '',
    position: '',
    jobDescription: '',
  });
  const [resumeFile, setResumeFile] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(false);

  // Load personas on component mount
  useEffect(() => {
    if (!authLoading && user) {
      loadPersonas();
    }
  }, [authLoading, user]);

  // Filter personas based on search
  useEffect(() => {
    if (searchTerm.trim()) {
      handleSearch();
    } else {
      setFilteredPersonas(personas);
    }
  }, [searchTerm, personas]);

  const loadPersonas = async () => {
    if (!user) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const result = await getUserPersonas();
      if (result.success) {
        setPersonas(result.data);
        setFilteredPersonas(result.data);
        console.log('Loaded personas:', result.data.length);
      } else {
        setError(result.error);
        console.error('Failed to load personas:', result.error);
      }
    } catch (err) {
      setError('An unexpected error occurred');
      console.error('Error loading personas:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = async () => {
    if (!searchTerm.trim()) {
      setFilteredPersonas(personas);
      return;
    }
    
    const result = await searchPersonas(searchTerm);
    if (result.success) {
      setFilteredPersonas(result.data);
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    if (file.type !== 'application/pdf') {
      alert('Please upload a PDF file');
      e.target.value = '';
      return;
    }
    
    if (file.size > 5 * 1024 * 1024) {
      alert('File size must be less than 5MB');
      e.target.value = '';
      return;
    }
    
    setResumeFile(file);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setUploadProgress(true);
    setError(null);

    try {
      let resumeUrl = '';
      let resumeFilename = '';
      let resumeFilePath = '';

      // Upload resume if provided
      if (resumeFile) {
        console.log('Uploading resume...');
        const uploadResult = await uploadResume(resumeFile);
        
        if (uploadResult.success) {
          resumeUrl = uploadResult.url;
          resumeFilename = uploadResult.filename;
          resumeFilePath = uploadResult.filePath;
          console.log('Resume uploaded successfully');
        } else {
          throw new Error(uploadResult.error || 'Failed to upload resume');
        }
      }

      // Create persona
      const personaData = {
        companyName: formData.companyName,
        companyDescription: formData.companyDescription,
        position: formData.position,
        jobDescription: formData.jobDescription,
        resumeUrl,
        resumeFilename,
        resumeFilePath,
      };

      console.log('Creating persona...');
      const result = await createPersona(personaData);
      
      if (result.success) {
        setShowCreateModal(false);
        resetForm();
        await loadPersonas();
        alert('Persona created successfully!');
      } else {
        throw new Error(result.error || 'Failed to create persona');
      }
    } catch (error) {
      console.error('Error in handleSubmit:', error);
      setError(error.message);
      alert('Error: ' + error.message);
    } finally {
      setUploadProgress(false);
    }
  };

  const handleDelete = async (personaId) => {
    if (!window.confirm('Are you sure you want to delete this persona?')) {
      return;
    }
    
    try {
      const result = await deletePersona(personaId);
      if (result.success) {
        await loadPersonas();
        alert('Persona deleted successfully');
      } else {
        throw new Error(result.error || 'Failed to delete persona');
      }
    } catch (error) {
      console.error('Error deleting persona:', error);
      alert('Error: ' + error.message);
    }
  };

  const resetForm = () => {
    setFormData({
      companyName: '',
      companyDescription: '',
      position: '',
      jobDescription: '',
    });
    setResumeFile(null);
    setError(null);
  };

  const openDetailsModal = (persona) => {
    setSelectedPersona(persona);
    setShowDetailsModal(true);
  };

  const getCompanyInitial = (companyName) => {
    return companyName?.charAt(0).toUpperCase() || '?';
  };

  if (authLoading) {
    return (
      <div className="min-h-screen bg-[#0a0a0f] flex items-center justify-center">
        <Loader2 className="w-10 h-10 text-purple-500 animate-spin" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#0a0a0f] p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-white text-3xl font-bold">Personas</h1>
            <p className="text-gray-400 text-sm mt-1">
              Manage your interview personas and resumes
            </p>
          </div>

          <div className="flex items-center gap-4">
            <div className="relative">
              <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
              <input
                type="text"
                placeholder="Search by name..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="bg-[#14141b] border border-gray-800 rounded-xl pl-12 pr-6 py-3 text-white placeholder-gray-500 focus:outline-none focus:border-purple-500/50 w-80"
              />
            </div>
            <button
              onClick={loadPersonas}
              disabled={loading}
              className="bg-gray-800 hover:bg-gray-700 text-white p-3 rounded-xl transition-all disabled:opacity-50"
              title="Refresh"
            >
              <RefreshCw className={`w-5 h-5 ${loading ? 'animate-spin' : ''}`} />
            </button>
            <button
              onClick={() => setShowCreateModal(true)}
              className="bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 text-white font-semibold px-6 py-3 rounded-xl transition-all flex items-center gap-2"
            >
              <Plus className="w-5 h-5" />
              Create New Persona
            </button>
          </div>
        </div>

        {/* Error Alert */}
        {error && (
          <div className="mb-6 bg-red-500/10 border border-red-500/30 rounded-xl p-4 flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
            <div className="flex-1">
              <p className="text-red-400 text-sm">{error}</p>
            </div>
            <button
              onClick={() => setError(null)}
              className="text-red-400 hover:text-red-300"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        )}

        {/* Personas Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          {/* New Persona Card */}
          <button
            onClick={() => setShowCreateModal(true)}
            className="bg-gradient-to-br from-gray-800/50 to-gray-900/50 hover:from-gray-800 hover:to-gray-900 border border-gray-700 hover:border-purple-500/50 rounded-xl p-8 transition-all"
          >
            <div className="flex flex-col items-center">
              <div className="w-24 h-24 bg-gray-700 rounded-2xl flex items-center justify-center mb-6">
                <Plus className="w-12 h-12 text-gray-400" />
              </div>
              <h3 className="text-white font-semibold text-lg mb-2">New Persona</h3>
              <p className="text-gray-400 text-sm">Create your own persona</p>
            </div>
          </button>

          {/* Persona Cards */}
          {loading ? (
            <div className="col-span-full flex items-center justify-center py-20">
              <div className="flex items-center gap-3 text-gray-400">
                <Loader2 className="w-6 h-6 animate-spin" />
                <span>Loading personas...</span>
              </div>
            </div>
          ) : filteredPersonas.length === 0 ? (
            <div className="col-span-full text-center text-gray-400 py-20">
              <p className="text-lg mb-2">No personas found</p>
              <p className="text-sm">Create your first persona to get started</p>
            </div>
          ) : (
            filteredPersonas.map((persona) => (
              <div
                key={persona.id}
                className="bg-gradient-to-br from-gray-800/50 to-gray-900/50 hover:from-gray-800 hover:to-gray-900 border border-gray-700 hover:border-purple-500/50 rounded-xl p-6 transition-all group relative"
              >
                <button
                  onClick={() => openDetailsModal(persona)}
                  className="w-full text-left"
                >
                  <div className="flex flex-col items-center mb-4">
                    <div className={`w-20 h-20 rounded-2xl flex items-center justify-center mb-4 text-3xl font-bold ${
                      persona.is_sample
                        ? 'bg-gradient-to-br from-purple-600/30 to-pink-600/30 text-white'
                        : 'bg-gradient-to-br from-purple-600 to-pink-600 text-white'
                    }`}>
                      {getCompanyInitial(persona.company_name)}
                    </div>
                    <h3 className="text-white font-semibold text-lg text-center">
                      {persona.company_name}
                    </h3>
                    <p className="text-gray-400 text-sm text-center mt-1">
                      {persona.position}
                    </p>
                    {persona.resume_filename && (
                      <div className="flex items-center gap-1 mt-3 text-xs text-green-400">
                        <FileText className="w-3 h-3" />
                        <span>Resume attached</span>
                      </div>
                    )}
                    {persona.is_sample && (
                      <div className="mt-2 px-2 py-1 bg-purple-500/20 rounded text-xs text-purple-400">
                        Sample
                      </div>
                    )}
                  </div>
                </button>
                {!persona.is_sample && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDelete(persona.id);
                    }}
                    className="absolute top-4 right-4 opacity-0 group-hover:opacity-100 transition-opacity text-red-400 hover:text-red-300 p-2 hover:bg-red-500/10 rounded-lg"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                )}
              </div>
            ))
          )}
        </div>
      </div>

      {/* Create Persona Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-4">
          <div className="bg-[#14141b] rounded-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto border border-gray-800">
            <div className="sticky top-0 bg-[#14141b] border-b border-gray-800 p-6 flex items-center justify-between z-10">
              <div>
                <h2 className="text-white text-2xl font-bold">Create a persona</h2>
                <p className="text-gray-400 text-sm mt-1">
                  Provide information about you and the interview to enable personalized support
                </p>
              </div>
              <button
                onClick={() => {
                  setShowCreateModal(false);
                  resetForm();
                }}
                className="text-gray-400 hover:text-white transition-colors"
              >
                <X className="w-6 h-6" />
              </button>
            </div>

            <form onSubmit={handleSubmit} className="p-6 space-y-6">
              {/* Resume Upload */}
              <div>
                <label className="text-white text-sm font-medium mb-2 block">
                  Select Resume <span className="text-gray-500">(Optional)</span>
                </label>
                <div className="flex items-center gap-3">
                  <div className="flex-1 bg-[#1a1a24] border border-gray-700 rounded-xl px-4 py-3 text-gray-400 overflow-hidden text-ellipsis whitespace-nowrap">
                    {resumeFile ? resumeFile.name : 'No resume uploaded'}
                  </div>
                  <label className="bg-gray-700 hover:bg-gray-600 text-white px-6 py-3 rounded-xl cursor-pointer transition-all flex items-center gap-2 flex-shrink-0">
                    <Upload className="w-5 h-5" />
                    Upload
                    <input
                      type="file"
                      accept=".pdf"
                      onChange={handleFileChange}
                      className="hidden"
                    />
                  </label>
                </div>
                <p className="text-gray-500 text-xs mt-2">PDF files only, max 5MB</p>
              </div>

              {/* Company Name */}
              <div>
                <label className="text-white text-sm font-medium mb-2 block">
                  Company Name <span className="text-red-400">*</span>
                </label>
                <input
                  type="text"
                  required
                  value={formData.companyName}
                  onChange={(e) => setFormData({ ...formData, companyName: e.target.value })}
                  placeholder="Enter the company name"
                  className="w-full bg-[#1a1a24] border border-gray-700 rounded-xl px-4 py-3 text-white placeholder-gray-500 focus:outline-none focus:border-purple-500/50"
                />
              </div>

              {/* Company Description */}
              <div>
                <label className="text-white text-sm font-medium mb-2 block">
                  Company Description <span className="text-red-400">*</span>
                </label>
                <textarea
                  required
                  value={formData.companyDescription}
                  onChange={(e) => setFormData({ ...formData, companyDescription: e.target.value })}
                  placeholder="Enter company information, including mission, cultural values, industry, and any other relevant details."
                  rows={4}
                  className="w-full bg-[#1a1a24] border border-gray-700 rounded-xl px-4 py-3 text-white placeholder-gray-500 focus:outline-none focus:border-purple-500/50 resize-none"
                />
              </div>

              {/* Position */}
              <div>
                <label className="text-white text-sm font-medium mb-2 block">
                  Position <span className="text-red-400">*</span>
                </label>
                <input
                  type="text"
                  required
                  value={formData.position}
                  onChange={(e) => setFormData({ ...formData, position: e.target.value })}
                  placeholder="Enter the job title"
                  className="w-full bg-[#1a1a24] border border-gray-700 rounded-xl px-4 py-3 text-white placeholder-gray-500 focus:outline-none focus:border-purple-500/50"
                />
              </div>

              {/* Job Description */}
              <div>
                <label className="text-white text-sm font-medium mb-2 block">
                  Job Description <span className="text-red-400">*</span>
                </label>
                <textarea
                  required
                  value={formData.jobDescription}
                  onChange={(e) => setFormData({ ...formData, jobDescription: e.target.value })}
                  placeholder="Enter job information, including qualifications, responsibilities, preferred skills, and any other relevant information"
                  rows={4}
                  className="w-full bg-[#1a1a24] border border-gray-700 rounded-xl px-4 py-3 text-white placeholder-gray-500 focus:outline-none focus:border-purple-500/50 resize-none"
                />
              </div>

              {/* Submit Button */}
              <button
                type="submit"
                disabled={uploadProgress}
                className="w-full bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 disabled:from-gray-700 disabled:to-gray-700 text-white font-semibold py-4 rounded-xl transition-all disabled:cursor-not-allowed flex items-center justify-center gap-2"
              >
                {uploadProgress && <Loader2 className="w-5 h-5 animate-spin" />}
                {uploadProgress ? 'Creating...' : 'Submit'}
              </button>
            </form>
          </div>
        </div>
      )}

      {/* Details Modal */}
      {showDetailsModal && selectedPersona && (
        <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-4">
          <div className="bg-[#14141b] rounded-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto border border-gray-800">
            <div className="sticky top-0 bg-[#14141b] border-b border-gray-800 p-6 z-10">
              <div className="flex items-start justify-between">
                <div className="flex items-start gap-4">
                  <div className={`w-16 h-16 rounded-xl flex items-center justify-center text-2xl text-white font-bold ${
                    selectedPersona.is_sample
                      ? 'bg-gradient-to-br from-purple-600/30 to-pink-600/30'
                      : 'bg-gradient-to-br from-purple-600 to-pink-600'
                  }`}>
                    {getCompanyInitial(selectedPersona.company_name)}
                  </div>
                  <div>
                    <h2 className="text-white text-2xl font-bold">{selectedPersona.position}</h2>
                    <p className="text-gray-400">@{selectedPersona.company_name}</p>
                    {selectedPersona.is_sample && (
                      <span className="inline-block mt-2 px-2 py-1 bg-purple-500/20 rounded text-xs text-purple-400">
                        Sample Persona
                      </span>
                    )}
                  </div>
                </div>
                <button
                  onClick={() => setShowDetailsModal(false)}
                  className="text-gray-400 hover:text-white transition-colors"
                >
                  <X className="w-6 h-6" />
                </button>
              </div>

              {selectedPersona.resume_filename && (
                <div className="mt-4 flex items-center gap-3 bg-[#1a1a24] border border-gray-700 rounded-xl p-3">
                  <FileText className="w-5 h-5 text-gray-400" />
                  <p className="text-white text-sm font-medium flex-1">Resume: {selectedPersona.resume_filename}</p>
                  {selectedPersona.resume_url && (
                    <a
                      href={selectedPersona.resume_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-purple-400 hover:text-purple-300 text-sm font-medium"
                    >
                      View
                    </a>
                  )}
                </div>
              )}
            </div>

            <div className="p-6 space-y-6">
              <div>
                <h3 className="text-white font-semibold mb-2">About the company</h3>
                <p className="text-gray-400 text-sm leading-relaxed">
                  {selectedPersona.company_description}
                </p>
              </div>

              <div>
                <h3 className="text-white font-semibold mb-2">About the role</h3>
                <p className="text-gray-400 text-sm leading-relaxed">
                  {selectedPersona.job_description}
                </p>
              </div>

              <div className="flex gap-3">
                <button
                  onClick={() => setShowDetailsModal(false)}
                  className="flex-1 bg-gray-700 hover:bg-gray-600 text-white font-semibold py-3 rounded-xl transition-all"
                >
                  Close
                </button>
                <button
                  onClick={() => {
                    localStorage.setItem('selectedPersona', `${selectedPersona.position}@${selectedPersona.company_name}`);
                    localStorage.setItem('selectedPersonaData', JSON.stringify(selectedPersona));
                    setShowDetailsModal(false);
                    alert('Persona selected! You can now proceed to the launchpad.');
                  }}
                  className="flex-1 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 text-white font-semibold py-3 rounded-xl transition-all"
                >
                  Select
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}