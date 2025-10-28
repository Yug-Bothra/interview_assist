import React, { useState, useEffect } from 'react';
import { X, ChevronDown, Check, AlertCircle, Plus, Trash2, Edit2, Settings as SettingsIcon } from 'lucide-react';
import { useAuth } from '../Auth/AuthContext';
import { responseStyleService } from '../../services/responseStyleService';
import { settingsService } from '../../services/settingsService';

export default function SettingsModal({ onClose }) {
  const { user } = useAuth();
  const [activeTab, setActiveTab] = useState('responseStyle');
  
  // Models Tab
  const [defaultModel, setDefaultModel] = useState('gpt-4o');
  const [codingModel, setCodingModel] = useState('gpt-4o');
  const [availableProviders, setAvailableProviders] = useState({
    'gpt-4o': true,
    'gemini-2.0-flash': true
  });
  
  // Response Style Tab
  const [responseStyles, setResponseStyles] = useState([]);
  const [selectedStyleId, setSelectedStyleId] = useState(null);
  const [showStyleModal, setShowStyleModal] = useState(false);
  const [editingStyle, setEditingStyle] = useState(null);
  
  // Audio Transcription Tab
  const [audioLanguage, setAudioLanguage] = useState('English');
  const [pauseInterval, setPauseInterval] = useState(2);
  const [advancedQuestionDetection, setAdvancedQuestionDetection] = useState(false);
  
  // Advanced Settings Tab
  const [messageDirection, setMessageDirection] = useState('bottom');
  const [autoScroll, setAutoScroll] = useState(true);
  const [programmingLanguage, setProgrammingLanguage] = useState('Python');
  const [interviewInstructions, setInterviewInstructions] = useState('');
  const [codingInstructions, setCodingInstructions] = useState('');
  
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState({ type: '', text: '' });

  // Load initial data
  useEffect(() => {
    fetchInitialData();
  }, [user]);

  const fetchInitialData = async () => {
    if (!user) return;
    
    try {
      setLoading(true);

      // Fetch model status with error handling
      try {
        const modelResponse = await fetch('http://127.0.0.1:8000/api/models/status');
        if (modelResponse.ok) {
          const modelData = await modelResponse.json();
          setDefaultModel(modelData.default_provider || 'gpt-4o');
          setCodingModel(modelData.coding_provider || 'gpt-4o');
          setAvailableProviders(modelData.available_providers || {
            'gpt-4o': true,
            'gemini-2.0-flash': true
          });
        }
      } catch (error) {
        console.warn('Model API not available:', error);
      }

      // Fetch user settings
      const settings = await settingsService.loadSettingsWithFallback(user.id);
      setAudioLanguage(settings.audioLanguage);
      setPauseInterval(settings.pauseInterval);
      setAdvancedQuestionDetection(settings.advancedQuestionDetection);
      setMessageDirection(settings.messageDirection);
      setAutoScroll(settings.autoScroll);
      setProgrammingLanguage(settings.programmingLanguage);
      setInterviewInstructions(settings.interviewInstructions);
      setCodingInstructions(settings.codingInstructions);
      setSelectedStyleId(settings.selectedResponseStyleId);

      // Fetch response styles
      const styles = await responseStyleService.getAllStyles(user.id);
      setResponseStyles(styles);
      
      // If no style selected, select first available
      if (!settings.selectedResponseStyleId && styles.length > 0) {
        setSelectedStyleId(styles[0].id);
      }

      setLoading(false);
    } catch (error) {
      console.error('Error loading settings:', error);
      setMessage({ type: 'error', text: 'Failed to load settings' });
      setLoading(false);
    }
  };

  const handleSave = async () => {
    if (!user) return;
    
    setSaving(true);
    setMessage({ type: '', text: '' });

    try {
      // Update models with error handling
      try {
        await fetch('http://127.0.0.1:8000/api/models/set-default', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ provider: defaultModel })
        });

        await fetch('http://127.0.0.1:8000/api/models/set-coding', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ provider: codingModel })
        });
      } catch (error) {
        console.warn('Model API update failed:', error);
      }

      // Save all settings to database
      await settingsService.saveUserSettings(user.id, {
        audioLanguage,
        pauseInterval,
        advancedQuestionDetection,
        messageDirection,
        autoScroll,
        programmingLanguage,
        interviewInstructions,
        codingInstructions,
        selectedResponseStyleId: selectedStyleId
      });

      setMessage({ type: 'success', text: 'Settings saved successfully!' });
      setTimeout(() => {
        onClose();
      }, 1500);
    } catch (error) {
      console.error('Error saving settings:', error);
      setMessage({ type: 'error', text: 'Failed to save settings' });
    } finally {
      setSaving(false);
    }
  };

  const handleCreateStyle = () => {
    setEditingStyle(null);
    setShowStyleModal(true);
  };

  const handleEditStyle = (style) => {
    setEditingStyle(style);
    setShowStyleModal(true);
  };

  const handleDeleteStyle = async (styleId) => {
    if (!user || !confirm('Are you sure you want to delete this style?')) return;
    
    try {
      await responseStyleService.deleteStyle(styleId, user.id);
      const updatedStyles = responseStyles.filter(s => s.id !== styleId);
      setResponseStyles(updatedStyles);
      
      // If deleted style was selected, select first available
      if (selectedStyleId === styleId) {
        setSelectedStyleId(updatedStyles[0]?.id || null);
      }
      
      setMessage({ type: 'success', text: 'Style deleted successfully' });
    } catch (error) {
      console.error('Error deleting style:', error);
      setMessage({ type: 'error', text: 'Failed to delete style' });
    }
  };

  const handleStyleSaved = async () => {
    try {
      const styles = await responseStyleService.getAllStyles(user.id);
      setResponseStyles(styles);
      setShowStyleModal(false);
      setMessage({ type: 'success', text: 'Style saved successfully!' });
    } catch (error) {
      console.error('Error refreshing styles:', error);
    }
  };

  const modelOptions = [
    { value: 'gpt-4o', label: 'GPT-4o', available: availableProviders?.['gpt-4o'] !== false },
    { value: 'gemini-2.0-flash', label: 'Gemini 2.0 Flash', available: availableProviders?.['gemini-2.0-flash'] !== false }
  ];

  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-[#1a1a24] rounded-2xl w-full max-w-4xl max-h-[90vh] overflow-hidden border border-gray-800 shadow-2xl">
        {/* Header */}
        <div className="bg-[#14141b] px-6 py-4 flex items-center justify-between border-b border-gray-800">
          <h2 className="text-xl font-bold text-white">Copilot Parameters</h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white transition-colors p-1"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        {/* Tabs */}
        <div className="bg-[#14141b] px-6 border-b border-gray-800">
          <div className="flex gap-1">
            {['responseStyle', 'audioTranscription', 'models', 'advancedSettings'].map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`px-4 py-3 text-sm font-medium transition-colors border-b-2 ${
                  activeTab === tab
                    ? 'border-purple-500 text-white'
                    : 'border-transparent text-gray-400 hover:text-gray-300'
                }`}
              >
                {tab === 'responseStyle' && 'Response Style'}
                {tab === 'audioTranscription' && 'Audio Transcription'}
                {tab === 'models' && 'Models'}
                {tab === 'advancedSettings' && 'Advanced Settings'}
              </button>
            ))}
          </div>
        </div>

        {/* Content */}
        <div className="p-6 overflow-y-auto max-h-[calc(90vh-200px)]">
          {loading ? (
            <div className="text-center text-gray-400 py-8">Loading settings...</div>
          ) : (
            <>
              {/* Response Style Tab */}
              {activeTab === 'responseStyle' && (
                <ResponseStyleTab
                  responseStyles={responseStyles}
                  selectedStyleId={selectedStyleId}
                  setSelectedStyleId={setSelectedStyleId}
                  onCreateStyle={handleCreateStyle}
                  onEditStyle={handleEditStyle}
                  onDeleteStyle={handleDeleteStyle}
                />
              )}

              {/* Audio Transcription Tab */}
              {activeTab === 'audioTranscription' && (
                <AudioTranscriptionTab
                  audioLanguage={audioLanguage}
                  setAudioLanguage={setAudioLanguage}
                  pauseInterval={pauseInterval}
                  setPauseInterval={setPauseInterval}
                  advancedQuestionDetection={advancedQuestionDetection}
                  setAdvancedQuestionDetection={setAdvancedQuestionDetection}
                />
              )}

              {/* Models Tab */}
              {activeTab === 'models' && (
                <ModelsTab
                  defaultModel={defaultModel}
                  setDefaultModel={setDefaultModel}
                  codingModel={codingModel}
                  setCodingModel={setCodingModel}
                  modelOptions={modelOptions}
                />
              )}

              {/* Advanced Settings Tab */}
              {activeTab === 'advancedSettings' && (
                <AdvancedSettingsTab
                  messageDirection={messageDirection}
                  setMessageDirection={setMessageDirection}
                  autoScroll={autoScroll}
                  setAutoScroll={setAutoScroll}
                  programmingLanguage={programmingLanguage}
                  setProgrammingLanguage={setProgrammingLanguage}
                  interviewInstructions={interviewInstructions}
                  setInterviewInstructions={setInterviewInstructions}
                  codingInstructions={codingInstructions}
                  setCodingInstructions={setCodingInstructions}
                />
              )}
            </>
          )}

          {/* Status Message */}
          {message.text && (
            <div className={`mt-4 rounded-lg p-4 flex items-center gap-2 ${
              message.type === 'success' 
                ? 'bg-green-500/10 border border-green-500/30' 
                : 'bg-red-500/10 border border-red-500/30'
            }`}>
              {message.type === 'success' ? (
                <Check className="w-5 h-5 text-green-400" />
              ) : (
                <AlertCircle className="w-5 h-5 text-red-400" />
              )}
              <span className={message.type === 'success' ? 'text-green-400' : 'text-red-400'}>
                {message.text}
              </span>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="bg-[#14141b] px-6 py-4 flex items-center justify-end gap-3 border-t border-gray-800">
          <button
            onClick={onClose}
            className="px-6 py-2.5 bg-gray-800 hover:bg-gray-700 text-white rounded-lg font-medium transition-colors"
          >
            Close
          </button>
          <button
            onClick={handleSave}
            disabled={saving || loading}
            className="px-6 py-2.5 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 disabled:from-gray-700 disabled:to-gray-700 text-white font-semibold transition-all disabled:cursor-not-allowed rounded-lg"
          >
            {saving ? 'Saving...' : 'Save'}
          </button>
        </div>
      </div>

      {/* Style Customization Modal */}
      {showStyleModal && (
        <StyleCustomizationModal
          style={editingStyle}
          onClose={() => setShowStyleModal(false)}
          onSave={handleStyleSaved}
          userId={user?.id}
        />
      )}
    </div>
  );
}

// Response Style Tab Component
function ResponseStyleTab({ responseStyles, selectedStyleId, setSelectedStyleId, onCreateStyle, onEditStyle, onDeleteStyle }) {
  const selectedStyle = responseStyles.find(s => s.id === selectedStyleId);

  return (
    <div className="space-y-6">
      <div>
        <label className="text-white font-semibold mb-2 block">Copilot Style</label>
        <p className="text-gray-400 text-sm mb-4">
          Select how your AI assistant responds
        </p>
        
        {/* Style Selector */}
        <div className="relative mb-6">
          <select 
            value={selectedStyleId || ''}
            onChange={(e) => setSelectedStyleId(e.target.value)}
            className="w-full bg-gray-900 border border-gray-700 rounded-xl px-4 py-3 text-white appearance-none focus:outline-none focus:border-purple-500/50"
          >
            {responseStyles.length === 0 && (
              <option value="">No styles available</option>
            )}
            {responseStyles.map(style => (
              <option key={style.id} value={style.id}>
                {style.style_name} {style.is_system_default ? '(System)' : '(Custom)'}
              </option>
            ))}
          </select>
          <ChevronDown className="absolute right-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400 pointer-events-none" />
        </div>

        {/* Selected Style Details */}
        {selectedStyle && (
          <div className="bg-gray-900 rounded-xl p-6 border border-gray-700">
            <div className="flex items-start justify-between mb-4">
              <div className="flex-1">
                <h3 className="text-white font-semibold text-lg mb-2">{selectedStyle.style_name}</h3>
                <p className="text-gray-400 text-sm mb-4">{selectedStyle.description}</p>
                
                <div className="grid grid-cols-2 gap-4 mb-4">
                  <div>
                    <p className="text-gray-500 text-xs mb-1">Approximate Length</p>
                    <p className="text-gray-300 text-sm">{selectedStyle.approximate_length}</p>
                  </div>
                  <div>
                    <p className="text-gray-500 text-xs mb-1">Tone</p>
                    <p className="text-gray-300 text-sm">{selectedStyle.tone}</p>
                  </div>
                </div>
              </div>
              
              {!selectedStyle.is_system_default && (
                <div className="flex gap-2 ml-4">
                  <button
                    onClick={() => onEditStyle(selectedStyle)}
                    className="p-2 text-blue-400 hover:text-blue-300 hover:bg-blue-500/10 rounded-lg transition-colors"
                    title="Edit style"
                  >
                    <Edit2 className="w-4 h-4" />
                  </button>
                  <button
                    onClick={() => onDeleteStyle(selectedStyle.id)}
                    className="p-2 text-red-400 hover:text-red-300 hover:bg-red-500/10 rounded-lg transition-colors"
                    title="Delete style"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              )}
            </div>

            {/* Example Response */}
            <div>
              <p className="text-purple-400 text-sm font-medium mb-2">Example Response</p>
              <div className="bg-gray-800 rounded-lg p-4 max-h-64 overflow-y-auto">
                <p className="text-gray-300 text-sm whitespace-pre-wrap leading-relaxed">
                  {selectedStyle.example_response}
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Create Custom Style Button */}
        <button
          onClick={onCreateStyle}
          className="mt-6 w-full bg-purple-600/20 hover:bg-purple-600/30 border-2 border-purple-500/50 hover:border-purple-500 text-purple-300 font-semibold py-3 px-4 rounded-xl transition-all flex items-center justify-center gap-2"
        >
          <SettingsIcon className="w-5 h-5" />
          Style Customization
        </button>
      </div>
    </div>
  );
}

// Style Customization Modal
function StyleCustomizationModal({ style, onClose, onSave, userId }) {
  const [formData, setFormData] = useState({
    styleName: style?.style_name || '',
    description: style?.description || '',
    approximateLength: style?.approximate_length || '~150 words',
    tone: style?.tone || 'Professional',
    exampleResponse: style?.example_response || ''
  });
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!formData.styleName || !formData.description || !formData.exampleResponse) {
      setError('Please fill in all required fields');
      return;
    }

    setSaving(true);
    setError('');

    try {
      if (style) {
        // Update existing style
        await responseStyleService.updateStyle(style.id, userId, formData);
      } else {
        // Create new style
        await responseStyleService.createStyle(userId, formData);
      }
      
      onSave();
    } catch (err) {
      setError('Failed to save style. Please try again.');
      console.error(err);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-[60] flex items-center justify-center p-4">
      <div className="bg-[#1a1a24] rounded-2xl w-full max-w-3xl max-h-[90vh] overflow-hidden border border-gray-800">
        <div className="bg-[#14141b] px-6 py-4 flex items-center justify-between border-b border-gray-800">
          <h3 className="text-xl font-bold text-white">
            {style ? 'Edit Response Style' : 'Create Custom Style'}
          </h3>
          <button onClick={onClose} className="text-gray-400 hover:text-white transition-colors">
            <X className="w-6 h-6" />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="p-6 overflow-y-auto max-h-[calc(90vh-140px)]">
          <div className="space-y-5">
            {/* Style Name */}
            <div>
              <label className="text-white font-medium mb-2 block">
                Style Name <span className="text-red-400">*</span>
              </label>
              <input
                type="text"
                value={formData.styleName}
                onChange={(e) => setFormData({...formData, styleName: e.target.value})}
                placeholder="e.g., Brief Professional, Storyteller, Technical Expert"
                className="w-full bg-gray-800 text-white px-4 py-3 rounded-lg border border-gray-700 focus:outline-none focus:ring-2 focus:ring-purple-500"
              />
            </div>

            {/* Description */}
            <div>
              <label className="text-white font-medium mb-2 block">
                Description <span className="text-red-400">*</span>
              </label>
              <input
                type="text"
                value={formData.description}
                onChange={(e) => setFormData({...formData, description: e.target.value})}
                placeholder="Brief description of this response style"
                className="w-full bg-gray-800 text-white px-4 py-3 rounded-lg border border-gray-700 focus:outline-none focus:ring-2 focus:ring-purple-500"
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              {/* Approximate Length */}
              <div>
                <label className="text-white font-medium mb-2 block">Approximate Length</label>
                <input
                  type="text"
                  value={formData.approximateLength}
                  onChange={(e) => setFormData({...formData, approximateLength: e.target.value})}
                  placeholder="e.g., ~150 words"
                  className="w-full bg-gray-800 text-white px-4 py-3 rounded-lg border border-gray-700 focus:outline-none focus:ring-2 focus:ring-purple-500"
                />
              </div>

              {/* Tone */}
              <div>
                <label className="text-white font-medium mb-2 block">Tone</label>
                <select
                  value={formData.tone}
                  onChange={(e) => setFormData({...formData, tone: e.target.value})}
                  className="w-full bg-gray-800 text-white px-4 py-3 rounded-lg border border-gray-700 focus:outline-none focus:ring-2 focus:ring-purple-500"
                >
                  <option>Professional</option>
                  <option>Casual</option>
                  <option>Friendly</option>
                  <option>Formal</option>
                  <option>Conversational</option>
                </select>
              </div>
            </div>

            {/* Example Response */}
            <div>
              <label className="text-white font-medium mb-2 block">
                Example Response <span className="text-red-400">*</span>
              </label>
              <p className="text-gray-400 text-sm mb-2">
                Provide an example answer in this style to help the AI understand your preferences
              </p>
              <textarea
                value={formData.exampleResponse}
                onChange={(e) => setFormData({...formData, exampleResponse: e.target.value})}
                placeholder="Write a sample response that demonstrates this style..."
                className="w-full bg-gray-800 text-white px-4 py-3 rounded-lg border border-gray-700 focus:outline-none focus:ring-2 focus:ring-purple-500 min-h-[200px] resize-none"
              />
              <p className="text-gray-500 text-xs mt-1">
                {formData.exampleResponse.length} characters
              </p>
            </div>

            {error && (
              <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-3">
                <p className="text-red-400 text-sm">{error}</p>
              </div>
            )}
          </div>

          <div className="flex gap-3 mt-6 pt-6 border-t border-gray-800">
            <button
              type="button"
              onClick={onClose}
              className="flex-1 px-6 py-3 bg-gray-800 hover:bg-gray-700 text-white rounded-lg font-medium transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={saving}
              className="flex-1 px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 disabled:from-gray-700 disabled:to-gray-700 text-white font-semibold rounded-lg transition-all disabled:cursor-not-allowed"
            >
              {saving ? 'Saving...' : style ? 'Update Style' : 'Create Style'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

// Audio Transcription Tab Component
function AudioTranscriptionTab({ audioLanguage, setAudioLanguage, pauseInterval, setPauseInterval, advancedQuestionDetection, setAdvancedQuestionDetection }) {
  return (
    <div className="space-y-6">
      <div>
        <label className="text-white font-medium mb-2 block">Language</label>
        <p className="text-gray-400 text-sm mb-3">Language of the interview</p>
        <select
          value={audioLanguage}
          onChange={(e) => setAudioLanguage(e.target.value)}
          className="w-full bg-gray-800 text-white px-4 py-3 rounded-lg border border-gray-700 focus:outline-none focus:ring-2 focus:ring-purple-500"
        >
          <option>English</option>
          <option>Spanish</option>
          <option>French</option>
          <option>German</option>
          <option>Hindi</option>
          <option>Mandarin</option>
        </select>
      </div>

      <div>
        <label className="text-white font-medium mb-2 block">Pause Interval</label>
        <p className="text-gray-400 text-sm mb-3">
          Duration of pause to detect the end of speech and generate a new transcription paragraph (seconds)
        </p>
        <div className="flex items-center gap-4">
          <input
            type="number"
            min="1"
            max="10"
            value={pauseInterval}
            onChange={(e) => setPauseInterval(Number(e.target.value))}
            className="flex-1 bg-gray-800 text-white px-4 py-3 rounded-lg border border-gray-700 focus:outline-none focus:ring-2 focus:ring-purple-500"
          />
          <span className="text-gray-400">sec</span>
        </div>
      </div>

      <div>
        <div className="flex items-center justify-between mb-3">
          <div>
            <label className="text-white font-medium block">Advanced Question Detection</label>
            <span className="bg-purple-600/20 text-purple-300 text-xs px-2 py-1 rounded-md inline-block mt-1">
              Experimental
            </span>
          </div>
          <label className="relative inline-flex items-center cursor-pointer">
            <input
              type="checkbox"
              checked={advancedQuestionDetection}
              onChange={(e) => setAdvancedQuestionDetection(e.target.checked)}
              className="sr-only peer"
            />
            <div className="w-11 h-6 bg-gray-700 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-purple-800 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-purple-600"></div>
          </label>
        </div>
        <p className="text-gray-400 text-sm">
          Intelligently filter conversation to focus on meaningful questions
        </p>
      </div>
    </div>
  );
}

// Models Tab Component
function ModelsTab({ defaultModel, setDefaultModel, codingModel, setCodingModel, modelOptions }) {
  return (
    <div className="space-y-6">
      <div>
        <label className="text-white font-semibold mb-2 block">Default Copilot Model</label>
        <p className="text-gray-400 text-sm mb-4">
          Select the AI model for general copilot interactions. Automatic fallback enabled if primary model fails.
        </p>
        <div className="relative">
          <select 
            value={defaultModel}
            onChange={(e) => setDefaultModel(e.target.value)}
            className="w-full bg-gray-900 border border-gray-700 rounded-xl px-4 py-3 text-white appearance-none focus:outline-none focus:border-purple-500/50"
          >
            {modelOptions.map(option => (
              <option 
                key={option.value} 
                value={option.value}
                disabled={!option.available}
              >
                {option.label} {!option.available ? '(Not Available)' : ''}
              </option>
            ))}
          </select>
          <ChevronDown className="absolute right-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400 pointer-events-none" />
        </div>
      </div>

      <div>
        <label className="text-white font-semibold mb-2 block">Coding Assistant Model</label>
        <p className="text-gray-400 text-sm mb-4">
          Select the AI model for coding-related tasks and technical questions
        </p>
        <div className="relative">
          <select 
            value={codingModel}
            onChange={(e) => setCodingModel(e.target.value)}
            className="w-full bg-gray-900 border border-gray-700 rounded-xl px-4 py-3 text-white appearance-none focus:outline-none focus:border-purple-500/50"
          >
            {modelOptions.map(option => (
              <option 
                key={option.value} 
                value={option.value}
                disabled={!option.available}
              >
                {option.label} {!option.available ? '(Not Available)' : ''}
              </option>
            ))}
          </select>
          <ChevronDown className="absolute right-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400 pointer-events-none" />
        </div>
      </div>

      <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
        <div className="flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-blue-400 flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-blue-400 text-sm font-medium mb-1">Automatic Fallback Enabled</p>
            <p className="text-blue-300/80 text-xs">
              If your primary model fails or is unavailable, the system will automatically switch to the backup model to ensure uninterrupted service.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

// Advanced Settings Tab Component
function AdvancedSettingsTab({ 
  messageDirection, 
  setMessageDirection, 
  autoScroll, 
  setAutoScroll, 
  programmingLanguage, 
  setProgrammingLanguage,
  interviewInstructions,
  setInterviewInstructions,
  codingInstructions,
  setCodingInstructions
}) {
  return (
    <div className="space-y-6">
      <div>
        <div className="flex items-center justify-between mb-3">
          <label className="text-white font-medium">Message Direction</label>
          <label className="relative inline-flex items-center cursor-pointer">
            <input
              type="checkbox"
              checked={messageDirection === 'bottom'}
              onChange={(e) => setMessageDirection(e.target.checked ? 'bottom' : 'top')}
              className="sr-only peer"
            />
            <div className="w-11 h-6 bg-gray-700 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-purple-800 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-purple-600"></div>
          </label>
        </div>
        <p className="text-gray-400 text-sm">
          Newer Interview Copilot messages appear at the {messageDirection}
        </p>
      </div>

      <div>
        <div className="flex items-center justify-between mb-3">
          <label className="text-white font-medium">Auto Scroll</label>
          <label className="relative inline-flex items-center cursor-pointer">
            <input
              type="checkbox"
              checked={autoScroll}
              onChange={(e) => setAutoScroll(e.target.checked)}
              className="sr-only peer"
            />
            <div className="w-11 h-6 bg-gray-700 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-purple-800 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-purple-600"></div>
          </label>
        </div>
        <p className="text-gray-400 text-sm">
          Enable auto scroll by default
        </p>
      </div>

      <div>
        <label className="text-white font-medium mb-2 block">Programming Language</label>
        <p className="text-gray-400 text-sm mb-3">Default Programming Language</p>
        <select
          value={programmingLanguage}
          onChange={(e) => setProgrammingLanguage(e.target.value)}
          className="w-full bg-gray-800 text-white px-4 py-3 rounded-lg border border-gray-700 focus:outline-none focus:ring-2 focus:ring-purple-500"
        >
          <option>Python</option>
          <option>JavaScript</option>
          <option>Java</option>
          <option>C++</option>
          <option>Go</option>
          <option>Rust</option>
          <option>TypeScript</option>
          <option>C#</option>
        </select>
      </div>

      <div>
        <label className="text-white font-medium mb-2 block">
          Additional Instruction (Interview Copilot)
        </label>
        <textarea
          value={interviewInstructions}
          onChange={(e) => setInterviewInstructions(e.target.value)}
          placeholder="Tell AI what to do or not to do (e.g. 'Don't use jargon' or 'Use most recent work experience')"
          className="w-full bg-gray-800 text-white px-4 py-3 rounded-lg border border-gray-700 focus:outline-none focus:ring-2 focus:ring-purple-500 min-h-[100px] resize-none"
        />
      </div>

      <div>
        <label className="text-white font-medium mb-2 block">
          Additional Instruction (Coding Copilot)
        </label>
        <textarea
          value={codingInstructions}
          onChange={(e) => setCodingInstructions(e.target.value)}
          placeholder="Instructions for coding solutions (e.g. 'Prefer clear code over optimized code' or 'Include detailed comments')"
          className="w-full bg-gray-800 text-white px-4 py-3 rounded-lg border border-gray-700 focus:outline-none focus:ring-2 focus:ring-purple-500 min-h-[100px] resize-none"
        />
      </div>
    </div>
  );
}