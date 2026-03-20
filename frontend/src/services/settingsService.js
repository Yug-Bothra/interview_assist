import { supabase } from '../database/supabaseClient';

export const settingsService = {
  /**
   * Get user settings from database (single row)
   */
  async getUserSettings(userId) {
    try {
      const { data, error } = await supabase
        .from('copilot_settings')
        .select('*')
        .eq('user_id', userId)
        .single();

      // If NOT FOUND → return defaults
      if (error && error.code === 'PGRST116') {
        return this.getDefaultSettings();
      }

      if (error) throw error;

      return data;
    } catch (error) {
      console.error('Error fetching settings:', error);
      return this.getDefaultSettings();
    }
  },

  /**
   * Save (upsert) user settings
   */
  async saveUserSettings(userId, settings) {
    try {
      const payload = {
        user_id: userId,

        // Response style
        response_style: settings.responseStyle || 'concise',
        selected_response_style_id: settings.selectedResponseStyleId || null,

        // Audio transcription
        audio_language: settings.audioLanguage || 'English',
        pause_interval: settings.pauseInterval || 2,
        advanced_question_detection: settings.advancedQuestionDetection || false,

        // UI behavior
        message_direction: settings.messageDirection || 'bottom',
        auto_scroll: settings.autoScroll !== false,

        // Coding + interview instructions
        programming_language: settings.programmingLanguage || 'Python',
        interview_instructions: settings.interviewInstructions || '',
        coding_instructions: settings.codingInstructions || '',

        // Model settings (NEW)
        default_model: settings.defaultModel || 'gpt-4o',
        coding_model: settings.codingModel || 'gpt-4o',

        // All model providers (NEW) - Added all models including Ollama
        available_providers: settings.availableProviders || {
          'gpt-3.0-mini': true,
          'gpt-3.0': true,
          'gpt-3.1-mini': true,
          'gpt-3.1': true,
          'gpt-4.1-mini': true,
          'gpt-4.1': true,
          'gpt-4o-mini': true,
          'gpt-4o': true,
          'gpt-o1-mini': true,
          'gemini-1.5-flash': true,
          'gemini-1.5-pro': true,
          'gemini-2.0-flash': true,
          'gemini-2.0-flash-lite': true,
          'gemini-2.0-pro': true,
          'ollama': true
        }
      };

      const { data, error } = await supabase
        .from('copilot_settings')
        .upsert(payload, { onConflict: 'user_id' })
        .select()
        .single();

      if (error) throw error;
      return data;
    } catch (error) {
      console.error('Error saving settings:', error);
      throw error;
    }
  },

  /**
   * Delete all settings for user
   */
  async deleteUserSettings(userId) {
    try {
      const { error } = await supabase
        .from('copilot_settings')
        .delete()
        .eq('user_id', userId);

      if (error) throw error;
    } catch (error) {
      console.error('Error deleting settings:', error);
      throw error;
    }
  },

  /**
   * Load settings → fallback to defaults
   */
  async loadSettingsWithFallback(userId) {
    try {
      const dbSettings = await this.getUserSettings(userId);

      return {
        responseStyle: dbSettings.response_style,
        selectedResponseStyleId: dbSettings.selected_response_style_id,

        audioLanguage: dbSettings.audio_language,
        pauseInterval: dbSettings.pause_interval,
        advancedQuestionDetection: dbSettings.advanced_question_detection,

        messageDirection: dbSettings.message_direction,
        autoScroll: dbSettings.auto_scroll,

        programmingLanguage: dbSettings.programming_language,
        interviewInstructions: dbSettings.interview_instructions,
        codingInstructions: dbSettings.coding_instructions,

        // Models
        defaultModel: dbSettings.default_model,
        codingModel: dbSettings.coding_model,
        availableProviders: dbSettings.available_providers
      };
    } catch (error) {
      console.warn('Failed to load settings, using defaults:', error);
      return this.getDefaultSettings();
    }
  },

  /**
   * Default settings (no DB row yet)
   */
  getDefaultSettings() {
    return {
      responseStyle: 'concise',
      selectedResponseStyleId: null,

      audioLanguage: 'English',
      pauseInterval: 2,
      advancedQuestionDetection: false,

      messageDirection: 'bottom',
      autoScroll: true,

      programmingLanguage: 'Python',
      interviewInstructions: '',
      codingInstructions: '',

      defaultModel: 'gpt-4o',
      codingModel: 'gpt-4o',
      availableProviders: {
        'gpt-3.0-mini': true,
        'gpt-3.0': true,
        'gpt-3.1-mini': true,
        'gpt-3.1': true,
        'gpt-4.1-mini': true,
        'gpt-4.1': true,
        'gpt-4o-mini': true,
        'gpt-4o': true,
        'gpt-o1-mini': true,
        'gemini-1.5-flash': true,
        'gemini-1.5-pro': true,
        'gemini-2.0-flash': true,
        'gemini-2.0-flash-lite': true,
        'gemini-2.0-pro': true,
        'ollama': true
      }
    };
  }
};
