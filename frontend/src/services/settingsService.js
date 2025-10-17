import { supabase } from '../database/supabaseClient';


export const settingsService = {
  /**
   * Get user settings from database
   */
  async getUserSettings(userId) {
    try {
      const { data, error } = await supabase
        .from('copilot_settings')
        .select('*')
        .eq('user_id', userId)
        .single();

      if (error && error.code !== 'PGRST116') {
        throw error;
      }

      // Return default settings if none exist
      if (!data) {
        return {
          response_style: 'concise',
          audio_language: 'English',
          pause_interval: 2,
          advanced_question_detection: false,
          message_direction: 'bottom',
          auto_scroll: true,
          programming_language: 'Python',
          interview_instructions: '',
          coding_instructions: '',
          selected_response_style_id: null
        };
      }

      return data;
    } catch (error) {
      console.error('Error fetching settings:', error);
      throw error;
    }
  },

  /**
   * Create or update user settings
   */
  async saveUserSettings(userId, settings) {
    try {
      const { data, error } = await supabase
        .from('copilot_settings')
        .upsert({
          user_id: userId,
          response_style: settings.responseStyle || 'concise',
          audio_language: settings.audioLanguage || 'English',
          pause_interval: settings.pauseInterval || 2,
          advanced_question_detection: settings.advancedQuestionDetection || false,
          message_direction: settings.messageDirection || 'bottom',
          auto_scroll: settings.autoScroll !== false,
          programming_language: settings.programmingLanguage || 'Python',
          interview_instructions: settings.interviewInstructions || '',
          coding_instructions: settings.codingInstructions || '',
          selected_response_style_id: settings.selectedResponseStyleId || null
        })
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
   * Delete user settings
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
   * Load settings with fallback to localStorage
   */
  async loadSettingsWithFallback(userId) {
    try {
      // Try to get from database first
      const dbSettings = await this.getUserSettings(userId);
      
      // Convert snake_case to camelCase for frontend
      return {
        responseStyle: dbSettings.response_style,
        audioLanguage: dbSettings.audio_language,
        pauseInterval: dbSettings.pause_interval,
        advancedQuestionDetection: dbSettings.advanced_question_detection,
        messageDirection: dbSettings.message_direction,
        autoScroll: dbSettings.auto_scroll,
        programmingLanguage: dbSettings.programming_language,
        interviewInstructions: dbSettings.interview_instructions,
        codingInstructions: dbSettings.coding_instructions,
        selectedResponseStyleId: dbSettings.selected_response_style_id
      };
    } catch (error) {
      console.warn('Failed to load settings from database, using defaults:', error);
      return this.getDefaultSettings();
    }
  },

  /**
   * Get default settings
   */
  getDefaultSettings() {
    return {
      responseStyle: 'concise',
      audioLanguage: 'English',
      pauseInterval: 2,
      advancedQuestionDetection: false,
      messageDirection: 'bottom',
      autoScroll: true,
      programmingLanguage: 'Python',
      interviewInstructions: '',
      codingInstructions: '',
      selectedResponseStyleId: null
    };
  }
};
