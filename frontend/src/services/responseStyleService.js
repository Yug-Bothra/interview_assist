import { supabase } from '../database/supabaseClient';


export const responseStyleService = {
  /**
   * Get all available response styles (system + user's custom)
   */
  async getAllStyles(userId) {
    try {
      const { data, error } = await supabase
        .from('response_styles')
        .select('*')
        .or(`user_id.eq.${userId},is_system_default.eq.true`)
        .order('is_system_default', { ascending: false })
        .order('created_at', { ascending: false });

      if (error) throw error;
      return data || [];
    } catch (error) {
      console.error('Error fetching response styles:', error);
      throw error;
    }
  },

  /**
   * Get a specific style by ID
   */
  async getStyleById(styleId) {
    try {
      const { data, error } = await supabase
        .from('response_styles')
        .select('*')
        .eq('id', styleId)
        .single();

      if (error) throw error;
      return data;
    } catch (error) {
      console.error('Error fetching style:', error);
      throw error;
    }
  },

  /**
   * Create a new custom response style
   */
  async createStyle(userId, style) {
    try {
      const { data, error } = await supabase
        .from('response_styles')
        .insert({
          user_id: userId,
          style_name: style.styleName,
          description: style.description,
          approximate_length: style.approximateLength,
          tone: style.tone,
          example_response: style.exampleResponse,
          is_system_default: false
        })
        .select()
        .single();

      if (error) throw error;
      return data;
    } catch (error) {
      console.error('Error creating response style:', error);
      throw error;
    }
  },

  /**
   * Update an existing custom response style
   */
  async updateStyle(styleId, userId, updates) {
    try {
      const { data, error } = await supabase
        .from('response_styles')
        .update({
          style_name: updates.styleName,
          description: updates.description,
          approximate_length: updates.approximateLength,
          tone: updates.tone,
          example_response: updates.exampleResponse
        })
        .eq('id', styleId)
        .eq('user_id', userId)
        .eq('is_system_default', false)
        .select()
        .single();

      if (error) throw error;
      return data;
    } catch (error) {
      console.error('Error updating response style:', error);
      throw error;
    }
  },

  /**
   * Delete a custom response style
   */
  async deleteStyle(styleId, userId) {
    try {
      const { error } = await supabase
        .from('response_styles')
        .delete()
        .eq('id', styleId)
        .eq('user_id', userId)
        .eq('is_system_default', false);

      if (error) throw error;
    } catch (error) {
      console.error('Error deleting response style:', error);
      throw error;
    }
  }
};
