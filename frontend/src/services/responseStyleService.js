import { supabase } from '../database/supabaseClient';

export const responseStyleService = {
  /**
   * Get all response styles (system + user-defined)
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
   * Auto "create or update" style.
   * - Creates new style
   * - If UNIQUE VIOLATION → updates instead
   * - If existing styleId provided → updates directly
   */
  async createOrUpdateStyle(userId, formData, existingId = null) {
    try {
      // If editing existing style → update directly
      if (existingId) {
        const { data, error } = await supabase
          .from('response_styles')
          .update({
            style_name: formData.styleName,
            description: formData.description,
            approximate_length: formData.approximateLength,
            tone: formData.tone,
            example_response: formData.exampleResponse,
          })
          .eq('id', existingId)
          .eq('user_id', userId)
          .eq('is_system_default', false)
          .select('id')
          .single();

        if (error) throw error;
        return data.id;
      }

      // Create new custom style
      const { data, error } = await supabase
        .from('response_styles')
        .insert({
          user_id: userId,
          style_name: formData.styleName,
          description: formData.description,
          approximate_length: formData.approximateLength,
          tone: formData.tone,
          example_response: formData.exampleResponse,
          is_system_default: false
        })
        .select('id')
        .single();

      // Handle UNIQUE CONSTRAINT (style_name + user_id)
      if (error) {
        if (error.code === '23505') {
          // → Update instead
          const update = await supabase
            .from('response_styles')
            .update({
              description: formData.description,
              approximate_length: formData.approximateLength,
              tone: formData.tone,
              example_response: formData.exampleResponse,
            })
            .eq('user_id', userId)
            .eq('style_name', formData.styleName)
            .select('id')
            .single();

          if (update.error) throw update.error;
          return update.data.id;
        }

        throw error;
      }

      return data.id;
    } catch (error) {
      console.error('Error saving style:', error);
      throw error;
    }
  },

  /**
   * Update direct wrapper
   */
  async updateStyle(styleId, userId, formData) {
    return this.createOrUpdateStyle(userId, formData, styleId);
  },

  /**
   * Delete a user-defined style
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
