// frontend/src/database/personaService.js

import { supabase, RESUME_BUCKET } from './supabaseClient';

/**
 * Get all personas for current user (including sample personas)
 */
export const getUserPersonas = async () => {
  try {
    const { data: { user }, error: authError } = await supabase.auth.getUser();
    
    if (authError || !user) {
      return { success: false, error: 'Not authenticated', data: [] };
    }

    const { data, error } = await supabase
      .from('personas')
      .select('*')
      .or(`user_id.eq.${user.id},is_sample.eq.true`)
      .order('is_sample', { ascending: false })
      .order('created_at', { ascending: false });

    if (error) throw error;

    return { success: true, data: data || [] };
  } catch (error) {
    console.error('Error fetching personas:', error);
    return { success: false, error: error.message, data: [] };
  }
};

/**
 * Get only sample personas
 */
export const getSamplePersonas = async () => {
  try {
    const { data, error } = await supabase
      .from('personas')
      .select('*')
      .eq('is_sample', true)
      .order('created_at', { ascending: false });

    if (error) throw error;
    return { success: true, data: data || [] };
  } catch (error) {
    console.error('Error fetching sample personas:', error);
    return { success: false, error: error.message, data: [] };
  }
};

/**
 * Get only user's custom personas (excluding samples)
 */
export const getUserCustomPersonas = async () => {
  try {
    const { data: { user }, error: authError } = await supabase.auth.getUser();
    
    if (authError || !user) {
      return { success: false, error: 'Not authenticated', data: [] };
    }

    const { data, error } = await supabase
      .from('personas')
      .select('*')
      .eq('user_id', user.id)
      .eq('is_sample', false)
      .order('created_at', { ascending: false });

    if (error) throw error;
    return { success: true, data: data || [] };
  } catch (error) {
    console.error('Error fetching custom personas:', error);
    return { success: false, error: error.message, data: [] };
  }
};

/**
 * Search personas by company name or position
 */
export const searchPersonas = async (searchTerm) => {
  try {
    const { data: { user }, error: authError } = await supabase.auth.getUser();
    
    if (authError || !user) {
      return { success: false, error: 'Not authenticated', data: [] };
    }

    const { data, error } = await supabase
      .from('personas')
      .select('*')
      .or(`user_id.eq.${user.id},is_sample.eq.true`)
      .or(`company_name.ilike.%${searchTerm}%,position.ilike.%${searchTerm}%`)
      .order('created_at', { ascending: false });

    if (error) throw error;
    return { success: true, data: data || [] };
  } catch (error) {
    console.error('Error searching personas:', error);
    return { success: false, error: error.message, data: [] };
  }
};

/**
 * Get a single persona by ID
 */
export const getPersonaById = async (personaId) => {
  try {
    const { data, error } = await supabase
      .from('personas')
      .select('*')
      .eq('id', personaId)
      .single();

    if (error) throw error;
    return { success: true, data };
  } catch (error) {
    console.error('Error fetching persona:', error);
    return { success: false, error: error.message };
  }
};

/**
 * Upload resume to Supabase Storage
 */
export const uploadResume = async (file) => {
  try {
    const { data: { user }, error: authError } = await supabase.auth.getUser();
    
    if (authError || !user) {
      return { success: false, error: 'Not authenticated' };
    }

    if (!file || !(file instanceof File)) {
      throw new Error('Invalid file');
    }

    if (file.type !== 'application/pdf') {
      throw new Error('Only PDF files are allowed');
    }

    const maxSize = 5 * 1024 * 1024; // 5MB
    if (file.size > maxSize) {
      throw new Error('File size must be less than 5MB');
    }

    const timestamp = Date.now();
    const filename = `${user.id}/${timestamp}_${file.name}`;

    const { data, error } = await supabase.storage
      .from(RESUME_BUCKET)
      .upload(filename, file, { 
        cacheControl: '3600', 
        upsert: false 
      });

    if (error) throw error;

    const { data: publicUrlData } = supabase.storage
      .from(RESUME_BUCKET)
      .getPublicUrl(filename);

    return {
      success: true,
      url: publicUrlData.publicUrl,
      filename: file.name,
      filePath: filename,
    };
  } catch (error) {
    console.error('Error uploading resume:', error);
    return { success: false, error: error.message };
  }
};

/**
 * Delete resume from Supabase Storage
 */
export const deleteResume = async (filePath) => {
  try {
    if (!filePath) return { success: true };

    const { error } = await supabase.storage
      .from(RESUME_BUCKET)
      .remove([filePath]);

    if (error) throw error;
    return { success: true };
  } catch (error) {
    console.error('Error deleting resume:', error);
    return { success: false, error: error.message };
  }
};

/**
 * Create a new persona
 */
export const createPersona = async (personaData) => {
  try {
    const { data: { user }, error: authError } = await supabase.auth.getUser();
    
    if (authError || !user) {
      return { success: false, error: 'Not authenticated' };
    }

    const { data, error } = await supabase
      .from('personas')
      .insert([
        {
          user_id: user.id,
          company_name: personaData.companyName,
          company_description: personaData.companyDescription,
          position: personaData.position,
          job_description: personaData.jobDescription,
          resume_url: personaData.resumeUrl || null,
          resume_filename: personaData.resumeFilename || null,
          resume_file_path: personaData.resumeFilePath || null,
          is_sample: false,
        },
      ])
      .select();

    if (error) throw error;
    return { success: true, data: data[0] };
  } catch (error) {
    console.error('Error creating persona:', error);
    return { success: false, error: error.message };
  }
};

/**
 * Update an existing persona
 */
export const updatePersona = async (personaId, personaData) => {
  try {
    const { data, error } = await supabase
      .from('personas')
      .update({
        company_name: personaData.companyName,
        company_description: personaData.companyDescription,
        position: personaData.position,
        job_description: personaData.jobDescription,
        resume_url: personaData.resumeUrl || null,
        resume_filename: personaData.resumeFilename || null,
        resume_file_path: personaData.resumeFilePath || null,
      })
      .eq('id', personaId)
      .select();

    if (error) throw error;
    return { success: true, data: data[0] };
  } catch (error) {
    console.error('Error updating persona:', error);
    return { success: false, error: error.message };
  }
};

/**
 * Delete a persona (and its resume if exists)
 */
export const deletePersona = async (personaId) => {
  try {
    const { data: persona, error: fetchError } = await supabase
      .from('personas')
      .select('resume_file_path, is_sample')
      .eq('id', personaId)
      .single();

    if (fetchError) throw fetchError;

    if (persona?.is_sample) {
      throw new Error('Cannot delete sample personas');
    }

    if (persona?.resume_file_path) {
      await deleteResume(persona.resume_file_path);
    }

    const { error: deleteError } = await supabase
      .from('personas')
      .delete()
      .eq('id', personaId);

    if (deleteError) throw deleteError;
    return { success: true };
  } catch (error) {
    console.error('Error deleting persona:', error);
    return { success: false, error: error.message };
  }
};

/**
 * Get personas count for current user
 */
export const getUserPersonasCount = async () => {
  try {
    const { data: { user }, error: authError } = await supabase.auth.getUser();
    
    if (authError || !user) {
      return { success: false, error: 'Not authenticated', count: 0 };
    }

    const { count, error } = await supabase
      .from('personas')
      .select('*', { count: 'exact', head: true })
      .eq('user_id', user.id)
      .eq('is_sample', false);

    if (error) throw error;
    return { success: true, count: count || 0 };
  } catch (error) {
    console.error('Error counting personas:', error);
    return { success: false, error: error.message, count: 0 };
  }
};