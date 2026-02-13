import { PromptSpec } from "@/components/networking";
import { getVersionNumber } from "./prompt_editor_view/utils";

interface ModelGroupInfo {
  model_group: string;
  providers: string[];
  [key: string]: any;
}

/**
 * Extract template variables from prompt content
 */
export const extractTemplateVariables = (content?: string): Record<string, string> => {
  if (!content) return {};
  
  const variables: Record<string, string> = {};
  const variableRegex = /\{\{(\w+)\}\}/g;
  let match;
  while ((match = variableRegex.exec(content)) !== null) {
    const varName = match[1];
    if (!variables[varName]) {
      variables[varName] = `example_${varName}`;
    }
  }
  return variables;
};

/**
 * Get base prompt ID (stripped of version) from PromptSpec
 */
export const getBasePromptId = (promptData?: PromptSpec): string => {
  return promptData?.prompt_id || "";
};

/**
 * Get versioned prompt ID from litellm_params (preserves version)
 */
export const getVersionedPromptId = (promptData?: PromptSpec): string => {
  const baseId = getBasePromptId(promptData);
  const versionedId = (promptData?.litellm_params as any)?.prompt_id || baseId;
  return versionedId;
};

/**
 * Get current version number from prompt data
 */
export const getCurrentVersion = (promptData?: PromptSpec): string => {
  // Use explicit version field if available (from API response)
  if (promptData?.version) {
    return String(promptData.version);
  }
  
  // Fallback: extract from versioned ID in litellm_params
  const versionedId = getVersionedPromptId(promptData);
  return getVersionNumber(versionedId);
};

/**
 * Extract model from prompt litellm_params
 */
export const extractModel = (prompt: PromptSpec): string | null => {
  try {
    const params = prompt.litellm_params as any;
    
    // Try to extract from dotprompt_content
    if (params?.dotprompt_content) {
      const match = params.dotprompt_content.match(/model:\s*([^\n]+)/);
      if (match) return match[1].trim();
    }
    
    // Try to extract from prompt_data
    if (params?.prompt_data?.model) {
      return params.prompt_data.model;
    }
    
    // Try to extract model from litellm_params directly
    if (params?.model) {
      return params.model;
    }
    
    return null;
  } catch (error) {
    console.error("Error extracting model:", error);
    return null;
  }
};

/**
 * Get provider from model hub data
 */
export const getProviderFromModelHub = (
  modelName: string | null,
  modelHubData: Map<string, ModelGroupInfo>
): string | null => {
  if (!modelName) return null;
  
  const modelInfo = modelHubData.get(modelName);
  if (modelInfo && modelInfo.providers && modelInfo.providers.length > 0) {
    // Return the first provider from the list
    return modelInfo.providers[0];
  }
  
  return null;
};

