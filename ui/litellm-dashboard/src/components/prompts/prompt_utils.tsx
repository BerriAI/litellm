import { PromptSpec } from "@/components/networking";

interface ModelGroupInfo {
  model_group: string;
  providers: string[];
  [key: string]: any;
}

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

