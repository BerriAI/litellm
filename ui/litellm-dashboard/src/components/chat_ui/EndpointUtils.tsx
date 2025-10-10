import { ModelGroup } from "./llm_calls/fetch_models";
import { EndpointType, getEndpointType } from "./mode_endpoint_mapping";

/**
 * Determines the appropriate endpoint type based on the selected model
 *
 * @param selectedModel - The model identifier string
 * @param modelInfo - Array of model information
 * @returns The appropriate endpoint type
 */
export const determineEndpointType = (selectedModel: string, modelInfo: ModelGroup[]): EndpointType => {
  // Find the model information for the selected model
  const selectedModelInfo = modelInfo.find((option) => option.model_group === selectedModel);

  // If model info is found and it has a mode, determine the endpoint type
  if (selectedModelInfo?.mode) {
    return getEndpointType(selectedModelInfo.mode);
  }

  // Default to chat endpoint if no match is found
  return EndpointType.CHAT;
};
