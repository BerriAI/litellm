import { StreamingResponse } from "../types";

export interface StreamProcessCallbacks {
  onContent: (content: string, model?: string) => void;
  onReasoningContent: (content: string) => void;
}

export const processStreamingResponse = (
  response: StreamingResponse, 
  callbacks: StreamProcessCallbacks
) => {
  // Extract model information if available
  const model = response.model;
  
  // Process regular content
  if (response.choices && response.choices.length > 0) {
    const choice = response.choices[0];
    
    if (choice.delta?.content) {
      callbacks.onContent(choice.delta.content, model);
    }
    
    // Process reasoning content if it exists
    if (choice.delta?.reasoning_content) {
      callbacks.onReasoningContent(choice.delta.reasoning_content);
    }
  }
}; 