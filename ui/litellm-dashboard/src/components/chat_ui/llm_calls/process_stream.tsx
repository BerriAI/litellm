import { TokenUsage } from "../ResponseMetrics";

export interface StreamingResponse {
  id: string;
  created: number;
  model: string;
  object: string;
  system_fingerprint?: string;
  choices: StreamingChoices[];
  provider_specific_fields?: any;
  stream_options?: any;
  citations?: any;
  usage?: Usage;
}

export interface StreamingChoices {
  finish_reason?: string | null;
  index: number;
  delta: Delta;
  logprobs?: any;
}

export interface Delta {
  content?: string;
  reasoning_content?: string;
  role?: string;
  function_call?: any;
  tool_calls?: any;
  audio?: any;
  refusal?: any;
  provider_specific_fields?: any;
}

export interface Usage {
  completion_tokens: number;
  prompt_tokens: number;
  total_tokens: number;
  completion_tokens_details?: {
    accepted_prediction_tokens?: number;
    audio_tokens?: number;
    reasoning_tokens?: number;
    rejected_prediction_tokens?: number;
    text_tokens?: number | null;
  };
  prompt_tokens_details?: {
    audio_tokens?: number;
    cached_tokens?: number;
    text_tokens?: number;
    image_tokens?: number;
  };
}

export interface StreamProcessCallbacks {
  onContent: (content: string, model?: string) => void;
  onReasoningContent: (content: string) => void;
  onUsage?: (usage: TokenUsage) => void;
}

export const processStreamingResponse = (response: StreamingResponse, callbacks: StreamProcessCallbacks) => {
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

  // Process usage information if it exists and we have a handler
  if (response.usage && callbacks.onUsage) {
    console.log("Processing usage data:", response.usage);
    const usageData: TokenUsage = {
      completionTokens: response.usage.completion_tokens,
      promptTokens: response.usage.prompt_tokens,
      totalTokens: response.usage.total_tokens,
    };

    // Extract reasoning tokens if available
    if (response.usage.completion_tokens_details?.reasoning_tokens) {
      usageData.reasoningTokens = response.usage.completion_tokens_details.reasoning_tokens;
    }

    callbacks.onUsage(usageData);
  }
};
