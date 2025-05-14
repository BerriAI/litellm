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

export interface CompletionTokensDetails {
  accepted_prediction_tokens?: number;
  audio_tokens?: number;
  reasoning_tokens?: number;
  rejected_prediction_tokens?: number;
  text_tokens?: number | null;
}

export interface PromptTokensDetails {
  audio_tokens?: number;
  cached_tokens?: number;
  text_tokens?: number;
  image_tokens?: number;
}

export interface Usage {
  completion_tokens: number;
  prompt_tokens: number;
  total_tokens: number;
  completion_tokens_details?: CompletionTokensDetails;
  prompt_tokens_details?: PromptTokensDetails;
}

export interface StreamingChoices {
  finish_reason?: string | null;
  index: number;
  delta: Delta;
  logprobs?: any;
}

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

export interface MessageType {
  role: string;
  content: string;
  model?: string;
  isImage?: boolean;
  reasoningContent?: string;
  timeToFirstToken?: number;
  usage?: {
    completionTokens?: number;
    promptTokens?: number;
    totalTokens?: number;
    reasoningTokens?: number;
  };
} 