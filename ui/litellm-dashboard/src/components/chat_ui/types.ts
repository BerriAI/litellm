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
}

export interface MessageType {
  role: string;
  content: string;
  model?: string;
  isImage?: boolean;
  reasoningContent?: string;
} 