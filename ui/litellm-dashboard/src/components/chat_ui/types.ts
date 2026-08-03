export interface VectorStoreSearchResult {
  score: number;
  content: Array<{ text: string; type: string }>;
  file_id?: string;
  filename?: string;
  attributes?: Record<string, any>;
}

export interface VectorStoreSearchResponse {
  object: string;
  search_query: string;
  data: VectorStoreSearchResult[];
}

export interface A2ATaskMetadata {
  taskId?: string;
  contextId?: string;
  status?: {
    state?: string;
    timestamp?: string;
    message?: string;
  };
  metadata?: Record<string, any>;
}

export interface MessageType {
  role: string;
  content: string | MultimodalContent[];
  model?: string;
  isImage?: boolean;
  isAudio?: boolean;
  isEmbeddings?: boolean;
  reasoningContent?: string;
  timeToFirstToken?: number;
  totalLatency?: number;
  usage?: {
    completionTokens?: number;
    promptTokens?: number;
    totalTokens?: number;
    reasoningTokens?: number;
    cost?: number;
  };
  toolName?: string;
  imagePreviewUrl?: string;
  image?: {
    url: string;
    detail: string;
  };
  searchResults?: VectorStoreSearchResponse[];
  a2aMetadata?: A2ATaskMetadata;
}

export interface MultimodalContent {
  type: "input_text" | "input_image";
  text?: string;
  image_url?: string;
}
