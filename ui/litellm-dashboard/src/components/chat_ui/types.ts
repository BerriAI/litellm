import type { TokenUsage } from "./ResponseMetrics";
import type { GatewayTrace } from "@/components/llm_calls/reply_trace";

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

export type ReplyTrace = Partial<GatewayTrace> &
  Readonly<{
    endpoint: string;
    requestedModel?: string;
    responseId?: string;
  }>;

export type MessageAttachment = Readonly<{
  name: string;
  size: number;
  kind: "image" | "pdf" | "audio";
  previewUrl?: string;
}>;

export type McpToolCallDisplay = Readonly<{
  tool: string;
  arguments: Readonly<Record<string, unknown>>;
}>;

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
  usage?: TokenUsage;
  toolName?: string;
  imagePreviewUrl?: string;
  image?: {
    url: string;
    detail: string;
  };
  searchResults?: VectorStoreSearchResponse[];
  a2aMetadata?: A2ATaskMetadata;
  trace?: ReplyTrace;
  attachments?: readonly MessageAttachment[];
  mcpToolCall?: McpToolCallDisplay;
  isError?: boolean;
}

export interface MultimodalContent {
  type: "input_text" | "input_image" | "input_file";
  filename?: string;
  file_data?: string;
  text?: string;
  image_url?: string;
  detail?: "low" | "high" | "auto";
}
