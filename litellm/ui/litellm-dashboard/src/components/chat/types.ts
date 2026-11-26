import type { MCPEvent } from "../mcp_tools/types";
import type { TokenUsage } from "../chat_ui/ResponseMetrics";
export type { MCPEvent };

interface AttachmentData {
  id: string;
  name: string;
  url: string;
  mimeType?: string;
}

export interface ChatImageAttachment extends AttachmentData {
  kind?: "image";
}

export interface ChatFileAttachment extends AttachmentData {
  kind: "file";
  mimeType: string;
}

export type ChatAttachment = ChatImageAttachment | ChatFileAttachment;

export interface ChatImageGeneration {
  model: string;
  prompt: string;
}

export interface ChatBackgroundRun {
  id: string;
  status: "queued" | "in_progress" | "completed" | "failed" | "cancelled" | "incomplete" | "expired";
}

export interface ChatCitation {
  url: string;
  title: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "tool";
  content: string;
  attachments?: ChatAttachment[];
  generatedImages?: ChatImageAttachment[];
  imageGeneration?: ChatImageGeneration;
  research?: boolean;
  citations?: ChatCitation[];
  webSearchCount?: number;
  reasoningContent?: string;
  backgroundRun?: ChatBackgroundRun;
  backgroundError?: string;
  /** MCP tool events that occurred during this assistant turn, in order. */
  mcpEvents?: MCPEvent[];
  toolName?: string;
  toolArgs?: Record<string, unknown>;
  toolResult?: string;
  timeToFirstToken?: number;
  totalLatency?: number;
  usage?: TokenUsage;
  timestamp: number;
}

export type AssistantMessageUpdate = Partial<
  Pick<
    ChatMessage,
    | "content"
    | "reasoningContent"
    | "mcpEvents"
    | "timeToFirstToken"
    | "totalLatency"
    | "usage"
    | "generatedImages"
    | "imageGeneration"
    | "backgroundRun"
    | "backgroundError"
    | "citations"
    | "webSearchCount"
  >
>;

export interface Conversation {
  id: string;
  title: string;
  model: string;
  messages: ChatMessage[];
  mcpServerNames: string[];
  createdAt: number;
  updatedAt: number;
}
