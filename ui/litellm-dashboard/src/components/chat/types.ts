import type { MCPEvent } from "../mcp_tools/types";
import type { TokenUsage } from "../chat_ui/ResponseMetrics";
export type { MCPEvent };

export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "tool";
  content: string;
  reasoningContent?: string;
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
  Pick<ChatMessage, "content" | "reasoningContent" | "mcpEvents" | "timeToFirstToken" | "totalLatency" | "usage">
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
