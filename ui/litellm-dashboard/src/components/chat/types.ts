import type { MCPEvent } from "../playground/chat_ui/MCPEventsDisplay";

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
  timestamp: number;
}

export interface Conversation {
  id: string;
  title: string;
  model: string;
  messages: ChatMessage[];
  mcpServerNames: string[];
  createdAt: number;
  updatedAt: number;
}
