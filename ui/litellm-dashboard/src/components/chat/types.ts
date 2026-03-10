// Minimal shape for an MCP event stored per assistant message.
// Matches the MCPEvent type in MCPEventsDisplay (kept loose to avoid a
// cross-component import cycle).
export interface StoredMCPEvent {
  type: string;
  sequence_number?: number;
  output_index?: number;
  item_id?: string;
  item?: unknown;
  delta?: unknown;
  arguments?: unknown;
  timestamp: number;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "tool";
  content: string;
  reasoningContent?: string;
  /** MCP tool events that occurred during this assistant turn, in order. */
  mcpEvents?: StoredMCPEvent[];
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
