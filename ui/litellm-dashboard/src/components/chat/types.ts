/** Represents a single MCP tool event emitted during an assistant turn. */
export interface MCPEvent {
  type: string;
  sequence_number?: number;
  output_index?: number;
  item_id?: string;
  item?: {
    id?: string;
    type?: string;
    server_label?: string;
    tools?: Array<{
      name: string;
      description: string;
      annotations?: {
        read_only?: boolean;
      };
      input_schema?: unknown;
    }>;
    name?: string;
    arguments?: string;
    output?: string;
  };
  delta?: string;
  arguments?: string;
  timestamp?: number;
}

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
