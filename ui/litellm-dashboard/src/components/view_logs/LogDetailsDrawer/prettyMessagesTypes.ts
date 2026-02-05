/**
 * Type definitions for pretty messages view
 */

export interface ParsedMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string;
  toolCalls?: ToolCall[];
  toolCallId?: string;
}

export interface ToolCall {
  id: string;
  name: string;
  arguments: Record<string, any>;
}

export interface ParsedMessages {
  requestMessages: ParsedMessage[];
  responseMessage: ParsedMessage | null;
}

export interface RoleStyle {
  background: string;
  borderColor: string;
  label: string;
  labelColor: string;
}
