/**
 * Type definitions for pretty messages view
 */

export type MessageRole = "system" | "user" | "assistant" | "tool";

export interface ParsedMessage {
  role: MessageRole;
  content: string;
  toolCalls?: ToolCall[];
  toolCallId?: string;
}

export type RequestPayload =
  | { kind: "chat"; messages: readonly unknown[] }
  | { kind: "responses"; instructions: string; input: string | readonly unknown[] }
  | { kind: "unknown" };

export type ResponsePayload =
  | { kind: "chat"; choices: readonly unknown[] }
  | { kind: "responses"; output: readonly unknown[] }
  | { kind: "unknown" };

export interface ToolCall {
  id: string;
  name: string;
  arguments: Record<string, unknown>;
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
