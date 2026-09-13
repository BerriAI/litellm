/**
 * Utility functions for parsing and formatting messages for pretty view
 */

import {
  MessageRole,
  ParsedMessage,
  ParsedMessages,
  RequestPayload,
  ResponsePayload,
  RoleStyle,
  ToolCall,
} from "./prettyMessagesTypes";

/**
 * Role color styles for message cards - minimal, professional design
 * Color only used for labels and left border accent
 */
export const ROLE_STYLES: Record<string, RoleStyle> = {
  system: {
    background: "transparent",
    borderColor: "var(--color-muted-foreground)",
    label: "SYSTEM",
    labelColor: "var(--color-muted-foreground)",
  },
  user: {
    background: "transparent",
    borderColor: "var(--color-info)",
    label: "USER",
    labelColor: "var(--color-info)",
  },
  assistant: {
    background: "transparent",
    borderColor: "var(--color-success)",
    label: "ASSISTANT",
    labelColor: "var(--color-success)",
  },
  tool: {
    background: "transparent",
    borderColor: "var(--color-warning)",
    label: "TOOL RESULT",
    labelColor: "var(--color-warning)",
  },
};

type UnknownRecord = Record<string, unknown>;

const isRecord = (value: unknown): value is UnknownRecord =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const asString = (value: unknown): string => (typeof value === "string" ? value : "");

const ROLES: readonly MessageRole[] = ["system", "user", "assistant", "tool"];

const toRole = (value: unknown, fallback: MessageRole): MessageRole => {
  if (value === "developer") return "system";
  if (value === "function") return "tool";
  return ROLES.includes(value as MessageRole) ? (value as MessageRole) : fallback;
};

const classifyRequest = (request: unknown): RequestPayload => {
  if (Array.isArray(request)) return { kind: "chat", messages: request };
  if (!isRecord(request)) return { kind: "unknown" };
  if (Array.isArray(request.messages)) return { kind: "chat", messages: request.messages };
  const { input } = request;
  if (typeof input === "string" || Array.isArray(input)) {
    return { kind: "responses", instructions: asString(request.instructions), input };
  }
  return { kind: "unknown" };
};

const classifyResponse = (response: unknown): ResponsePayload => {
  if (!isRecord(response)) return { kind: "unknown" };
  if (Array.isArray(response.choices)) return { kind: "chat", choices: response.choices };
  if (Array.isArray(response.output)) return { kind: "responses", output: response.output };
  return { kind: "unknown" };
};

/**
 * Parse request messages and response message from log data
 */
export const parseMessages = (request: unknown, response: unknown): ParsedMessages => ({
  requestMessages: parseRequestMessages(classifyRequest(request)),
  responseMessage: parseResponseMessage(classifyResponse(response)),
});

const parseRequestMessages = (payload: RequestPayload): ParsedMessage[] => {
  switch (payload.kind) {
    case "chat":
      return payload.messages.map(parseChatMessage);
    case "responses": {
      const instructions: ParsedMessage[] = payload.instructions
        ? [{ role: "system", content: payload.instructions }]
        : [];
      const input: ParsedMessage[] =
        typeof payload.input === "string"
          ? [{ role: "user", content: payload.input }]
          : payload.input.flatMap(parseResponsesInputItem);
      return [...instructions, ...input];
    }
    case "unknown":
      return [];
  }
};

const parseResponseMessage = (payload: ResponsePayload): ParsedMessage | null => {
  switch (payload.kind) {
    case "chat": {
      const choice = payload.choices[0];
      const message = isRecord(choice) ? choice.message : undefined;
      if (!isRecord(message)) return null;
      return {
        role: toRole(message.role, "assistant"),
        content: parseMessageContent(message.content),
        toolCalls: parseChatToolCalls(message.tool_calls),
      };
    }
    case "responses": {
      const content = payload.output
        .filter((item): item is UnknownRecord => isRecord(item) && item.type === "message")
        .map((item) => parseMessageContent(item.content))
        .filter((text) => text.length > 0)
        .join("\n");
      const toolCalls = payload.output.filter(isResponsesFunctionCall).map(parseResponsesFunctionCall);
      if (content.length === 0 && toolCalls.length === 0) return null;
      return { role: "assistant", content, toolCalls: toolCalls.length > 0 ? toolCalls : undefined };
    }
    case "unknown":
      return null;
  }
};

const parseChatMessage = (message: unknown): ParsedMessage => {
  if (!isRecord(message)) return { role: "user", content: parseMessageContent(message) };
  return {
    role: toRole(message.role, "user"),
    content: parseMessageContent(message.content),
    toolCalls: parseChatToolCalls(message.tool_calls),
    toolCallId: typeof message.tool_call_id === "string" ? message.tool_call_id : undefined,
  };
};

const parseResponsesInputItem = (item: unknown): ParsedMessage[] => {
  if (typeof item === "string") return [{ role: "user", content: item }];
  if (!isRecord(item)) return [];
  if (item.type === "function_call") {
    return [{ role: "assistant", content: "", toolCalls: [parseResponsesFunctionCall(item)] }];
  }
  if (item.type === "function_call_output") {
    return [{ role: "tool", content: parseMessageContent(item.output), toolCallId: asString(item.call_id) }];
  }
  if (item.type === "reasoning") return [];
  if ("role" in item || "content" in item) {
    return [{ role: toRole(item.role, "user"), content: parseMessageContent(item.content) }];
  }
  return [];
};

const isResponsesFunctionCall = (item: unknown): item is UnknownRecord =>
  isRecord(item) && item.type === "function_call";

const parseResponsesFunctionCall = (item: UnknownRecord): ToolCall => ({
  id: asString(item.call_id) || asString(item.id),
  name: asString(item.name) || "unknown",
  arguments: parseToolArguments(item.arguments),
});

/**
 * Parse message content - handle strings and content arrays (for vision, etc.)
 */
const parseMessageContent = (content: unknown): string => {
  if (typeof content === "string") return content;
  if (content === null || content === undefined) return "";
  if (Array.isArray(content)) return content.map(parseContentPart).join("\n");
  return JSON.stringify(content);
};

const parseContentPart = (part: unknown): string => {
  if (typeof part === "string") return part;
  if (!isRecord(part)) return JSON.stringify(part);
  switch (part.type) {
    case "text":
    case "input_text":
    case "output_text":
      return asString(part.text);
    case "refusal":
      return asString(part.refusal);
    case "image_url":
    case "input_image":
      return "[Image]";
    case "input_file":
      return "[File]";
    case "input_audio":
      return "[Audio]";
    default:
      return JSON.stringify(part);
  }
};

/**
 * Parse tool calls from response message
 */
const parseChatToolCalls = (toolCalls: unknown): ToolCall[] | undefined => {
  if (!Array.isArray(toolCalls)) return undefined;
  return toolCalls.map((toolCall) => {
    const call = isRecord(toolCall) ? toolCall : {};
    const fn = isRecord(call.function) ? call.function : {};
    return {
      id: asString(call.id),
      name: asString(fn.name) || "unknown",
      arguments: parseToolArguments(fn.arguments),
    };
  });
};

/**
 * Parse tool arguments - handle both string and object formats
 */
const parseToolArguments = (args: unknown): Record<string, unknown> => {
  if (!args) return {};
  if (typeof args === "string") {
    try {
      const parsed: unknown = JSON.parse(args);
      return isRecord(parsed) ? parsed : { raw: args };
    } catch {
      return { raw: args };
    }
  }
  return isRecord(args) ? args : {};
};
