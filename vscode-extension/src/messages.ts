import type * as vscode from "vscode";
import type {
  ChatCompletionAssistantMessageParam,
  ChatCompletionContentPart,
  ChatCompletionCreateParamsStreaming,
  ChatCompletionMessageParam,
  ChatCompletionMessageToolCall,
  ChatCompletionTool,
  ChatCompletionToolMessageParam,
} from "openai/resources/chat/completions";
import { estimateTokens } from "./models";

export interface ChatRequestInput {
  readonly model: string;
  readonly messages: readonly vscode.LanguageModelChatRequestMessage[];
  readonly tools: readonly vscode.LanguageModelChatTool[];
  readonly requireToolCall: boolean;
  readonly reasoningEffort: string | undefined;
  readonly modelOptions: { readonly [key: string]: unknown };
}

interface TextPart {
  readonly value: string;
}

interface ToolCallPart {
  readonly callId: string;
  readonly name: string;
  readonly input: object;
}

interface ToolResultPart {
  readonly callId: string;
  readonly content: ReadonlyArray<unknown>;
}

interface DataPart {
  readonly mimeType: string;
  readonly data: Uint8Array;
}

const USER_ROLE = 1;
const ASSISTANT_ROLE = 2;
const SYSTEM_ROLE = 3;

export const ESTIMATED_TOKENS_PER_IMAGE = 1000;

const isRecord = (value: unknown): value is Record<string, unknown> => typeof value === "object" && value !== null;

const isTextPart = (part: unknown): part is TextPart => isRecord(part) && typeof part.value === "string";

const isToolCallPart = (part: unknown): part is ToolCallPart =>
  isRecord(part) && typeof part.callId === "string" && typeof part.name === "string" && isRecord(part.input);

const isToolResultPart = (part: unknown): part is ToolResultPart =>
  isRecord(part) && typeof part.callId === "string" && Array.isArray(part.content);

const isDataPart = (part: unknown): part is DataPart =>
  isRecord(part) && typeof part.mimeType === "string" && part.data instanceof Uint8Array;

const isImagePart = (part: unknown): part is DataPart => isDataPart(part) && part.mimeType.startsWith("image/");

const dataUrl = (part: DataPart): string => `data:${part.mimeType};base64,${Buffer.from(part.data).toString("base64")}`;

const textOf = (part: unknown): string => {
  if (isTextPart(part)) {
    return part.value;
  }
  if (isDataPart(part) && part.mimeType.startsWith("text/")) {
    return Buffer.from(part.data).toString("utf8");
  }
  if (isRecord(part) && "value" in part) {
    return JSON.stringify(part.value);
  }
  return "";
};

const contentParts = (parts: readonly unknown[]): readonly ChatCompletionContentPart[] =>
  parts.flatMap((part): readonly ChatCompletionContentPart[] => {
    if (isImagePart(part)) {
      return [{ type: "image_url", image_url: { url: dataUrl(part) } }];
    }
    const text = textOf(part);
    return text === "" ? [] : [{ type: "text", text }];
  });

const toolMessage = (part: ToolResultPart): ChatCompletionToolMessageParam => ({
  role: "tool",
  tool_call_id: part.callId,
  content: part.content.filter((item) => !isImagePart(item)).map(textOf).join(""),
});

const userMessages = (parts: readonly unknown[]): readonly ChatCompletionMessageParam[] => {
  const toolResults = parts.filter(isToolResultPart);
  const toolResultImages = toolResults.flatMap((result) => result.content.filter(isImagePart));
  const remaining = parts.filter((part) => !isToolResultPart(part));
  const userContent = contentParts([...remaining, ...toolResultImages]);
  const userMessage: readonly ChatCompletionMessageParam[] =
    userContent.length === 0 ? [] : [{ role: "user", content: [...userContent] }];
  return [...toolResults.map(toolMessage), ...userMessage];
};

const toolCall = (part: ToolCallPart): ChatCompletionMessageToolCall => ({
  id: part.callId,
  type: "function",
  function: { name: part.name, arguments: JSON.stringify(part.input) },
});

const assistantMessages = (parts: readonly unknown[]): readonly ChatCompletionAssistantMessageParam[] => {
  const text = parts.filter(isTextPart).map((part) => part.value).join("");
  const toolCalls = parts.filter(isToolCallPart).map(toolCall);
  if (text === "" && toolCalls.length === 0) {
    return [];
  }
  return [
    {
      role: "assistant",
      content: text === "" ? null : text,
      ...(toolCalls.length === 0 ? {} : { tool_calls: toolCalls }),
    },
  ];
};

const convertMessage = (message: vscode.LanguageModelChatRequestMessage): readonly ChatCompletionMessageParam[] => {
  const role: number = message.role;
  switch (role) {
    case USER_ROLE:
      return userMessages(message.content);
    case ASSISTANT_ROLE:
      return assistantMessages(message.content);
    case SYSTEM_ROLE:
      return [{ role: "system", content: message.content.map(textOf).join("") }];
    default:
      return [];
  }
};

export const toChatCompletionMessages = (
  messages: readonly vscode.LanguageModelChatRequestMessage[],
): readonly ChatCompletionMessageParam[] => messages.flatMap(convertMessage);

const imagePartsIn = (parts: readonly unknown[]): readonly DataPart[] => [
  ...parts.filter(isImagePart),
  ...parts.filter(isToolResultPart).flatMap((result) => result.content.filter(isImagePart)),
];

const withoutImageData = (key: string, value: unknown): unknown => (key === "image_url" ? undefined : value);

export const estimateMessageTokens = (message: vscode.LanguageModelChatRequestMessage): number => {
  const converted = convertMessage(message);
  if (converted.length === 0) {
    return 0;
  }
  const images = imagePartsIn(message.content).length;
  return estimateTokens(JSON.stringify(converted, withoutImageData)) + images * ESTIMATED_TOKENS_PER_IMAGE;
};

const toTool = (tool: vscode.LanguageModelChatTool): ChatCompletionTool => ({
  type: "function",
  function: {
    name: tool.name,
    description: tool.description,
    ...(tool.inputSchema === undefined ? {} : { parameters: tool.inputSchema as Record<string, unknown> }),
  },
});

const NUMERIC_OPTIONS = ["temperature", "top_p", "max_tokens", "presence_penalty", "frequency_penalty", "seed"] as const;

const forwardedModelOptions = (modelOptions: { readonly [key: string]: unknown }): Record<string, number> =>
  Object.fromEntries(
    NUMERIC_OPTIONS.flatMap((key) => {
      const value = modelOptions[key];
      return typeof value === "number" ? [[key, value] as const] : [];
    }),
  );

export const buildChatCompletionParams = (input: ChatRequestInput): ChatCompletionCreateParamsStreaming => ({
  model: input.model,
  messages: [...toChatCompletionMessages(input.messages)],
  stream: true,
  stream_options: { include_usage: true },
  ...forwardedModelOptions(input.modelOptions),
  ...(input.tools.length === 0 ? {} : { tools: input.tools.map(toTool) }),
  ...(input.tools.length === 0 || !input.requireToolCall ? {} : { tool_choice: "required" }),
  ...(input.reasoningEffort === undefined
    ? {}
    : { reasoning_effort: input.reasoningEffort as ChatCompletionCreateParamsStreaming["reasoning_effort"] }),
});
