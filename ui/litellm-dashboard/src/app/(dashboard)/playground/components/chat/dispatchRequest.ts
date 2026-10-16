import { EndpointType } from "@/components/chat_ui/mode_endpoint_mapping";
import type { MessageType } from "@/components/chat_ui/types";
import { makeOpenAIChatCompletionRequest } from "@/components/llm_calls/chat_completion";
import type { CustomHeaders } from "@/components/llm_calls/request_headers";
import type { ResponseHeadersListener } from "@/lib/http/observeResponseHeaders";
import { makeOpenAIResponsesRequest } from "@/components/llm_calls/responses_api";
import type { MCPServer, MCPToolset } from "@/components/mcp_tools/types";
import { callMCPTool } from "@/components/networking";
import type { UseChatHistoryReturn } from "../../hooks/useChatHistory";
import type { UseCodeInterpreterReturn } from "../../hooks/useCodeInterpreter";
import { makeA2ASendMessageRequest } from "../../llm_calls/a2a_send_message";
import { makeAnthropicMessagesRequest } from "../../llm_calls/anthropic_messages";
import { makeOpenAIAudioSpeechRequest } from "../../llm_calls/audio_speech";
import { makeOpenAIAudioTranscriptionRequest } from "../../llm_calls/audio_transcriptions";
import { makeOpenAIEmbeddingsRequest } from "../../llm_calls/embeddings_api";
import { makeOpenAIImageEditsRequest } from "../../llm_calls/image_edits";
import { makeOpenAIImageGenerationRequest } from "../../llm_calls/image_generation";
import { makeInteractionsRequest } from "../../llm_calls/interactions_api";
import type { PlaygroundSettings } from "./playgroundSettings";

type ApiMessage = { role: string; content: string | any[] };

export type DirectToolCall = Readonly<{ serverId: string; tool: string; arguments: Record<string, unknown> }>;

export type DispatchContext = Readonly<{
  settings: PlaygroundSettings;
  apiKey: string;
  text: string;
  userMessage: ApiMessage;
  history: readonly MessageType[];
  traceId: string;
  signal: AbortSignal;
  files: readonly File[];
  customHeaders: CustomHeaders;
  mcpServers: readonly MCPServer[];
  mcpToolsets: readonly MCPToolset[];
  directCall: DirectToolCall | null;
  chat: UseChatHistoryReturn;
  codeInterpreter: UseCodeInterpreterReturn;
  onHeaders: ResponseHeadersListener;
  onResponseId: (responseId: string) => void;
}>;

const optional = (values: readonly string[]): string[] | undefined => (values.length > 0 ? [...values] : undefined);

const NON_TEXT_FLAGS = ["isImage", "isAudio", "isEmbeddings", "isError", "mcpToolCall"] as const;

const isText = (message: MessageType): boolean => NON_TEXT_FLAGS.every((flag) => !message[flag]);

const isFilled = (content: MessageType["content"]): boolean => typeof content !== "string" || content.trim() !== "";

const isTurn = (message: MessageType | undefined, role: "user" | "assistant"): boolean =>
  message?.role === role && isText(message) && isFilled(message.content);

const isReply = (history: readonly MessageType[], index: number): boolean =>
  isTurn(history[index], "assistant") && !history[index + 1]?.isError;

const inExchange = (history: readonly MessageType[], index: number): boolean =>
  isTurn(history[index], "user")
    ? isReply(history, index + 1)
    : isReply(history, index) && isTurn(history[index - 1], "user");

export const textHistory = (history: readonly MessageType[]): ApiMessage[] =>
  history.filter((_, index) => inExchange(history, index)).map(({ role, content }) => ({ role, content }));

const toolResultText = (result: { content?: { type: string; text?: string }[] } | null): string => {
  if (result?.content && result.content.length > 0) {
    return JSON.stringify(
      result.content.map((item) => (item.type === "text" ? item.text : item)).filter(Boolean),
      null,
      2,
    );
  }
  return JSON.stringify(result, null, 2) || "Tool executed successfully.";
};

export async function dispatchRequest(context: DispatchContext): Promise<void> {
  const { settings, apiKey, text, userMessage, history, traceId, signal, chat, onHeaders } = context;
  const model = settings.model ?? "";
  const tags = [...settings.tags];
  const baseUrl = settings.proxyBaseUrl || undefined;
  const headers = context.customHeaders;
  const vectorStores = optional(settings.vectorStores);
  const guardrails = optional(settings.guardrails);
  const policies = optional(settings.policies);
  const mcpSelection = [...settings.mcpSelection];
  const mcpServers = [...context.mcpServers];
  const toolLimits = Object.fromEntries(Object.entries(settings.toolLimits).map(([key, tools]) => [key, [...tools]]));
  const mcpToolsets = [...context.mcpToolsets];

  switch (settings.endpoint) {
    case EndpointType.CHAT:
      return makeOpenAIChatCompletionRequest(
        [
          ...textHistory(history).map(({ role, content }) => ({
            role,
            content: typeof content === "string" ? content : "",
          })),
          userMessage,
        ],
        (chunk, servedModel) => chat.updateTextUI("assistant", chunk, servedModel),
        model,
        apiKey,
        tags,
        signal,
        chat.updateReasoningContent,
        chat.updateTimingData,
        chat.updateUsageData,
        traceId,
        vectorStores,
        guardrails,
        policies,
        mcpSelection,
        chat.updateChatImageUI,
        chat.updateSearchResults,
        settings.temperature ?? undefined,
        settings.maxTokens ?? undefined,
        chat.updateTotalLatency,
        baseUrl,
        mcpServers,
        toolLimits,
        chat.handleMCPEvent,
        settings.simulateFallback,
        mcpToolsets,
        settings.stream,
        headers,
        onHeaders,
      );
    case EndpointType.RESPONSES: {
      const continuing = chat.useApiSessionManagement && chat.responsesSessionId;
      return makeOpenAIResponsesRequest(
        (continuing ? [userMessage] : [...textHistory(history), userMessage]) as MessageType[],
        (role, delta, servedModel) => chat.updateTextUI(role, delta, servedModel),
        model,
        apiKey,
        tags,
        signal,
        chat.updateReasoningContent,
        chat.updateTimingData,
        chat.updateUsageData,
        traceId,
        vectorStores,
        guardrails,
        policies,
        mcpSelection,
        chat.useApiSessionManagement ? chat.responsesSessionId : null,
        context.onResponseId,
        chat.handleMCPEvent,
        context.codeInterpreter.enabled,
        context.codeInterpreter.setResult,
        baseUrl,
        mcpServers,
        toolLimits,
        mcpToolsets,
        settings.stream,
        chat.updateTotalLatency,
        headers,
        onHeaders,
      );
    }
    case EndpointType.ANTHROPIC_MESSAGES:
      return makeAnthropicMessagesRequest(
        [...textHistory(history), userMessage] as MessageType[],
        (role, delta, servedModel) => chat.updateTextUI(role, delta, servedModel),
        model,
        apiKey,
        tags,
        signal,
        chat.updateReasoningContent,
        chat.updateTimingData,
        chat.updateUsageData,
        traceId,
        vectorStores,
        guardrails,
        policies,
        mcpSelection,
        baseUrl,
        mcpServers,
        toolLimits,
        mcpToolsets,
        settings.stream,
        headers,
        onHeaders,
      );
    case EndpointType.IMAGE:
      return makeOpenAIImageGenerationRequest(
        text,
        (imageUrl, servedModel) => chat.updateImageUI(imageUrl, servedModel),
        model,
        apiKey,
        tags,
        signal,
        baseUrl,
        headers,
        onHeaders,
      );
    case EndpointType.IMAGE_EDITS:
      return makeOpenAIImageEditsRequest(
        context.files.length === 1 ? context.files[0] : [...context.files],
        text,
        (imageUrl, servedModel) => chat.updateImageUI(imageUrl, servedModel),
        model,
        apiKey,
        tags,
        signal,
        baseUrl,
        headers,
        onHeaders,
      );
    case EndpointType.SPEECH:
      return makeOpenAIAudioSpeechRequest(
        text,
        settings.voice,
        (audioUrl, servedModel) => chat.updateAudioUI(audioUrl, servedModel),
        model,
        apiKey,
        tags,
        signal,
        undefined,
        undefined,
        baseUrl,
        headers,
        onHeaders,
      );
    case EndpointType.TRANSCRIPTION:
      return makeOpenAIAudioTranscriptionRequest(
        context.files[0],
        (transcription, servedModel) => chat.updateTextUI("assistant", transcription, servedModel),
        model,
        apiKey,
        tags,
        signal,
        undefined,
        undefined,
        undefined,
        undefined,
        baseUrl,
        headers,
        onHeaders,
      );
    case EndpointType.EMBEDDINGS:
      return makeOpenAIEmbeddingsRequest(
        text,
        (embeddings, servedModel) => chat.updateEmbeddingsUI(embeddings, servedModel),
        model,
        apiKey,
        tags,
        baseUrl,
        headers,
        onHeaders,
      );
    case EndpointType.INTERACTIONS:
      return makeInteractionsRequest(
        text,
        (reply, servedModel) => chat.updateTextUI("assistant", reply, servedModel),
        model,
        apiKey,
        tags,
        signal,
        baseUrl,
        undefined,
        headers,
        onHeaders,
      );
    case EndpointType.A2A_AGENTS:
      return makeA2ASendMessageRequest(
        settings.agent ?? "",
        text,
        (chunk, servedModel) => chat.updateTextUI("assistant", chunk, servedModel),
        apiKey,
        signal,
        chat.updateTimingData,
        chat.updateTotalLatency,
        chat.updateA2AMetadata,
        baseUrl,
        guardrails,
        headers,
      );
    case EndpointType.MCP: {
      if (!context.directCall) return;
      const result = await callMCPTool(
        apiKey,
        context.directCall.serverId,
        context.directCall.tool,
        context.directCall.arguments,
        { ...(guardrails ? { guardrails } : {}), customHeaders: { ...headers } },
      );
      chat.updateTextUI("assistant", toolResultText(result));
      return;
    }
    case EndpointType.REALTIME:
    case EndpointType.VIDEO:
      return;
  }
}
