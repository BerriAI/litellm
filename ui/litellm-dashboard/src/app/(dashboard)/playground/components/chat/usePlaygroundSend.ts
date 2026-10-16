import { useRef, useState, type RefObject } from "react";
import { v4 as uuidv4 } from "uuid";
import { EndpointType } from "@/components/chat_ui/mode_endpoint_mapping";
import type { MessageType, ReplyTrace } from "@/components/chat_ui/types";
import { gatewayTraceFromHeaders } from "@/components/llm_calls/reply_trace";
import type { CustomHeaders } from "@/components/llm_calls/request_headers";
import type { MCPToolArgumentsFormRef } from "@/components/mcp_tools/MCPToolArgumentsForm";
import type { MCPServer, MCPToolset } from "@/components/mcp_tools/types";
import { toast } from "@/lib/toast";
import type { UseChatHistoryReturn } from "../../hooks/useChatHistory";
import type { UseCodeInterpreterReturn } from "../../hooks/useCodeInterpreter";
import { createChatMultimodalMessage } from "../chat_ui/ChatImageUtils";
import { createMultimodalMessage } from "../chat_ui/ResponsesImageUtils";
import { addFiles, toAttachment, type DraftFile } from "./composerFiles";
import { dispatchRequest, type DirectToolCall, type DispatchContext } from "./dispatchRequest";
import type { AttachKind, Capabilities } from "./endpointCapabilities";
import type { PlaygroundSettings } from "./playgroundSettings";

const MODEL_ENDPOINTS: readonly EndpointType[] = [
  EndpointType.CHAT,
  EndpointType.IMAGE,
  EndpointType.SPEECH,
  EndpointType.IMAGE_EDITS,
  EndpointType.RESPONSES,
  EndpointType.ANTHROPIC_MESSAGES,
  EndpointType.EMBEDDINGS,
  EndpointType.TRANSCRIPTION,
  EndpointType.INTERACTIONS,
];

export type SendDeps = Readonly<{
  settings: PlaygroundSettings;
  capabilities: Capabilities;
  apiKey: string | null;
  signedIn: boolean;
  customHeaders: CustomHeaders;
  mcpServers: readonly MCPServer[];
  mcpToolsets: readonly MCPToolset[];
  directCall: () => Omit<DirectToolCall, "arguments"> | null;
  toolArgsForm: RefObject<MCPToolArgumentsFormRef | null>;
  chat: UseChatHistoryReturn;
  codeInterpreter: UseCodeInterpreterReturn;
}>;

const errorText = (error: unknown): string => (error instanceof Error ? error.message : String(error));

const previewFor = (file: File): string | undefined => {
  const url = URL.createObjectURL(file);
  return url.startsWith("blob:") ? url : undefined;
};

const missingInput = (deps: SendDeps, text: string, files: readonly DraftFile[]): string | null => {
  const { settings, capabilities } = deps;
  if (settings.endpoint === EndpointType.IMAGE_EDITS && files.length === 0)
    return "Please upload at least one image for editing";
  if (settings.endpoint === EndpointType.TRANSCRIPTION && files.length === 0)
    return "Please upload an audio file for transcription";
  if (capabilities.textRequired && text.trim() === "") return "";
  if (settings.endpoint === EndpointType.A2A_AGENTS && !settings.agent)
    return "Please select an agent to send a message";
  if (MODEL_ENDPOINTS.includes(settings.endpoint) && !settings.model)
    return "Please select a model before sending a request";
  return null;
};

const apiMessage = async (endpoint: EndpointType, text: string, file: File | undefined) => {
  if (file && endpoint === EndpointType.RESPONSES) return createMultimodalMessage(text, file);
  if (file && endpoint === EndpointType.CHAT) return createChatMultimodalMessage(text, file);
  return { role: "user", content: text };
};

const targetName = (
  settings: PlaygroundSettings,
  direct: Omit<DirectToolCall, "arguments"> | null,
): string | undefined => {
  if (settings.endpoint === EndpointType.A2A_AGENTS) return settings.agent ?? undefined;
  if (settings.endpoint === EndpointType.MCP) return direct?.tool;
  return settings.model ?? undefined;
};

const withTrace = (history: MessageType[], from: number, trace: ReplyTrace): MessageType[] => {
  const replyIndex = history.findLastIndex((message, index) => index > from && message.role === "assistant");
  if (replyIndex < 0) return history;
  return history.map((message, index) => (index === replyIndex ? { ...message, trace } : message));
};

const untilAborted = <T extends object>(target: T, signal: AbortSignal): T =>
  new Proxy(target, {
    get: (object, key, receiver) => {
      const value: unknown = Reflect.get(object, key, receiver);
      return typeof value === "function"
        ? (...args: unknown[]) => (signal.aborted ? undefined : value(...args))
        : value;
    },
  });

type Draft = Readonly<{ kind: AttachKind | null; files: readonly DraftFile[] }>;

type PreparedRequest = Readonly<{
  direct: Omit<DirectToolCall, "arguments"> | null;
  toolArguments: Record<string, unknown>;
  userMessage: Awaited<ReturnType<typeof apiMessage>>;
}>;

export function usePlaygroundSend(deps: SendDeps) {
  const [text, setText] = useState("");
  const [draft, setDraft] = useState<Draft>({ kind: null, files: [] });
  const [isLoading, setIsLoading] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const preparing = useRef(false);
  const files = draft.kind === deps.capabilities.attach ? draft.files : [];
  const setFiles = (next: readonly DraftFile[]) => setDraft({ kind: deps.capabilities.attach, files: next });

  const attach = (incoming: readonly File[]) => {
    if (!deps.capabilities.attach) return;
    const result = addFiles(deps.capabilities.attach, files, incoming, previewFor);
    result.errors.forEach((error) => toast.error(error));
    setFiles(result.files);
  };

  const removeFile = (index: number) => {
    const removed = files[index];
    if (removed?.previewUrl) URL.revokeObjectURL(removed.previewUrl);
    setFiles(files.filter((_, i) => i !== index));
  };

  const stop = () => {
    abortRef.current?.abort();
    abortRef.current = null;
    setIsLoading(false);
  };

  const clearDraft = () => {
    stop();
    files.forEach((file) => file.previewUrl && URL.revokeObjectURL(file.previewUrl));
    setFiles([]);
    setText("");
  };

  const cancel = () => {
    if (!abortRef.current) return;
    stop();
    toast.info("Request cancelled");
  };

  const prepare = async (): Promise<PreparedRequest | null> => {
    const { settings } = deps;
    const missing = missingInput(deps, text, files);
    if (missing !== null) {
      if (missing) toast.fromError(missing);
      return null;
    }
    const direct = settings.endpoint === EndpointType.MCP ? deps.directCall() : null;
    if (settings.endpoint === EndpointType.MCP && !direct) {
      toast.fromError("Pick an MCP server and a tool to call");
      return null;
    }
    const toolArguments = await (direct
      ? deps.toolArgsForm.current?.getSubmitValues().catch((error: unknown) => {
          toast.fromError(error instanceof Error ? error.message : "Please fill in all required parameters");
          return null;
        })
      : Promise.resolve({}));
    if (toolArguments === null) return null;
    if (!deps.signedIn) return null;
    if (!deps.apiKey) {
      toast.fromError("Please provide a Virtual Key or select Current UI Session");
      return null;
    }

    const userMessage = await apiMessage(settings.endpoint, text, files[0]?.file).catch(() => null);
    if (!userMessage) {
      toast.fromError("Failed to process the attachment. Please try again.");
      return null;
    }
    return { direct, toolArguments: toolArguments ?? {}, userMessage };
  };

  const send = async () => {
    if (preparing.current || abortRef.current) return;
    preparing.current = true;
    const prepared = await prepare().finally(() => {
      preparing.current = false;
    });
    if (!prepared || !deps.apiKey) return;
    const { settings, chat } = deps;
    const { direct, toolArguments, userMessage } = prepared;

    const traceId = chat.messageTraceId ?? uuidv4();
    if (!chat.messageTraceId) chat.setMessageTraceId(traceId);
    const history = chat.chatHistory;
    const requestedModel = targetName(settings, direct);
    chat.setChatHistory([
      ...history,
      {
        role: "user",
        content: text,
        model: requestedModel,
        ...(files.length > 0 ? { attachments: files.map(toAttachment) } : {}),
        ...(direct ? { mcpToolCall: { tool: direct.tool, arguments: toolArguments } } : {}),
      },
    ]);
    chat.clearMCPEvents();
    deps.codeInterpreter.clearResult();
    const sentFiles = files.map((draft) => draft.file);
    setText("");
    setFiles([]);
    setIsLoading(true);

    const controller = new AbortController();
    abortRef.current = controller;
    const live = untilAborted(chat, controller.signal);
    const captured: { headers?: Headers; responseId?: string } = {};
    const trace = (): ReplyTrace => ({
      ...(captured.headers ? gatewayTraceFromHeaders(captured.headers) : {}),
      endpoint: settings.endpoint,
      requestedModel,
      responseId: captured.responseId,
    });

    try {
      const context: DispatchContext = {
        settings,
        apiKey: deps.apiKey,
        text,
        userMessage,
        history,
        traceId,
        signal: controller.signal,
        files: sentFiles,
        customHeaders: deps.customHeaders,
        mcpServers: deps.mcpServers,
        mcpToolsets: deps.mcpToolsets,
        directCall: direct ? { ...direct, arguments: toolArguments } : null,
        chat: live,
        codeInterpreter: untilAborted(deps.codeInterpreter, controller.signal),
        onHeaders: (headers) => {
          captured.headers = headers;
        },
        onResponseId: (responseId) => {
          captured.responseId = responseId;
          live.handleResponseId(responseId);
        },
      };
      await dispatchRequest(context);
      if (controller.signal.aborted) return;
      chat.setChatHistory((current) => withTrace(current, history.length, trace()));
    } catch (error) {
      if (controller.signal.aborted) return;
      chat.setChatHistory((current) => [
        ...current,
        { role: "assistant", content: errorText(error), isError: true, model: requestedModel, trace: trace() },
      ]);
    } finally {
      if (abortRef.current === controller) {
        abortRef.current = null;
        setIsLoading(false);
      }
    }
  };

  return { text, setText, files, attach, removeFile, clearDraft, isLoading, send, cancel } as const;
}
