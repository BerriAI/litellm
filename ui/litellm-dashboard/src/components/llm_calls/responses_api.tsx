import { observeResponseHeaders, type ResponseHeadersListener } from "@/lib/http/observeResponseHeaders";
import openai from "openai";
import type { ResponseFunctionToolCall, ResponseInputItem } from "openai/resources/responses/responses";
import { MessageType } from "../chat_ui/types";
import { TokenUsage } from "../chat_ui/ResponseMetrics";
import { getGatewayBaseUrl, getProxyBaseUrl } from "@/components/networking";
import { toast } from "@/lib/toast";
import { extractPromptCacheTokens } from "@/utils/promptCacheUsage";
import { parseUsageCost } from "./usage_cost";
import { buildPlaygroundHeaders, type CustomHeaders } from "./request_headers";
import type { MCPEvent } from "@/components/mcp_tools/types";
import { MCPServer, MCPToolset } from "@/components/mcp_tools/types";
import {
  CodeInterpreterResult,
  CodeInterpreterState,
  handleCodeInterpreterCall,
  handleCodeInterpreterOutput,
} from "./code_interpreter_handler";

export type { CodeInterpreterResult } from "./code_interpreter_handler";

export interface ResponsesMessage extends MessageType {
  mcpEvents?: MCPEvent[];
}

interface ResponseOutputPart {
  type?: string;
  text?: string;
}

interface ResponseOutputItem {
  type?: string;
  content?: ResponseOutputPart[];
  summary?: ResponseOutputPart[];
}

interface NonStreamedResponse {
  output?: ResponseOutputItem[];
}

type SynthesizedResponseEvent =
  | { type: "response.output_item.done"; item: ResponseOutputItem }
  | { type: "response.reasoning.delta"; delta: string }
  | { type: "response.output_text.delta"; delta: string }
  | { type: "response.completed"; response: NonStreamedResponse };

const responseAsEvents = (response: NonStreamedResponse): SynthesizedResponseEvent[] => {
  const outputItems = response.output ?? [];
  const outputText = outputItems
    .filter((item) => item.type === "message")
    .flatMap((item) => item.content ?? [])
    .filter((part) => part.type === "output_text")
    .map((part) => part.text ?? "")
    .join("");
  const reasoningText = outputItems
    .filter((item) => item.type === "reasoning")
    .flatMap((item) => item.summary ?? [])
    .map((part) => part.text ?? "")
    .join("");

  return [
    ...outputItems.map((item) => ({ type: "response.output_item.done" as const, item })),
    ...(reasoningText ? [{ type: "response.reasoning.delta" as const, delta: reasoningText }] : []),
    ...(outputText ? [{ type: "response.output_text.delta" as const, delta: outputText }] : []),
    { type: "response.completed" as const, response },
  ];
};

export async function makeOpenAIResponsesRequest(
  messages: ResponsesMessage[],
  updateTextUI: (role: string, delta: string, model?: string) => void,
  selectedModel: string,
  accessToken: string | null,
  tags: string[] = [],
  signal?: AbortSignal,
  onReasoningContent?: (content: string) => void,
  onTimingData?: (timeToFirstToken: number) => void,
  onUsageData?: (usage: TokenUsage, toolName?: string) => void,
  traceId?: string,
  vector_store_ids?: string[],
  guardrails?: string[],
  policies?: string[],
  selectedMCPServers?: string[],
  previousResponseId?: string | null,
  onResponseId?: (responseId: string) => void,
  onMCPEvent?: (event: MCPEvent) => void,
  codeInterpreterEnabled?: boolean,
  onCodeInterpreterResult?: (result: CodeInterpreterResult) => void,
  customBaseUrl?: string,
  mcpServers?: MCPServer[],
  mcpServerToolRestrictions?: Record<string, string[]>,
  mcpToolsets?: MCPToolset[],
  streamingEnabled: boolean = true,
  onTotalLatency?: (latency: number) => void,
  customHeaders?: CustomHeaders,
  onResponseHeaders?: ResponseHeadersListener,
) {
  if (!accessToken) {
    throw new Error("Virtual Key is required");
  }

  if (!selectedModel || selectedModel.trim() === "") {
    throw new Error("Model is required. Please select a model before sending a request.");
  }

  // Base URL should be the current base_url
  const isLocal = process.env.NODE_ENV === "development";
  if (isLocal !== true) {
    console.log = function () {};
  }

  const proxyBaseUrl = customBaseUrl || getGatewayBaseUrl();
  const mcpBaseUrl = customBaseUrl || getProxyBaseUrl();
  const headers = buildPlaygroundHeaders(tags, customHeaders);

  const client = new openai.OpenAI({
    apiKey: accessToken,
    baseURL: proxyBaseUrl,
    dangerouslyAllowBrowser: true,
    defaultHeaders: headers,
    ...(onResponseHeaders ? { fetch: observeResponseHeaders(onResponseHeaders) } : {}),
  });

  try {
    const startTime = Date.now();
    let firstTokenReceived = false;
    let servedFromResponseCache = false;

    const formattedInput = messages.flatMap((message) => {
      const mcpInput =
        message.role === "assistant"
          ? (message.mcpEvents ?? []).flatMap<ResponseFunctionToolCall | ResponseInputItem.FunctionCallOutput>(
              (event) => {
                const item = event.item;
                const callId = item?.id || event.item_id;
                if (event.type !== "response.output_item.done" || item?.type !== "mcp_call" || !callId) {
                  return [];
                }
                if (!item.name || typeof item.arguments !== "string" || typeof item.output !== "string") {
                  return [];
                }
                return [
                  { type: "function_call", call_id: callId, name: item.name, arguments: item.arguments },
                  { type: "function_call_output", call_id: callId, output: item.output },
                ];
              },
            )
          : [];
      return [
        ...mcpInput,
        ...(mcpInput.length > 0 && message.content === ""
          ? []
          : [{ role: message.role, content: message.content, type: "message" }]),
      ];
    });

    // Build tools array
    const tools: any[] = [];

    // Add MCP servers if selected
    if (selectedMCPServers && selectedMCPServers.length > 0) {
      if (selectedMCPServers.includes("__all__")) {
        // All MCP Servers selected
        const allServersTool = {
          type: "mcp",
          server_label: "litellm",
          server_url: `${mcpBaseUrl}/mcp`,
          require_approval: "never",
        };
        tools.push(allServersTool);
      } else {
        // Individual servers/toolsets selected - create one entry per item
        selectedMCPServers.forEach((serverId) => {
          if (serverId.startsWith("toolset:")) {
            // Toolset: same /{name}/mcp pattern as individual servers
            const toolsetId = serverId.slice("toolset:".length);
            const toolset = mcpToolsets?.find((t) => t.toolset_id === toolsetId);
            const toolsetName = toolset?.toolset_name || toolsetId;
            const toolsetTool = {
              type: "mcp",
              server_label: toolsetName,
              server_url: `${mcpBaseUrl}/mcp/${encodeURIComponent(toolsetName)}`,
              require_approval: "never",
            };
            tools.push(toolsetTool);
          } else {
            const server = mcpServers?.find((s) => s.server_id === serverId);
            // Use server_name for both routing and labelling. server_name is the
            // unique registered identifier; aliases can collide across servers.
            const routeName = server?.server_name || serverId;
            const allowedTools = mcpServerToolRestrictions?.[serverId] || [];

            const serverTool = {
              type: "mcp",
              server_label: routeName, // unique per request — collisions cause silent tool-routing failures
              server_url: `${mcpBaseUrl}/mcp/${encodeURIComponent(routeName)}`,
              require_approval: "never",
              ...(allowedTools.length > 0 ? { allowed_tools: allowedTools } : {}),
            };
            tools.push(serverTool);
          }
        });
      }
    }

    // Add code_interpreter tool if enabled (OpenAI auto-creates container)
    if (codeInterpreterEnabled) {
      tools.push({
        type: "code_interpreter",
        container: { type: "auto" },
      });
    }

    const requestBody = {
      model: selectedModel,
      input: formattedInput,
      litellm_trace_id: traceId,
      ...(previousResponseId ? { previous_response_id: previousResponseId } : {}),
      ...(vector_store_ids ? { vector_store_ids } : {}),
      ...(guardrails ? { guardrails } : {}),
      ...(policies ? { policies } : {}),
      ...(tools.length > 0 ? { tools, tool_choice: "auto" } : {}),
    };

    // Create request to OpenAI responses API
    // Use 'any' type to avoid TypeScript issues with the experimental API
    const response = streamingEnabled
      ? await (client as any).responses.create({ ...requestBody, stream: true }, { signal })
      : await (async () => {
          const nonStreamingResponse = await (client as any).responses
            .create({ ...requestBody, stream: false }, { signal })
            .withResponse();
          servedFromResponseCache = nonStreamingResponse.response.headers.get("x-litellm-cache-key") !== null;
          return nonStreamingResponse.data;
        })();
    const events = streamingEnabled ? response : responseAsEvents(response);

    let mcpToolUsed = "";
    let codeInterpreterState: CodeInterpreterState = { code: "", containerId: "" };

    for await (const event of events) {
      // Use a type-safe approach to handle events
      if (typeof event === "object" && event !== null) {
        // Handle MCP events first
        if (
          event.type?.startsWith("response.mcp_") ||
          (event.type === "response.output_item.done" &&
            (event.item?.type === "mcp_list_tools" || event.item?.type === "mcp_call"))
        ) {
          if (onMCPEvent) {
            const mcpEvent: MCPEvent = {
              type: event.type,
              sequence_number: event.sequence_number,
              output_index: event.output_index,
              item_id: event.item_id || event.item?.id, // Handle both structures
              item: event.item,
              delta: event.delta,
              arguments: event.arguments,
              timestamp: Date.now(),
            };
            onMCPEvent(mcpEvent);
          }

          // Continue processing other aspects of the event
        }

        // Check for MCP tool usage
        if (event.type === "response.output_item.done" && event.item?.type === "mcp_call" && event.item?.name) {
          mcpToolUsed = event.item.name;
        }

        // Handle code interpreter events
        codeInterpreterState = handleCodeInterpreterCall(event, codeInterpreterState);
        handleCodeInterpreterOutput(event, codeInterpreterState, onCodeInterpreterResult);

        // Handle output text delta
        // 1) drop any "role" streams
        if (event.type === "response.role.delta") {
          continue;
        }

        // 2) only handle actual text deltas
        if (event.type === "response.output_text.delta" && typeof event.delta === "string") {
          const delta = event.delta;
          if (delta.length > 0) {
            updateTextUI("assistant", delta, selectedModel);

            // Calculate time to first token
            if (!firstTokenReceived) {
              firstTokenReceived = true;
              const timeToFirstToken = Date.now() - startTime;

              if (onTimingData && streamingEnabled) {
                onTimingData(timeToFirstToken);
              }
            }
          }
        }

        // Handle reasoning content
        if (event.type === "response.reasoning.delta" && "delta" in event) {
          const delta = event.delta;
          if (typeof delta === "string" && onReasoningContent) {
            onReasoningContent(delta);
          }
        }

        // Handle usage data at the response.completed event
        if (event.type === "response.completed" && "response" in event) {
          const response_obj = event.response;
          const usage = response_obj.usage;

          // Extract response_id for session management
          if (response_obj.id && onResponseId) {
            onResponseId(response_obj.id);
          }

          if (usage && onUsageData) {
            // Extract usage data safely
            const usageData: TokenUsage = {
              completionTokens: usage.output_tokens,
              promptTokens: usage.input_tokens,
              totalTokens: usage.total_tokens,
              ...extractPromptCacheTokens(usage),
              ...(servedFromResponseCache ? { servedFromResponseCache: true } : {}),
            };

            // Add reasoning tokens if available
            const reasoningTokens =
              usage.output_tokens_details?.reasoning_tokens ?? usage.completion_tokens_details?.reasoning_tokens;
            if (reasoningTokens) {
              usageData.reasoningTokens = reasoningTokens;
            }

            const parsedCost = parseUsageCost(usage.cost);
            if (parsedCost !== undefined) {
              usageData.cost = parsedCost;
            }

            onUsageData(usageData, mcpToolUsed);
          }
        }
      }
    }

    if (onTotalLatency) {
      onTotalLatency(Date.now() - startTime);
    }

    return response;
  } catch (error) {
    if (signal?.aborted) {
    } else {
      toast.fromError(`Error occurred while generating model response. Please try again. Error: ${error}`);
    }
    throw error; // Re-throw to allow the caller to handle the error
  }
}
