import openai from "openai";
import { MessageType } from "../chat_ui/types";
import { TokenUsage } from "../chat_ui/ResponseMetrics";
import { getProxyBaseUrl } from "@/components/networking";
import NotificationManager from "@/components/molecules/notifications_manager";
import type { MCPEvent } from "@/components/mcp_tools/types";
import { MCPServer, MCPToolset } from "@/components/mcp_tools/types";
import {
  CodeInterpreterResult,
  CodeInterpreterState,
  handleCodeInterpreterCall,
  handleCodeInterpreterOutput,
} from "./code_interpreter_handler";
import { buildMcpToolBlocks } from "./mcp_tool_blocks";

export type { CodeInterpreterResult } from "./code_interpreter_handler";

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
  messages: MessageType[],
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

  const proxyBaseUrl = customBaseUrl || getProxyBaseUrl();
  // Prepare headers with tags and trace ID
  const headers: Record<string, string> = {};
  if (tags && tags.length > 0) {
    headers["x-litellm-tags"] = tags.join(",");
  }

  const client = new openai.OpenAI({
    apiKey: accessToken,
    baseURL: proxyBaseUrl,
    dangerouslyAllowBrowser: true,
    defaultHeaders: headers,
  });

  try {
    const startTime = Date.now();
    let firstTokenReceived = false;

    // Format messages for the API
    const formattedInput = messages.map((message) => {
      // If content is already an array (multimodal), use it directly
      if (Array.isArray(message.content)) {
        return {
          role: message.role,
          content: message.content,
          type: "message",
        };
      }
      // Otherwise, wrap text content in the expected format
      return {
        role: message.role,
        content: message.content,
        type: "message",
      };
    });

    const tools: Array<Record<string, unknown>> = [
      ...buildMcpToolBlocks({
        selectedMCPServers,
        mcpServers,
        mcpToolsets,
        mcpServerToolRestrictions,
      }),
    ];

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
    const response = await (client as any).responses.create({ ...requestBody, stream: streamingEnabled }, { signal });
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
            };

            // Add reasoning tokens if available
            if (usage.completion_tokens_details?.reasoning_tokens) {
              usageData.reasoningTokens = usage.completion_tokens_details.reasoning_tokens;
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
      NotificationManager.fromBackend(
        `Error occurred while generating model response. Please try again. Error: ${error}`,
      );
    }
    throw error; // Re-throw to allow the caller to handle the error
  }
}
