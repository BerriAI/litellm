import openai from "openai";
import { MessageType } from "../chat_ui/types";
import { TokenUsage } from "../chat_ui/ResponseMetrics";
import { getProxyBaseUrl } from "@/components/networking";
import NotificationManager from "@/components/molecules/notifications_manager";
import type { MCPEvent } from "../../mcp_tools/types";
import { MCPServer, MCPToolset } from "../../mcp_tools/types";
import {
  CodeInterpreterResult,
  CodeInterpreterState,
  handleCodeInterpreterCall,
  handleCodeInterpreterOutput,
} from "./code_interpreter_handler";

export type { CodeInterpreterResult } from "./code_interpreter_handler";

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

    // Build tools array
    const tools: any[] = [];

    // Add MCP servers if selected
    if (selectedMCPServers && selectedMCPServers.length > 0) {
      if (selectedMCPServers.includes("__all__")) {
        // All MCP Servers selected
        tools.push({
          type: "mcp",
          server_label: "litellm",
          server_url: `${proxyBaseUrl}/mcp`,
          require_approval: "never",
        });
      } else {
        // Individual servers/toolsets selected - create one entry per item
        selectedMCPServers.forEach((serverId) => {
          if (serverId.startsWith("toolset:")) {
            // Toolset: same /{name}/mcp pattern as individual servers
            const toolsetId = serverId.slice("toolset:".length);
            const toolset = mcpToolsets?.find((t) => t.toolset_id === toolsetId);
            const toolsetName = toolset?.toolset_name || toolsetId;
            tools.push({
              type: "mcp",
              server_label: toolsetName,
              server_url: `${proxyBaseUrl}/mcp/${encodeURIComponent(toolsetName)}`,
              require_approval: "never",
            });
          } else {
            const server = mcpServers?.find((s) => s.server_id === serverId);
            // Use server_name for both routing and labelling. server_name is the
            // unique registered identifier; aliases can collide across servers.
            const routeName = server?.server_name || serverId;
            const allowedTools = mcpServerToolRestrictions?.[serverId] || [];

            tools.push({
              type: "mcp",
              server_label: routeName, // unique per request — collisions cause silent tool-routing failures
              server_url: `${proxyBaseUrl}/mcp/${encodeURIComponent(routeName)}`,
              require_approval: "never",
              ...(allowedTools.length > 0 ? { allowed_tools: allowedTools } : {}),
            });
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

    // Create request to OpenAI responses API
    // Use 'any' type to avoid TypeScript issues with the experimental API
    const response = await (client as any).responses.create(
      {
        model: selectedModel,
        input: formattedInput,
        stream: true,
        litellm_trace_id: traceId,
        ...(previousResponseId ? { previous_response_id: previousResponseId } : {}),
        ...(vector_store_ids ? { vector_store_ids } : {}),
        ...(guardrails ? { guardrails } : {}),
        ...(policies ? { policies } : {}),
        ...(tools.length > 0 ? { tools, tool_choice: "auto" } : {}),
      },
      { signal },
    );

    let mcpToolUsed = "";
    let codeInterpreterState: CodeInterpreterState = { code: "", containerId: "" };

    for await (const event of response) {
      console.log("Response event:", event);

      // Use a type-safe approach to handle events
      if (typeof event === "object" && event !== null) {
        // Handle MCP events first
        if (
          event.type?.startsWith("response.mcp_") ||
          (event.type === "response.output_item.done" &&
            (event.item?.type === "mcp_list_tools" || event.item?.type === "mcp_call"))
        ) {
          console.log("MCP event received:", event);

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
          console.log("MCP tool used:", mcpToolUsed);
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
          console.log("Text delta", delta);
          if (delta.length > 0) {
            updateTextUI("assistant", delta, selectedModel);

            // Calculate time to first token
            if (!firstTokenReceived) {
              firstTokenReceived = true;
              const timeToFirstToken = Date.now() - startTime;
              console.log("First token received! Time:", timeToFirstToken, "ms");

              if (onTimingData) {
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
          console.log("Usage data:", usage);
          console.log("Response completed event:", response_obj);

          // Extract response_id for session management
          if (response_obj.id && onResponseId) {
            console.log("Response ID for session management:", response_obj.id);
            onResponseId(response_obj.id);
          }

          if (usage && onUsageData) {
            console.log("Usage data:", usage);

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

    return response;
  } catch (error) {
    if (signal?.aborted) {
      console.log("Responses API request was cancelled");
    } else {
      NotificationManager.fromBackend(
        `Error occurred while generating model response. Please try again. Error: ${error}`,
      );
    }
    throw error; // Re-throw to allow the caller to handle the error
  }
}
