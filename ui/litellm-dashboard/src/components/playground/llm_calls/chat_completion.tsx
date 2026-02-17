import openai from "openai";
import { ChatCompletionMessageParam } from "openai/resources/chat/completions";
import { TokenUsage } from "../chat_ui/ResponseMetrics";
import { VectorStoreSearchResponse } from "../chat_ui/types";
import { getProxyBaseUrl } from "@/components/networking";
import { MCPServer } from "../../mcp_tools/types";
import { MCPEvent } from "../chat_ui/MCPEventsDisplay";

export async function makeOpenAIChatCompletionRequest(
  chatHistory: { role: string; content: string | any[] }[],
  updateUI: (chunk: string, model?: string) => void,
  selectedModel: string,
  accessToken: string,
  tags?: string[],
  signal?: AbortSignal,
  onReasoningContent?: (content: string) => void,
  onTimingData?: (timeToFirstToken: number) => void,
  onUsageData?: (usage: TokenUsage) => void,
  traceId?: string,
  vector_store_ids?: string[],
  guardrails?: string[],
  policies?: string[],
  selectedMCPServers?: string[],
  onImageGenerated?: (imageUrl: string, model?: string) => void,
  onSearchResults?: (searchResults: VectorStoreSearchResponse[]) => void,
  temperature?: number,
  max_tokens?: number,
  onTotalLatency?: (latency: number) => void,
  customBaseUrl?: string,
  mcpServers?: MCPServer[],
  mcpServerToolRestrictions?: Record<string, string[]>,
  onMCPEvent?: (event: MCPEvent) => void,
  mockTestFallbacks?: boolean,
) {
  // base url should be the current base_url
  const isLocal = process.env.NODE_ENV === "development";
  if (isLocal !== true) {
    console.log = function () { };
  }
  console.log("isLocal:", isLocal);
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
    let timeToFirstToken: number | undefined = undefined;

    // For collecting complete response text
    let fullResponseContent = "";
    let fullReasoningContent = "";

    // Track MCP metadata cumulatively across chunks
    let mcpMetadata: {
      mcp_list_tools?: any[];
      mcp_tool_calls?: any[];
      mcp_call_results?: any[];
    } = {};
    let mcpListToolsProcessed = false;

    // Build tools array
    const tools: any[] = [];

    // Add MCP servers if selected
    if (selectedMCPServers && selectedMCPServers.length > 0) {
      if (selectedMCPServers.includes("__all__")) {
        // All MCP Servers selected
        tools.push({
          type: "mcp",
          server_label: "litellm",
          server_url: "litellm_proxy/mcp",
          require_approval: "never",
        });
      } else {
        // Individual servers selected - create one entry per server
        selectedMCPServers.forEach((serverId) => {
          const server = mcpServers?.find((s) => s.server_id === serverId);
          const serverName = server?.alias || server?.server_name || serverId;
          const allowedTools = mcpServerToolRestrictions?.[serverId] || [];

          tools.push({
            type: "mcp",
            server_label: "litellm",
            server_url: `litellm_proxy/mcp/${serverName}`,
            require_approval: "never",
            ...(allowedTools.length > 0 ? { allowed_tools: allowedTools } : {}),
          });
        });
      }
    }

    // @ts-ignore
    const response = await client.chat.completions.create(
      {
        model: selectedModel,
        stream: true,
        stream_options: {
          include_usage: true,
        },
        litellm_trace_id: traceId,
        messages: chatHistory as ChatCompletionMessageParam[],
        ...(vector_store_ids ? { vector_store_ids } : {}),
        ...(guardrails ? { guardrails } : {}),
        ...(policies ? { policies } : {}),
        ...(tools.length > 0 ? { tools, tool_choice: "auto" } : {}),
        ...(temperature !== undefined ? { temperature } : {}),
        ...(max_tokens !== undefined ? { max_tokens } : {}),
        ...(mockTestFallbacks ? { mock_testing_fallbacks: true } : {}),
      },
      { signal },
    );

    for await (const chunk of response) {
      console.log("Stream chunk:", chunk);

      // Process content and measure time to first token
      const delta = chunk.choices[0]?.delta as any;

      // Debug what's in the delta
      console.log("Delta content:", chunk.choices[0]?.delta?.content);
      console.log("Delta reasoning content:", delta?.reasoning_content);

      // Measure time to first token for either content or reasoning_content
      if (!firstTokenReceived && (chunk.choices[0]?.delta?.content || (delta && delta.reasoning_content))) {
        firstTokenReceived = true;
        timeToFirstToken = Date.now() - startTime;
        console.log("First token received! Time:", timeToFirstToken, "ms");
        if (onTimingData) {
          console.log("Calling onTimingData with:", timeToFirstToken);
          onTimingData(timeToFirstToken);
        } else {
          console.log("onTimingData callback is not defined!");
        }
      }

      // Process content
      if (chunk.choices[0]?.delta?.content) {
        const content = chunk.choices[0].delta.content;
        updateUI(content, chunk.model);
        fullResponseContent += content;
      }

      // Process image generation if present
      if (delta && delta.image && onImageGenerated) {
        console.log("Image generated:", delta.image);
        onImageGenerated(delta.image.url, chunk.model);
      }

      // Process reasoning content if present - using type assertion
      if (delta && delta.reasoning_content) {
        const reasoningContent = delta.reasoning_content;
        if (onReasoningContent) {
          onReasoningContent(reasoningContent);
        }
        fullReasoningContent += reasoningContent;
      }

      // Check for search results in provider_specific_fields
      if (delta && delta.provider_specific_fields?.search_results && onSearchResults) {
        console.log("Search results found:", delta.provider_specific_fields.search_results);
        onSearchResults(delta.provider_specific_fields.search_results);
      }

      // Check for MCP metadata in provider_specific_fields
      if (delta && delta.provider_specific_fields) {
        const providerFields = delta.provider_specific_fields;
        
        // Merge MCP metadata cumulatively (don't overwrite)
        if (providerFields.mcp_list_tools && !mcpMetadata.mcp_list_tools) {
          mcpMetadata.mcp_list_tools = providerFields.mcp_list_tools;
          // Process mcp_list_tools immediately when found (typically in first chunk)
          if (onMCPEvent && !mcpListToolsProcessed) {
            mcpListToolsProcessed = true;
            const toolsEvent: MCPEvent = {
              type: "response.output_item.done",
              item_id: "mcp_list_tools", // Add item_id to prevent duplicate detection issues
              item: {
                type: "mcp_list_tools",
                tools: providerFields.mcp_list_tools.map((tool: any) => ({
                  name: tool.function?.name || tool.name || "",
                  description: tool.function?.description || tool.description || "",
                  input_schema: tool.function?.parameters || tool.input_schema || {},
                })),
              },
              timestamp: Date.now(),
            };
            onMCPEvent(toolsEvent);
            console.log("MCP list_tools event sent:", toolsEvent);
          }
        }
        
        if (providerFields.mcp_tool_calls) {
          mcpMetadata.mcp_tool_calls = providerFields.mcp_tool_calls;
        }
        
        if (providerFields.mcp_call_results) {
          mcpMetadata.mcp_call_results = providerFields.mcp_call_results;
        }
        
        if (providerFields.mcp_list_tools || providerFields.mcp_tool_calls || providerFields.mcp_call_results) {
          console.log("MCP metadata found in chunk:", {
            mcp_list_tools: providerFields.mcp_list_tools ? "present" : "absent",
            mcp_tool_calls: providerFields.mcp_tool_calls ? "present" : "absent",
            mcp_call_results: providerFields.mcp_call_results ? "present" : "absent",
          });
        }
      }

      // Check for usage data using type assertion
      const chunkWithUsage = chunk as any;
      if (chunkWithUsage.usage && onUsageData) {
        console.log("Usage data found:", chunkWithUsage.usage);
        const usageData: TokenUsage = {
          completionTokens: chunkWithUsage.usage.completion_tokens,
          promptTokens: chunkWithUsage.usage.prompt_tokens,
          totalTokens: chunkWithUsage.usage.total_tokens,
        };

        // Check for reasoning tokens
        if (chunkWithUsage.usage.completion_tokens_details?.reasoning_tokens) {
          usageData.reasoningTokens = chunkWithUsage.usage.completion_tokens_details.reasoning_tokens;
        }

        // Extract cost from usage object if available
        if (chunkWithUsage.usage.cost !== undefined && chunkWithUsage.usage.cost !== null) {
          usageData.cost = parseFloat(chunkWithUsage.usage.cost);
        }

        onUsageData(usageData);
      }
    }

    // Process remaining MCP metadata (mcp_tool_calls and mcp_call_results) after stream completes
    // Note: mcp_list_tools is already processed when found in the first chunk
    if (onMCPEvent && (mcpMetadata.mcp_tool_calls || mcpMetadata.mcp_call_results)) {
      // Convert mcp_tool_calls and mcp_call_results to MCPEvent[]
      if (mcpMetadata.mcp_tool_calls && mcpMetadata.mcp_tool_calls.length > 0) {
        mcpMetadata.mcp_tool_calls.forEach((toolCall: any, index: number) => {
          const functionName = toolCall.function?.name || toolCall.name || "";
          const functionArgs = toolCall.function?.arguments || toolCall.arguments || "{}";

          // Find corresponding result
          const result = mcpMetadata.mcp_call_results?.find(
            (r: any) => r.tool_call_id === toolCall.id || r.tool_call_id === toolCall.call_id
          ) || mcpMetadata.mcp_call_results?.[index];

          const callEvent: MCPEvent = {
            type: "response.output_item.done",
            item: {
              type: "mcp_call",
              name: functionName,
              arguments: typeof functionArgs === "string" ? functionArgs : JSON.stringify(functionArgs),
              output: result?.result ? (typeof result.result === "string" ? result.result : JSON.stringify(result.result)) : undefined,
            },
            item_id: toolCall.id || toolCall.call_id,
            timestamp: Date.now(),
          };
          onMCPEvent(callEvent);
          console.log("MCP call event sent:", callEvent);
        });
      }
    }

    const endTime = Date.now();
    const totalLatency = endTime - startTime;
    if (onTotalLatency) {
      onTotalLatency(totalLatency);
    }
  } catch (error) {
    if (signal?.aborted) {
      console.log("Chat completion request was cancelled");
    }
    throw error; // Re-throw to allow the caller to handle the error
  }
}
