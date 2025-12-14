import Anthropic from "@anthropic-ai/sdk";
import { MessageType } from "../chat_ui/types";
import { TokenUsage } from "../chat_ui/ResponseMetrics";
import { getProxyBaseUrl } from "@/components/networking";
import NotificationManager from "@/components/molecules/notifications_manager";

export async function makeAnthropicMessagesRequest(
  messages: MessageType[],
  updateTextUI: (role: string, delta: string, model?: string) => void,
  selectedModel: string,
  accessToken: string | null,
  tags: string[] = [],
  signal?: AbortSignal,
  onReasoningContent?: (content: string) => void,
  onTimingData?: (timeToFirstToken: number) => void,
  onUsageData?: (usage: TokenUsage) => void,
  traceId?: string,
  vector_store_ids?: string[],
  guardrails?: string[],
  selectedMCPTools?: string[],
) {
  if (!accessToken) {
    throw new Error("Virtual Key is required");
  }

  const isLocal = process.env.NODE_ENV === "development";
  if (isLocal !== true) {
    console.log = function () {};
  }

  const proxyBaseUrl = getProxyBaseUrl();

  // Prepare headers with tags and trace ID
  const headers: Record<string, string> = {};
  if (tags && tags.length > 0) {
    headers["x-litellm-tags"] = tags.join(",");
  }

  const client = new Anthropic({
    apiKey: accessToken,
    baseURL: proxyBaseUrl,
    dangerouslyAllowBrowser: true,
    defaultHeaders: headers,
  });

  try {
    const startTime = Date.now();
    let firstTokenReceived = false;

    // Format MCP tools if selected
    const tools =
      selectedMCPTools && selectedMCPTools.length > 0
        ? [
            {
              type: "mcp",
              server_label: "litellm",
              server_url: `${proxyBaseUrl}/mcp`,
              require_approval: "never",
              allowed_tools: selectedMCPTools,
              headers: {
                "x-litellm-api-key": `Bearer ${accessToken}`,
              },
            },
          ]
        : undefined;

    const requestBody: any = {
      model: selectedModel,
      messages: messages.map((m) => ({ role: m.role, content: m.content })),
      stream: true,
      max_tokens: 1024,
      // @ts-ignore - litellm specific parameter
      litellm_trace_id: traceId,
    };

    if (vector_store_ids) requestBody.vector_store_ids = vector_store_ids;
    if (guardrails) requestBody.guardrails = guardrails;
    if (tools) {
      requestBody.tools = tools;
      requestBody.tool_choice = "auto";
    }

    // Use the streaming helper method for cleaner async iteration
    // @ts-ignore - The SDK types might not include all litellm-specific parameters
    const stream = client.messages.stream(requestBody, { signal });

    for await (const messageStreamEvent of stream) {
      console.log("Stream event:", messageStreamEvent);

      // Process content block deltas
      if (messageStreamEvent.type === "content_block_delta") {
        const delta = messageStreamEvent.delta;

        // Measure time to first token
        if (!firstTokenReceived) {
          firstTokenReceived = true;
          const timeToFirstToken = Date.now() - startTime;
          console.log("First token received! Time:", timeToFirstToken, "ms");
          if (onTimingData) {
            onTimingData(timeToFirstToken);
          }
        }

        // Handle different types of deltas
        if (delta.type === "text_delta") {
          updateTextUI("assistant", delta.text, selectedModel);
        }
        // @ts-ignore - reasoning_content might not be in the official types yet
        else if (delta.type === "reasoning_delta" && onReasoningContent) {
          // @ts-ignore
          onReasoningContent(delta.text);
        }
      }

      // Process usage data from message_delta events
      if (messageStreamEvent.type === "message_delta" && (messageStreamEvent as any).usage && onUsageData) {
        const usage = (messageStreamEvent as any).usage;
        console.log("Usage data found:", usage);
        const usageData: TokenUsage = {
          completionTokens: usage.output_tokens,
          promptTokens: usage.input_tokens,
          totalTokens: usage.input_tokens + usage.output_tokens,
        };
        onUsageData(usageData);
      }
    }
  } catch (error) {
    if (signal?.aborted) {
      console.log("Anthropic messages request was cancelled");
    } else {
      NotificationManager.fromBackend(
        `Error occurred while generating model response. Please try again. Error: ${error}`,
      );
    }
    throw error;
  }
}
