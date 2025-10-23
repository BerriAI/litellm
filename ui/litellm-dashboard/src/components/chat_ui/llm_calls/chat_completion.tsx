import openai from "openai";
import { ChatCompletionMessageParam } from "openai/resources/chat/completions";
import { TokenUsage } from "../ResponseMetrics";
import { VectorStoreSearchResponse } from "../types";
import { getProxyBaseUrl } from "@/components/networking";

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
  selectedMCPTools?: string[],
  onImageGenerated?: (imageUrl: string, model?: string) => void,
  onSearchResults?: (searchResults: VectorStoreSearchResponse[]) => void,
) {
  // base url should be the current base_url
  const isLocal = process.env.NODE_ENV === "development";
  if (isLocal !== true) {
    console.log = function () {};
  }
  console.log("isLocal:", isLocal);
  const proxyBaseUrl = getProxyBaseUrl();
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
        ...(tools ? { tools, tool_choice: "auto" } : {}),
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

        onUsageData(usageData);
      }
    }
  } catch (error) {
    if (signal?.aborted) {
      console.log("Chat completion request was cancelled");
    }
    throw error; // Re-throw to allow the caller to handle the error
  }
}
