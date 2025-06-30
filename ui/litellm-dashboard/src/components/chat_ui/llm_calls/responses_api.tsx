import openai from "openai";
import { message } from "antd";
import { MessageType } from "../types";
import { TokenUsage } from "../ResponseMetrics";
import { getProxyBaseUrl } from "@/components/networking";

export async function makeOpenAIResponsesRequest(
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
  guardrails?: string[]
) {
  if (!accessToken) {
    throw new Error("API key is required");
  }

  // Base URL should be the current base_url
  const isLocal = process.env.NODE_ENV === "development";
  if (isLocal !== true) {
    console.log = function () {};
  }
  
  const proxyBaseUrl = getProxyBaseUrl()
  // Prepare headers with tags and trace ID
  const headers: Record<string, string> = {};
  if (tags && tags.length > 0) {
    headers['x-litellm-tags'] = tags.join(',');
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
    const formattedInput = messages.map(message => ({
      role: message.role,
      content: message.content,
      type: "message"
    }));

    // Create request to OpenAI responses API
    // Use 'any' type to avoid TypeScript issues with the experimental API
    const response = await (client as any).responses.create({
      model: selectedModel,
      input: formattedInput,
      stream: true,
      litellm_trace_id: traceId,
      ...(vector_store_ids ? { vector_store_ids } : {}),
      ...(guardrails ? { guardrails } : {}),
    }, { signal });

    for await (const event of response) {
      console.log("Response event:", event);
      
      // Use a type-safe approach to handle events
      if (typeof event === 'object' && event !== null) {
        // Handle output text delta
        // 1) drop any "role" streams
        if (event.type === "response.role.delta") {
            continue;
        }

        // 2) only handle actual text deltas
        if (event.type === "response.output_text.delta" && typeof event.delta === "string") {
            const delta = event.delta;
            console.log("Text delta", delta);
            // skip pure whitespace/newlines
            if (delta.trim().length > 0) {
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
        if (event.type === "response.reasoning.delta" && 'delta' in event) {
          const delta = event.delta;
          if (typeof delta === 'string' && onReasoningContent) {
            onReasoningContent(delta);
          }
        }
        
        // Handle usage data at the response.completed event
        if (event.type === "response.completed" && 'response' in event) {
          const response_obj = event.response;
          const usage = response_obj.usage;
          console.log("Usage data:", usage);
          if (usage && onUsageData) {
            console.log("Usage data:", usage);
            
            // Extract usage data safely
            const usageData: TokenUsage = {
              completionTokens: usage.output_tokens,
              promptTokens: usage.input_tokens,
              totalTokens: usage.total_tokens
            };
            
            // Add reasoning tokens if available
            if (usage.completion_tokens_details?.reasoning_tokens) {
              usageData.reasoningTokens = usage.completion_tokens_details.reasoning_tokens;
            }
            
            onUsageData(usageData);
          }
        }
      }
    }
  } catch (error) {
    if (signal?.aborted) {
      console.log("Responses API request was cancelled");
    } else {
      message.error(`Error occurred while generating model response. Please try again. Error: ${error}`, 20);
    }
    throw error; // Re-throw to allow the caller to handle the error
  }
} 