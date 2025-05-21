import openai from "openai";
import { ChatCompletionMessageParam } from "openai/resources/chat/completions";
import { message } from "antd";
import { TokenUsage } from "../ResponseMetrics";

export async function makeOpenAIChatCompletionRequest(
    chatHistory: { role: string; content: string }[],
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
    guardrails?: string[]
  ) {
    // base url should be the current base_url
    const isLocal = process.env.NODE_ENV === "development";
    if (isLocal !== true) {
      console.log = function () {};
    }
    console.log("isLocal:", isLocal);
    const proxyBaseUrl = isLocal
      ? "http://localhost:4000"
      : window.location.origin;
      
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
      let timeToFirstToken: number | undefined = undefined;
      
      // For collecting complete response text
      let fullResponseContent = "";
      let fullReasoningContent = "";
      
      // @ts-ignore
      const response = await client.chat.completions.create({
        model: selectedModel,
        stream: true,
        stream_options: {
          include_usage: true,
        },
        litellm_trace_id: traceId, 
        messages: chatHistory as ChatCompletionMessageParam[],
        ...(vector_store_ids ? { vector_store_ids } : {}),
        ...(guardrails ? { guardrails } : {}),
      }, { signal });
  
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
        
        // Process reasoning content if present - using type assertion
        if (delta && delta.reasoning_content) {
          const reasoningContent = delta.reasoning_content;
          if (onReasoningContent) {
            onReasoningContent(reasoningContent);
          }
          fullReasoningContent += reasoningContent;
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
  