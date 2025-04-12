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
    onUsageData?: (usage: TokenUsage) => void
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
    const client = new openai.OpenAI({
      apiKey: accessToken,
      baseURL: proxyBaseUrl,
      dangerouslyAllowBrowser: true,
      defaultHeaders: tags && tags.length > 0 ? { 'x-litellm-tags': tags.join(',') } : undefined,
    });
  
    try {
      const startTime = Date.now();
      let firstTokenReceived = false;
      let timeToFirstToken: number | undefined = undefined;
      
      // For collecting complete response text
      let fullResponseContent = "";
      let fullReasoningContent = "";

      const response = await client.chat.completions.create({
        model: selectedModel,
        stream: true,
        messages: chatHistory as ChatCompletionMessageParam[],
      }, { signal });
  
      for await (const chunk of response) {
        console.log("Stream chunk:", chunk);
        
        // Measure time to first token
        if (!firstTokenReceived && chunk.choices[0]?.delta?.content) {
          firstTokenReceived = true;
          timeToFirstToken = Date.now() - startTime;
          if (onTimingData) {
            onTimingData(timeToFirstToken);
          }
        }
        
        // Process content
        if (chunk.choices[0]?.delta?.content) {
          const content = chunk.choices[0].delta.content;
          updateUI(content, chunk.model);
          fullResponseContent += content;
        }
        
        // Process reasoning content if present - using type assertion
        const delta = chunk.choices[0]?.delta as any;
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
      
      // Always create an estimated usage
      if (onUsageData) {
        try {
          console.log("Creating estimated usage data");
          // Create a simple usage estimate - approximately 4 characters per token
          const estimatedUsage: TokenUsage = {
            promptTokens: Math.ceil(JSON.stringify(chatHistory).length / 4), 
            completionTokens: Math.ceil((fullResponseContent.length) / 4),
            totalTokens: Math.ceil((JSON.stringify(chatHistory).length + fullResponseContent.length) / 4)
          };
          
          if (fullReasoningContent) {
            estimatedUsage.reasoningTokens = Math.ceil(fullReasoningContent.length / 4);
          }
          
          onUsageData(estimatedUsage);
        } catch (error) {
          console.error("Error estimating usage data:", error);
        }
      }
    } catch (error) {
      if (signal?.aborted) {
        console.log("Chat completion request was cancelled");
      } else {
        message.error(`Error occurred while generating model response. Please try again. Error: ${error}`, 20);
      }
      throw error; // Re-throw to allow the caller to handle the error
    }
  }
  