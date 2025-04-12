import openai from "openai";
import { ChatCompletionMessageParam } from "openai/resources/chat/completions";
import { message } from "antd";
import { processStreamingResponse } from "./process_stream";

export async function makeOpenAIChatCompletionRequest(
    chatHistory: { role: string; content: string }[],
    updateUI: (chunk: string, model?: string) => void,
    selectedModel: string,
    accessToken: string,
    tags?: string[],
    signal?: AbortSignal,
    onReasoningContent?: (content: string) => void
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
      apiKey: accessToken, // Replace with your OpenAI API key
      baseURL: proxyBaseUrl, // Replace with your OpenAI API base URL
      dangerouslyAllowBrowser: true, // using a temporary litellm proxy key
      defaultHeaders: tags && tags.length > 0 ? { 'x-litellm-tags': tags.join(',') } : undefined,
    });
  
    try {
      const response = await client.chat.completions.create({
        model: selectedModel,
        stream: true,
        messages: chatHistory as ChatCompletionMessageParam[],
      }, { signal });
  
      for await (const chunk of response) {
        console.log(chunk);
        // Process the chunk using our utility
        processStreamingResponse(chunk, {
          onContent: updateUI,
          onReasoningContent: onReasoningContent || (() => {})
        });
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
  