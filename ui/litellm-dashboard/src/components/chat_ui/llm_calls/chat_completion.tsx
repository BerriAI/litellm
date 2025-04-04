import openai from "openai";
import { ChatCompletionMessageParam } from "openai/resources/chat/completions";
import { message } from "antd";

export async function makeOpenAIChatCompletionRequest(
    chatHistory: { role: string; content: string }[],
    updateUI: (chunk: string, model: string) => void,
    selectedModel: string,
    accessToken: string,
    signal?: AbortSignal
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
    });
  
    try {
      const response = await client.chat.completions.create({
        model: selectedModel,
        stream: true,
        messages: chatHistory as ChatCompletionMessageParam[],
      }, { signal });
  
      for await (const chunk of response) {
        console.log(chunk);
        if (chunk.choices[0].delta.content) {
          updateUI(chunk.choices[0].delta.content, chunk.model);
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
  