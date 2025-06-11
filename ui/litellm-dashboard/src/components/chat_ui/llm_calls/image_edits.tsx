import openai from "openai";
import { message } from "antd";
import { getProxyBaseUrl } from "@/components/networking";

export async function makeOpenAIImageEditsRequest(
  imageFile: File,
  prompt: string,
  updateUI: (imageUrl: string, model: string) => void,
  selectedModel: string,
  accessToken: string,
  tags?: string[],
  signal?: AbortSignal
) {
  // base url should be the current base_url
  const isLocal = process.env.NODE_ENV === "development";
  if (isLocal !== true) {
    console.log = function () {};
  }
  console.log("isLocal:", isLocal);
  const proxyBaseUrl = getProxyBaseUrl()
  
  const client = new openai.OpenAI({
    apiKey: accessToken,
    baseURL: proxyBaseUrl,
    dangerouslyAllowBrowser: true,
    defaultHeaders: tags && tags.length > 0 ? { 'x-litellm-tags': tags.join(',') } : undefined,
  });

  try {
    const response = await client.images.edit({
      model: selectedModel,
      image: imageFile,
      prompt: prompt,
    }, { signal });

    console.log(response.data);
    
    if (response.data && response.data[0]) {
      // Handle either URL or base64 data from response
      if (response.data[0].url) {
        // Use the URL directly
        updateUI(response.data[0].url, selectedModel);
      } else if (response.data[0].b64_json) {
        // Convert base64 to data URL format
        const base64Data = response.data[0].b64_json;
        updateUI(`data:image/png;base64,${base64Data}`, selectedModel);
      } else {
        throw new Error("No image data found in response");
      }
    } else {
      throw new Error("Invalid response format");
    }
  } catch (error) {
    if (signal?.aborted) {
      console.log("Image edits request was cancelled");
    } else {
      message.error(`Error occurred while editing image. Please try again. Error: ${error}`, 20);
    }
    throw error; // Re-throw to allow the caller to handle the error
  }
} 