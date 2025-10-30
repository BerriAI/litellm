import NotificationManager from "@/components/molecules/notifications_manager";
import { getProxyBaseUrl } from "@/components/networking";
import openai from "openai";

export async function makeOpenAIEmbeddingsRequest(
  input: string,
  updateEmbeddingsUI: (embeddings: string, model?: string) => void,
  selectedModel: string,
  accessToken: string,
  tags?: string[],
) {
  if (!accessToken) {
    throw new Error("API key is required");
  }

  // Base URL should be the current base_url
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

  const client = new openai.OpenAI({
    apiKey: accessToken,
    baseURL: proxyBaseUrl,
    dangerouslyAllowBrowser: true,
    defaultHeaders: headers,
  });

  try {
    const response = await client.embeddings.create({
      model: selectedModel,
      input: input,
    });

    updateEmbeddingsUI(JSON.stringify(response.data[0].embedding), selectedModel);
  } catch (error: unknown) {
    NotificationManager.fromBackend(
      `Error occurred while making embeddings request. Please try again. Error: ${error}`,
    );

    throw error; // Re-throw to allow the caller to handle the error
  }
}
