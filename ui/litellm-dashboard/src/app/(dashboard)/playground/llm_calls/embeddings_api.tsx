import NotificationManager from "@/components/molecules/notifications_manager";
import { getProxyBaseUrl, getGlobalLitellmHeaderName } from "@/components/networking";

export async function makeOpenAIEmbeddingsRequest(
  input: string,
  updateEmbeddingsUI: (embeddings: string, model?: string) => void,
  selectedModel: string,
  accessToken: string,
  tags?: string[],
  customBaseUrl?: string,
) {
  if (!accessToken) {
    throw new Error("Virtual Key is required");
  }

  // Base URL should be the current base_url
  const isLocal = process.env.NODE_ENV === "development";
  if (isLocal !== true) {
    console.log = function () {};
  }

  const proxyBaseUrl = customBaseUrl || getProxyBaseUrl();
  // Prepare headers with tags and trace ID
  const headers: Record<string, string> = {};
  if (tags && tags.length > 0) {
    headers["x-litellm-tags"] = tags.join(",");
  }

  try {
    const normalizedBaseUrl = proxyBaseUrl.endsWith("/") ? proxyBaseUrl.slice(0, -1) : proxyBaseUrl;
    const requestUrl = `${normalizedBaseUrl}/embeddings`;

    const response = await fetch(requestUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        [getGlobalLitellmHeaderName()]: `Bearer ${accessToken}`,
        ...headers,
      },
      body: JSON.stringify({
        model: selectedModel,
        input,
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(errorText || `Request failed with status ${response.status}`);
    }

    const responseData = await response.json();
    const embedding = responseData?.data?.[0]?.embedding;

    if (!embedding) {
      throw new Error("No embedding returned from server");
    }

    updateEmbeddingsUI(JSON.stringify(embedding), responseData?.model ?? selectedModel);
  } catch (error: unknown) {
    NotificationManager.fromBackend(
      `Error occurred while making embeddings request. Please try again. Error: ${error}`,
    );

    throw error; // Re-throw to allow the caller to handle the error
  }
}
