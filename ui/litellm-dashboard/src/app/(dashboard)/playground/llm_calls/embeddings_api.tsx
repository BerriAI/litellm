import { toast } from "@/lib/toast";
import { getProxyBaseUrl, getGlobalLitellmHeaderName } from "@/components/networking";
import { buildPlaygroundHeaders, type CustomHeaders } from "@/components/llm_calls/request_headers";

export async function makeOpenAIEmbeddingsRequest(
  input: string,
  updateEmbeddingsUI: (embeddings: string, model?: string) => void,
  selectedModel: string,
  accessToken: string,
  tags?: string[],
  customBaseUrl?: string,
  customHeaders?: CustomHeaders,
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
  const headers = buildPlaygroundHeaders(tags, customHeaders);

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
    toast.fromError(`Error occurred while making embeddings request. Please try again. Error: ${error}`);

    throw error; // Re-throw to allow the caller to handle the error
  }
}
