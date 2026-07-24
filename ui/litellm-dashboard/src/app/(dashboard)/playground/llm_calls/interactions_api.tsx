import NotificationManager from "@/components/molecules/notifications_manager";
import { getGlobalLitellmHeaderName, getProxyBaseUrl } from "@/components/networking";

export async function makeInteractionsRequest(
  input: string,
  updateUI: (text: string, model?: string) => void,
  selectedModel: string,
  accessToken: string,
  tags?: string[],
  signal?: AbortSignal,
  customBaseUrl?: string,
  previousInteractionId?: string,
): Promise<void> {
  if (!accessToken) {
    throw new Error("Virtual Key is required");
  }

  const isLocal = process.env.NODE_ENV === "development";
  if (isLocal !== true) {
    console.log = function () {};
  }

  const proxyBaseUrl = customBaseUrl || getProxyBaseUrl();
  const normalizedBaseUrl = proxyBaseUrl.endsWith("/") ? proxyBaseUrl.slice(0, -1) : proxyBaseUrl;
  const requestUrl = `${normalizedBaseUrl}/v1beta/interactions`;

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    [getGlobalLitellmHeaderName()]: `Bearer ${accessToken}`,
  };
  if (tags && tags.length > 0) {
    headers["x-litellm-tags"] = tags.join(",");
  }

  const body: Record<string, unknown> = {
    model: selectedModel,
    input,
    stream: true,
  };
  if (previousInteractionId) {
    body.previous_interaction_id = previousInteractionId;
  }

  try {
    const response = await fetch(requestUrl, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
      signal,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(errorText || `Request failed with status ${response.status}`);
    }

    if (!response.body) {
      throw new Error("No response body received");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let responseModel: string | undefined;
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // SSE lines are separated by double newlines; split on single newlines and
      // look for "data: " prefixed lines.
      const lines = buffer.split("\n");
      // Keep the last (potentially incomplete) line in the buffer
      buffer = lines.pop() ?? "";

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed.startsWith("data:")) continue;

        const jsonStr = trimmed.slice("data:".length).trim();
        if (!jsonStr || jsonStr === "[DONE]") continue;

        let event: Record<string, unknown>;
        try {
          event = JSON.parse(jsonStr);
        } catch {
          continue;
        }

        const eventType = event.event_type as string | undefined;

        if (eventType === "interaction.start" || eventType === "interaction.complete") {
          // Capture model from either the native Gemini shape (nested under
          // `interaction`) or the bridge shape (top-level `model` field).
          const interaction = event.interaction as Record<string, unknown> | undefined;
          if (typeof interaction?.model === "string" && interaction.model) {
            responseModel = interaction.model;
          } else if (typeof event.model === "string" && event.model) {
            responseModel = event.model;
          }
        } else if (eventType === "content.delta" || eventType === "content.start") {
          const delta = event.delta as Record<string, unknown> | undefined;
          // Accept both native Gemini format {"type":"text","text":"..."} and bridge
          // format {"text":"..."} (no type discriminator)
          if (typeof delta?.text === "string" && delta.text) {
            updateUI(delta.text, responseModel ?? selectedModel);
          }
        }
        // content.start, content.stop, interaction.status_update — no UI action needed
      }
    }
  } catch (error: unknown) {
    if (signal?.aborted) {
      throw error;
    }
    NotificationManager.fromBackend(`Error occurred while making Interactions API request. Error: ${error}`);
    throw error;
  }
}
