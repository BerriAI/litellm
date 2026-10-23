import { makeOpenAIResponsesRequest } from "@/components/llm_calls/responses_api";
import { generateChatImages, toResponsesMessages } from "./imageHelpers";
import type { AssistantMessageUpdate, ChatMessage, MCPEvent } from "./types";

interface ChatRequestOptions {
  model: string;
  mode: "chat" | "image";
  prompt: string;
  accessToken: string;
  signal: AbortSignal;
  messages: Pick<ChatMessage, "role" | "content" | "attachments" | "generatedImages" | "imageGeneration">[];
  previousResponseId?: string | null;
  mcpServers?: string[];
  onResponseId: (id: string | null) => void;
  onUpdate: (update: AssistantMessageUpdate) => void;
}

export async function runChatRequest({
  model,
  mode,
  prompt,
  accessToken,
  signal,
  messages,
  previousResponseId,
  mcpServers,
  onResponseId,
  onUpdate,
}: ChatRequestOptions): Promise<void> {
  let content = "";
  let reasoningContent = "";
  const mcpEvents: MCPEvent[] = [];

  try {
    signal.throwIfAborted();
    if (mode === "image") {
      const imageRequest = { prompt, model, accessToken, signal };
      const generatedImages = await generateChatImages(imageRequest);
      signal.throwIfAborted();
      onUpdate({ generatedImages });
      return;
    }
    await makeOpenAIResponsesRequest(
      toResponsesMessages(messages),
      (_role, chunk) => {
        content += chunk;
        onUpdate({ content });
      },
      model,
      accessToken,
      undefined,
      signal,
      (chunk) => {
        reasoningContent += chunk;
        onUpdate({ reasoningContent });
      },
      (timeToFirstToken) => onUpdate({ timeToFirstToken }),
      (usage) => onUpdate({ usage }),
      undefined,
      undefined,
      undefined,
      undefined,
      mcpServers?.length ? mcpServers : undefined,
      previousResponseId,
      (id) => {
        if (!signal.aborted) onResponseId(id);
      },
      (event) => mcpEvents.push(event),
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      true,
      (totalLatency) => onUpdate({ totalLatency }),
    );
    signal.throwIfAborted();
    if (mcpEvents.length) onUpdate({ mcpEvents });
  } catch (error: unknown) {
    onResponseId(null);
    const stopped = signal.aborted || (error instanceof Error && error.name === "AbortError");
    const errorMessage = error instanceof Error ? error.message : "Something went wrong. Please try again.";
    onUpdate({ content: stopped ? content + " [stopped]" : content || errorMessage });
  }
}
