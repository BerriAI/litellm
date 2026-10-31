import OpenAI from "openai";
import type { Response } from "openai/resources/responses/responses";
import { getGatewayBaseUrl } from "@/components/networking";
import { extractPromptCacheTokens } from "@/utils/promptCacheUsage";
import type { AssistantMessageUpdate, ChatBackgroundRun, MCPEvent } from "./types";

export const isBackgroundRunPending = (run?: ChatBackgroundRun): boolean =>
  run?.status === "queued" || run?.status === "in_progress";

export function backgroundResponseUpdate(response: Response): AssistantMessageUpdate {
  const output = response.output ?? [];
  const content = output
    .filter((item) => item.type === "message")
    .flatMap((item) => item.content)
    .map((part) => (part.type === "output_text" ? part.text : part.refusal))
    .join("");
  const reasoningContent = output
    .filter((item) => item.type === "reasoning")
    .flatMap((item) => item.summary ?? [])
    .map((part) => part.text)
    .join("\n\n");
  const mcpEvents: MCPEvent[] = output.flatMap<MCPEvent>((item) => {
    if (item.type === "mcp_call") {
      return [{ type: "response.output_item.done", item: { ...item, output: item.output ?? undefined } }];
    }
    if (item.type === "mcp_list_tools") {
      return [
        {
          type: "response.output_item.done",
          item: {
            ...item,
            tools: item.tools.map((tool) => ({
              name: tool.name,
              input_schema: tool.input_schema,
              description: tool.description ?? "",
            })),
          },
        },
      ];
    }
    return [];
  });
  const status = response.status ?? "in_progress";
  const errors: Partial<Record<ChatBackgroundRun["status"], string>> = {
    failed: response.error?.message || "The background run failed.",
    incomplete: "The run ended before completing its answer.",
  };
  const backgroundError = errors[status];
  return {
    content,
    reasoningContent,
    mcpEvents,
    backgroundRun: { id: response.id, status },
    backgroundError,
    ...(response.usage && {
      usage: {
        promptTokens: response.usage.input_tokens,
        completionTokens: response.usage.output_tokens,
        totalTokens: response.usage.total_tokens,
        reasoningTokens: response.usage.output_tokens_details?.reasoning_tokens,
        ...extractPromptCacheTokens(response.usage),
      },
    }),
  };
}

const backgroundClient = (accessToken: string) =>
  new OpenAI({
    apiKey: accessToken,
    baseURL: getGatewayBaseUrl(),
    dangerouslyAllowBrowser: true,
  });

function waitForPoll(signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const abort = () => {
      clearTimeout(timer);
      reject(signal.reason);
    };
    const timer = setTimeout(() => {
      signal.removeEventListener("abort", abort);
      resolve();
    }, 1000);
    signal.addEventListener("abort", abort, { once: true });
    if (signal.aborted) abort();
  });
}

export async function watchBackgroundChat({
  accessToken,
  responseId,
  signal,
  onUpdate,
}: {
  accessToken: string;
  responseId: string;
  signal: AbortSignal;
  onUpdate: (update: AssistantMessageUpdate) => void;
}): Promise<void> {
  const client = backgroundClient(accessToken);
  while (!signal.aborted) {
    try {
      const response = await client.responses.retrieve(responseId, { stream: false }, { signal });
      if (signal.aborted) return;
      onUpdate(backgroundResponseUpdate(response));
      if (!isBackgroundRunPending({ id: response.id, status: response.status ?? "in_progress" })) return;
      await waitForPoll(signal);
    } catch (error) {
      if (signal.aborted) return;
      if (error instanceof OpenAI.APIError && error.status === 404) {
        onUpdate({
          backgroundRun: { id: responseId, status: "expired" },
          backgroundError: "This saved run is no longer available. Its result may have expired on the server.",
        });
        return;
      }
      throw error;
    }
  }
}

export async function cancelBackgroundChat(accessToken: string, responseId: string): Promise<AssistantMessageUpdate> {
  const response = await backgroundClient(accessToken).responses.cancel(responseId);
  return backgroundResponseUpdate(response);
}
