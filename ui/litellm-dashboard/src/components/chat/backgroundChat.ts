import OpenAI from "openai";
import type { Response } from "openai/resources/responses/responses";
import { getGatewayBaseUrl } from "@/components/networking";
import { parseUsageCost } from "@/components/llm_calls/usage_cost";
import { extractPromptCacheTokens } from "@/utils/promptCacheUsage";
import type { AssistantMessageUpdate, ChatBackgroundRun, ChatCitation, MCPEvent } from "./types";

export const isBackgroundRunPending = (run?: ChatBackgroundRun): boolean =>
  run?.status === "queued" || run?.status === "in_progress";

type TimedResponse = Response & {
  litellm_response_timing?: {
    time_to_first_token_ms?: number;
    total_latency_ms?: number;
  };
};

export function backgroundResponseUpdate(response: TimedResponse): AssistantMessageUpdate {
  const output = response.output ?? [];
  const citations = Array.from(
    new Map(
      output
        .filter((item) => item.type === "message")
        .flatMap((item) => item.content ?? [])
        .flatMap((part) => (part.type === "output_text" ? part.annotations ?? [] : []))
        .filter((annotation) => annotation.type === "url_citation")
        .filter((citation) => /^https?:\/\//i.test(citation.url))
        .map((citation): [string, ChatCitation] => [
          citation.url,
          { url: citation.url, title: citation.title || citation.url },
        ]),
    ).values(),
  );
  const webSearchCount = output.filter(
    (item) => item.type === "web_search_call" && item.action?.type === "search",
  ).length;
  const content = output
    .filter((item) => item.type === "message")
    .flatMap((item) => item.content ?? [])
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
            tools: (item.tools ?? []).map((tool) => ({
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
  const cost = response.usage && "cost" in response.usage ? parseUsageCost(response.usage.cost) : undefined;
  const timing = response.litellm_response_timing;
  return {
    content,
    citations,
    webSearchCount,
    reasoningContent,
    mcpEvents,
    backgroundRun: { id: response.id, status },
    backgroundError,
    ...(timing?.time_to_first_token_ms !== undefined && { timeToFirstToken: timing.time_to_first_token_ms }),
    ...(timing?.total_latency_ms !== undefined && { totalLatency: timing.total_latency_ms }),
    ...(response.usage && {
      usage: {
        promptTokens: response.usage.input_tokens,
        completionTokens: response.usage.output_tokens,
        totalTokens: response.usage.total_tokens,
        reasoningTokens: response.usage.output_tokens_details?.reasoning_tokens,
        ...extractPromptCacheTokens(response.usage),
        ...(cost !== undefined ? { cost } : {}),
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
