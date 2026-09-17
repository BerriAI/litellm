import { z } from "zod";

import { buildPlaygroundHeaders } from "@/components/llm_calls/request_headers";
import { extractPromptCacheTokens } from "@/utils/promptCacheUsage";

export const PROMPT_CACHING_TEST_MIN_SYSTEM_TOKENS = 4096;

const FILLER_LINE_COUNT = 600;

export const buildTestSystemPrompt = (nonce: string): string => {
  const filler = Array.from(
    { length: FILLER_LINE_COUNT },
    (_, index) => `Reference item ${index + 1}: the gateway forwards requests to the configured deployment.`,
  ).join("\n");
  return `Test run ${nonce}.\n${filler}`;
};

export interface PromptCachingTestRequestBody {
  readonly model: string;
  readonly messages: readonly { readonly role: string; readonly content: string }[];
  readonly max_tokens: number;
  readonly temperature: number;
}

export const buildTestRequestBody = (model: string, systemPrompt: string): PromptCachingTestRequestBody => ({
  model,
  messages: [
    { role: "system", content: systemPrompt },
    { role: "user", content: "Reply with the single word OK." },
  ],
  max_tokens: 5,
  temperature: 0,
});

export interface PromptCachingCallResult {
  cacheCreationTokens: number;
  cacheReadTokens: number;
  promptTokens: number;
  responseCost: number | null;
  model: string | null;
  durationMs: number;
}

export type PromptCachingVerdict = "injected" | "not_injected" | "cache_hit_only";

export const judgePromptCachingTest = (
  first: PromptCachingCallResult,
  second: PromptCachingCallResult,
): PromptCachingVerdict => {
  if (first.cacheCreationTokens > 0 && second.cacheReadTokens > 0) {
    return "injected";
  }
  if (first.cacheCreationTokens === 0 && second.cacheCreationTokens === 0) {
    if (first.cacheReadTokens > 0 || second.cacheReadTokens > 0) {
      return "cache_hit_only";
    }
  }
  return "not_injected";
};

const tokenDetailsSchema = z
  .object({
    cached_tokens: z.number().nullish(),
    cache_write_tokens: z.number().nullish(),
  })
  .nullish();

const usageSchema = z
  .object({
    prompt_tokens: z.number().nullish(),
    cache_creation_input_tokens: z.number().nullish(),
    cache_read_input_tokens: z.number().nullish(),
    prompt_tokens_details: tokenDetailsSchema,
    input_tokens_details: tokenDetailsSchema,
  })
  .nullish();

const chatCompletionResponseSchema = z.object({
  model: z.string().nullish(),
  usage: usageSchema,
});

const errorBodySchema = z.object({
  error: z.object({ message: z.string().optional() }).optional(),
});

const parseResponseCost = (headers: Headers): number | null => {
  const raw = headers.get("x-litellm-response-cost");
  if (raw === null) {
    return null;
  }
  const cost = Number.parseFloat(raw);
  return Number.isFinite(cost) ? cost : null;
};

const sendTestCall = async (
  accessToken: string,
  baseUrl: string,
  body: PromptCachingTestRequestBody,
  fetchImpl: typeof fetch,
): Promise<PromptCachingCallResult> => {
  const startedAt = Date.now();
  const response = await fetchImpl(`${baseUrl}/v1/chat/completions`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
      ...buildPlaygroundHeaders(["prompt-caching-test"]),
    },
    body: JSON.stringify(body),
  });
  const durationMs = Date.now() - startedAt;

  const json: unknown = await response.json();
  if (!response.ok) {
    const parsedError = errorBodySchema.safeParse(json);
    const detail = parsedError.success ? parsedError.data.error?.message : undefined;
    throw new Error(`Request failed with status ${response.status}${detail ? `: ${detail}` : ""}`);
  }

  const parsed = chatCompletionResponseSchema.parse(json);
  const cacheTokens = extractPromptCacheTokens(parsed.usage);

  return {
    cacheCreationTokens: cacheTokens.cacheCreationTokens ?? 0,
    cacheReadTokens: cacheTokens.cacheReadTokens ?? 0,
    promptTokens: parsed.usage?.prompt_tokens ?? 0,
    responseCost: parseResponseCost(response.headers),
    model: parsed.model ?? null,
    durationMs,
  };
};

export interface RunPromptCachingTestOptions {
  readonly accessToken: string;
  readonly model: string;
  readonly baseUrl: string;
  readonly fetchImpl?: typeof fetch;
  readonly onCallStart?: (callIndex: 1 | 2) => void;
}

export const runPromptCachingTest = async ({
  accessToken,
  model,
  baseUrl,
  fetchImpl = fetch,
  onCallStart,
}: RunPromptCachingTestOptions): Promise<{
  first: PromptCachingCallResult;
  second: PromptCachingCallResult;
  verdict: PromptCachingVerdict;
}> => {
  const systemPrompt = buildTestSystemPrompt(`${Date.now()}-${Math.random().toString(36).slice(2)}`);
  const body = buildTestRequestBody(model, systemPrompt);

  onCallStart?.(1);
  const first = await sendTestCall(accessToken, baseUrl, body, fetchImpl);
  onCallStart?.(2);
  const second = await sendTestCall(accessToken, baseUrl, body, fetchImpl);

  return { first, second, verdict: judgePromptCachingTest(first, second) };
};
