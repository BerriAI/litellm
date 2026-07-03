/**
 * Pure helpers for the request log detail cache/cost display (issue #32045).
 *
 * Two distinct caches must never be conflated:
 *  - LiteLLM response cache: LiteLLM serves the whole response from its own cache,
 *    no provider call is made, and cost is $0. Tracked by `cache_hit`.
 *  - Provider prompt cache: the provider (OpenAI, Anthropic, ...) reuses cached
 *    prompt tokens and bills them at the discounted `cache_read_input_token_cost`.
 *    The request is still billed. Tracked by cached-token counts.
 */

export interface ProviderCacheTokens {
  /** Total prompt tokens billed for this request. */
  promptTokens: number;
  /** Prompt tokens served from the provider prompt cache (billed at the read rate). */
  cacheHitTokens: number;
  /** Prompt tokens not in the provider cache (billed at the full input rate). */
  cacheMissTokens: number;
}

const toFiniteNumber = (value: unknown): number | undefined => {
  if (value === undefined || value === null) return undefined;
  const n = Number(value);
  return Number.isFinite(n) ? n : undefined;
};

/**
 * Provider prompt-cache hit tokens, checked in the order the issue specifies:
 * OpenAI-style `prompt_tokens_details.cached_tokens` first, then Anthropic-style
 * `cache_read_input_tokens`. Returns 0 when neither is present.
 */
export function getProviderCacheReadTokens(metadata: Record<string, unknown> | undefined | null): number {
  if (!metadata) return 0;
  const usageObject = metadata.usage_object as { prompt_tokens_details?: { cached_tokens?: unknown } } | undefined;
  const openAiCached = toFiniteNumber(usageObject?.prompt_tokens_details?.cached_tokens);
  if (openAiCached !== undefined && openAiCached > 0) return openAiCached;

  const additional = metadata.additional_usage_values as { cache_read_input_tokens?: unknown } | undefined;
  const anthropicCached = toFiniteNumber(additional?.cache_read_input_tokens);
  if (anthropicCached !== undefined && anthropicCached > 0) return anthropicCached;

  return 0;
}

/** True when the provider reported any prompt-cache hit tokens. */
export function hasProviderPromptCacheHit(metadata: Record<string, unknown> | undefined | null): boolean {
  return getProviderCacheReadTokens(metadata) > 0;
}

/**
 * Splits prompt tokens into provider cache-hit vs cache-miss. `cacheHitTokens`
 * is clamped to `[0, promptTokens]` so a malformed payload can never produce a
 * negative miss count or a hit count larger than the prompt.
 */
export function splitPromptTokens(promptTokens: number, cacheReadTokens: number): ProviderCacheTokens {
  const safePrompt = Number.isFinite(promptTokens) && promptTokens > 0 ? promptTokens : 0;
  const safeHit = Number.isFinite(cacheReadTokens) && cacheReadTokens > 0 ? Math.min(cacheReadTokens, safePrompt) : 0;
  return {
    promptTokens: safePrompt,
    cacheHitTokens: safeHit,
    cacheMissTokens: safePrompt - safeHit,
  };
}

/**
 * Derives a per-token rate from a cost and its token count. Returns undefined
 * when the token count is 0 (division would be undefined) so callers can hide
 * the formula rather than print `Infinity` or `NaN`.
 */
export function deriveRate(cost: number | undefined, tokens: number | undefined): number | undefined {
  if (cost === undefined || cost === null || cost < 0) return undefined;
  if (tokens === undefined || tokens === null || tokens <= 0) return undefined;
  const rate = cost / tokens;
  return Number.isFinite(rate) ? rate : undefined;
}
