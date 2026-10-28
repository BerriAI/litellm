export const PROMPT_CACHE_READ_TOOLTIP =
  "Input tokens read from the LLM provider's prompt cache (e.g. Anthropic / OpenAI), billed at a discounted rate. Reported by the provider.";
export const PROMPT_CACHE_CREATION_TOOLTIP =
  "Input tokens written to the LLM provider's prompt cache for reuse by later requests.";

interface CachedTokenDetails {
  cached_tokens?: number | null;
  cache_creation_tokens?: number | null;
  cache_write_tokens?: number | null;
}

export interface ProviderCacheUsage {
  cache_read_input_tokens?: number | null;
  cache_creation_input_tokens?: number | null;
  prompt_tokens_details?: CachedTokenDetails | null;
  input_tokens_details?: CachedTokenDetails | null;
}

export interface PromptCacheTokens {
  cacheReadTokens?: number;
  cacheCreationTokens?: number;
}

const tokenCount = (value: number | null | undefined, includeZero: boolean): number | undefined =>
  typeof value === "number" && Number.isFinite(value) && (includeZero ? value >= 0 : value > 0) ? value : undefined;

export const extractPromptCacheTokens = (
  usage: ProviderCacheUsage | null | undefined,
  includeZero = false,
): PromptCacheTokens => {
  const details = usage?.prompt_tokens_details ?? usage?.input_tokens_details;
  const cacheReadTokens =
    tokenCount(usage?.cache_read_input_tokens, includeZero) ?? tokenCount(details?.cached_tokens, includeZero);
  const cacheCreationTokens =
    tokenCount(usage?.cache_creation_input_tokens, includeZero) ??
    tokenCount(details?.cache_creation_tokens, includeZero) ??
    tokenCount(details?.cache_write_tokens, includeZero);

  return {
    ...(cacheReadTokens !== undefined && { cacheReadTokens }),
    ...(cacheCreationTokens !== undefined && { cacheCreationTokens }),
  };
};
