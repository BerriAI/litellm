import { describe, expect, it } from "vitest";
import { extractPromptCacheTokens } from "./promptCacheUsage";

describe("extractPromptCacheTokens", () => {
  it("reads the Anthropic Messages shape", () => {
    expect(
      extractPromptCacheTokens({ cache_read_input_tokens: 5678, cache_creation_input_tokens: 1234 }),
    ).toStrictEqual({ cacheReadTokens: 5678, cacheCreationTokens: 1234 });
  });

  it("reads the chat completions shape", () => {
    expect(
      extractPromptCacheTokens({ prompt_tokens_details: { cached_tokens: 4695, cache_write_tokens: 0 } }),
    ).toStrictEqual({ cacheReadTokens: 4695 });
  });

  it("reads the Responses API shape", () => {
    expect(
      extractPromptCacheTokens({ input_tokens_details: { cached_tokens: 0, cache_write_tokens: 4695 } }),
    ).toStrictEqual({ cacheCreationTokens: 4695 });
  });

  it("returns nothing for usage without cache fields", () => {
    expect(extractPromptCacheTokens({})).toStrictEqual({});
    expect(extractPromptCacheTokens(undefined)).toStrictEqual({});
    expect(extractPromptCacheTokens(null)).toStrictEqual({});
  });

  it("drops zero and non-finite counts so non-caching providers render nothing", () => {
    expect(
      extractPromptCacheTokens({
        cache_read_input_tokens: 0,
        cache_creation_input_tokens: null,
        prompt_tokens_details: { cached_tokens: 0, cache_write_tokens: 0 },
      }),
    ).toStrictEqual({});
    expect(extractPromptCacheTokens({ cache_read_input_tokens: Number.NaN })).toStrictEqual({});
  });

  it("should preserve explicitly recorded zero counts when totals require them", () => {
    const usage = {
      cache_read_input_tokens: 0,
      prompt_tokens_details: { cached_tokens: 100, cache_creation_tokens: 0 },
    };
    expect(extractPromptCacheTokens(usage, true)).toStrictEqual({ cacheReadTokens: 0, cacheCreationTokens: 0 });
    expect(extractPromptCacheTokens(usage)).toStrictEqual({ cacheReadTokens: 100 });
  });

  it("should preserve Responses API cache counts with zero-inclusive extraction", () => {
    const usage = { input_tokens_details: { cached_tokens: 0, cache_write_tokens: 120 } };
    expect(extractPromptCacheTokens(usage, true)).toStrictEqual({ cacheReadTokens: 0, cacheCreationTokens: 120 });
  });

  it("should keep missing and invalid counts unavailable when including zero", () => {
    expect(extractPromptCacheTokens(undefined, true)).toStrictEqual({});
    const invalid = { cache_read_input_tokens: -1, cache_creation_input_tokens: Number.NaN };
    expect(extractPromptCacheTokens(invalid, true)).toStrictEqual({});
  });
});
