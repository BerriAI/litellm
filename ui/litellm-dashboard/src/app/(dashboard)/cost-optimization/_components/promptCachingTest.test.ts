import { describe, expect, it, vi } from "vitest";

import {
  PROMPT_CACHING_TEST_MIN_SYSTEM_TOKENS,
  PromptCachingCallResult,
  buildTestSystemPrompt,
  judgePromptCachingTest,
  runPromptCachingTest,
} from "./promptCachingTest";

const callResult = (overrides: Partial<PromptCachingCallResult> = {}): PromptCachingCallResult => ({
  cacheCreationTokens: 0,
  cacheReadTokens: 0,
  promptTokens: 5000,
  responseCost: null,
  model: null,
  durationMs: 10,
  ...overrides,
});

const jsonResponse = (body: unknown, init: { status?: number; headers?: Record<string, string> } = {}) =>
  new Response(JSON.stringify(body), {
    status: init.status ?? 200,
    headers: { "Content-Type": "application/json", ...init.headers },
  });

const anthropicUsage = {
  prompt_tokens: 5010,
  cache_creation_input_tokens: 5000,
  cache_read_input_tokens: 0,
};

describe("buildTestSystemPrompt", () => {
  it("is comfortably above Anthropic's largest minimum cacheable size and differs by nonce", () => {
    const first = buildTestSystemPrompt("run-a");
    const second = buildTestSystemPrompt("run-b");

    expect(first.length).toBeGreaterThan(20000);
    expect(first.length).toBeGreaterThan(PROMPT_CACHING_TEST_MIN_SYSTEM_TOKENS * 4);
    expect(first).not.toBe(second);
    expect(first.startsWith("Test run run-a.")).toBe(true);
    expect(first.slice(first.indexOf("\n"))).toBe(second.slice(second.indexOf("\n")));
  });
});

describe("judgePromptCachingTest", () => {
  it("returns injected when call 1 writes and call 2 reads", () => {
    expect(
      judgePromptCachingTest(callResult({ cacheCreationTokens: 5000 }), callResult({ cacheReadTokens: 5000 })),
    ).toBe("injected");
  });

  it("returns cache_hit_only when neither call writes but a call reads", () => {
    expect(judgePromptCachingTest(callResult(), callResult({ cacheReadTokens: 5000 }))).toBe("cache_hit_only");
    expect(judgePromptCachingTest(callResult({ cacheReadTokens: 5000 }), callResult({ cacheReadTokens: 5000 }))).toBe(
      "cache_hit_only",
    );
  });

  it("returns injected_no_read when call 1 writes but call 2 reads nothing", () => {
    expect(judgePromptCachingTest(callResult({ cacheCreationTokens: 5000 }), callResult())).toBe("injected_no_read");
  });

  it("returns not_injected with no cache activity at all", () => {
    expect(judgePromptCachingTest(callResult(), callResult())).toBe("not_injected");
  });
});

describe("runPromptCachingTest", () => {
  it("sends two sequential identical requests with no cache_control and parses usage", async () => {
    const seenBodies: unknown[] = [];
    const responses = [
      jsonResponse(
        { model: "claude-haiku-4-5", usage: anthropicUsage },
        { headers: { "x-litellm-response-cost": "0.0042" } },
      ),
      jsonResponse({
        model: "claude-haiku-4-5",
        usage: { prompt_tokens: 5010, prompt_tokens_details: { cached_tokens: 5000 } },
      }),
    ];
    const fetchImpl = vi.fn(async (_url: unknown, init?: RequestInit) => {
      seenBodies.push(JSON.parse(String(init?.body)));
      return responses[seenBodies.length - 1];
    });

    const testOptions = {
      accessToken: "token-123",
      model: "claude-haiku-4-5",
      baseUrl: "http://proxy:4000",
      fetchImpl,
      customHeaders: { "x-custom-auth": "abc", Authorization: "Bearer wrong" },
    };
    const result = await runPromptCachingTest(testOptions);

    expect(fetchImpl).toHaveBeenCalledTimes(2);
    const [url, init] = fetchImpl.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe("http://proxy:4000/v1/chat/completions");
    expect((init.headers as Record<string, string>).Authorization).toBe("Bearer token-123");
    expect((init.headers as Record<string, string>)["x-custom-auth"]).toBe("abc");
    expect((init.headers as Record<string, string>)["x-litellm-tags"]).toBe("prompt-caching-test");

    const serialized = JSON.stringify(seenBodies);
    expect(serialized).not.toContain("cache_control");
    expect((seenBodies[0] as { messages: { content: string }[] }).messages[0].content).toBe(
      (seenBodies[1] as { messages: { content: string }[] }).messages[0].content,
    );

    expect(result.first.cacheCreationTokens).toBe(5000);
    expect(result.first.responseCost).toBeCloseTo(0.0042);
    expect(result.second.cacheReadTokens).toBe(5000);
    expect(result.verdict).toBe("injected");
  });

  it("rejects with status and error message on non-2xx", async () => {
    const fetchImpl = vi.fn(async () => jsonResponse({ error: { message: "boom" } }, { status: 400 }));

    const testOptions = {
      accessToken: "token-123",
      model: "claude-haiku-4-5",
      baseUrl: "http://proxy:4000",
      fetchImpl,
    };
    await expect(runPromptCachingTest(testOptions)).rejects.toThrow(/400.*boom|boom.*400/);
  });
});
