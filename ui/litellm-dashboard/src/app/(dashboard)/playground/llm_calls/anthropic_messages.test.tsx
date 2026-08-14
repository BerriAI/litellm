import { afterEach, describe, expect, it, vi } from "vitest";
import { makeAnthropicMessagesRequest } from "./anthropic_messages";
import type { TokenUsage } from "@/components/chat_ui/ResponseMetrics";

vi.mock("@/components/networking", () => ({
  getProxyBaseUrl: vi.fn(() => "https://example.com"),
}));

const mockMessagesStream = vi.fn();

vi.mock("@anthropic-ai/sdk", () => ({
  default: vi.fn(() => ({ messages: { stream: mockMessagesStream } })),
}));

describe("anthropic_messages prompt cache usage", () => {
  const captureUsage = async (usage: Record<string, unknown>): Promise<TokenUsage> => {
    async function* mockStream() {
      yield {
        type: "message_delta",
        usage: { input_tokens: 5000, output_tokens: 2, ...usage },
      };
    }
    mockMessagesStream.mockReturnValue(mockStream());

    const onUsageData = vi.fn();
    await makeAnthropicMessagesRequest(
      [{ role: "user", content: "Hello" }],
      vi.fn(),
      "claude-haiku-4-5",
      "test-token",
      undefined,
      undefined,
      undefined,
      undefined,
      onUsageData,
    );

    expect(onUsageData).toHaveBeenCalledTimes(1);
    return onUsageData.mock.calls[0][0] as TokenUsage;
  };

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("surfaces read and creation tokens from Anthropic-shape usage", async () => {
    await expect(
      captureUsage({ cache_read_input_tokens: 4695, cache_creation_input_tokens: 1234 }),
    ).resolves.toMatchObject({ cacheReadTokens: 4695, cacheCreationTokens: 1234, promptTokens: 5000 });
  });

  it("omits cache fields entirely when Anthropic reports no prompt caching", async () => {
    const usageData = await captureUsage({});

    expect(usageData).not.toHaveProperty("cacheReadTokens");
    expect(usageData).not.toHaveProperty("cacheCreationTokens");
    expect(usageData.promptTokens).toBe(5000);
  });
});
