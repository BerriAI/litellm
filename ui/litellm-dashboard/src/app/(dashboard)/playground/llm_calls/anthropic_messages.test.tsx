import { afterEach, describe, expect, it, vi } from "vitest";
import { makeAnthropicMessagesRequest } from "./anthropic_messages";
import type { TokenUsage } from "@/components/chat_ui/ResponseMetrics";

vi.mock("@/components/networking", () => ({
  getProxyBaseUrl: vi.fn(() => "https://example.com"),
}));

const mockMessagesStream = vi.fn();
const mockMessagesCreate = vi.fn();

vi.mock("@anthropic-ai/sdk", () => ({
  default: vi.fn(function () {
    return { messages: { stream: mockMessagesStream, create: mockMessagesCreate } };
  }),
}));

const NON_STREAMING_ARGS = [
  undefined, // traceId
  undefined, // vector_store_ids
  undefined, // guardrails
  undefined, // policies
  undefined, // selectedMCPServers
  undefined, // customBaseUrl
  undefined, // mcpServers
  undefined, // mcpServerToolRestrictions
  undefined, // mcpToolsets
  false, // streamingEnabled
] as const;

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

describe("anthropic_messages non-streaming", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("sends stream:false through messages.create and renders the full reply at once", async () => {
    mockMessagesCreate.mockResolvedValue({
      content: [
        { type: "thinking", thinking: "considering" },
        { type: "text", text: "OK" },
      ],
      usage: { input_tokens: 12, output_tokens: 3, cache_read_input_tokens: 7 },
    });
    const updateTextUI = vi.fn();
    const onReasoningContent = vi.fn();
    const onUsageData = vi.fn();

    await makeAnthropicMessagesRequest(
      [{ role: "user", content: "Hello" }],
      updateTextUI,
      "claude-haiku-4-5",
      "test-token",
      undefined,
      undefined,
      onReasoningContent,
      undefined,
      onUsageData,
      ...NON_STREAMING_ARGS,
    );

    expect(mockMessagesStream).not.toHaveBeenCalled();
    expect(mockMessagesCreate).toHaveBeenCalledTimes(1);
    expect(mockMessagesCreate.mock.calls[0][0]).toMatchObject({ model: "claude-haiku-4-5", stream: false });
    expect(updateTextUI).toHaveBeenCalledWith("assistant", "OK", "claude-haiku-4-5");
    expect(onReasoningContent).toHaveBeenCalledWith("considering");
    const expectedUsage: TokenUsage = { completionTokens: 3, promptTokens: 12, totalTokens: 15, cacheReadTokens: 7 };
    expect(onUsageData).toHaveBeenCalledWith(expectedUsage);
  });

  it("keeps streaming as the default when the flag is omitted", async () => {
    async function* emptyStream() {}
    mockMessagesStream.mockReturnValue(emptyStream());

    await makeAnthropicMessagesRequest([{ role: "user", content: "Hello" }], vi.fn(), "claude-haiku-4-5", "test-token");

    expect(mockMessagesCreate).not.toHaveBeenCalled();
    expect(mockMessagesStream.mock.calls[0][0]).toMatchObject({ stream: true });
  });
});
