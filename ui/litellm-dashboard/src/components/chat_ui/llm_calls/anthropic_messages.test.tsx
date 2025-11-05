import { describe, it, vi, expect, beforeEach } from "vitest";
import { makeAnthropicMessagesRequest } from "./anthropic_messages";

// Mock the Anthropic client
vi.mock("@anthropic-ai/sdk", () => {
  return {
    default: vi.fn(),
  };
});

// Mock the getProxyBaseUrl function
vi.mock("@/components/networking", () => ({
  getProxyBaseUrl: vi.fn(() => "http://localhost:4000"),
}));

// Mock the NotificationManager
vi.mock("@/components/molecules/notifications_manager", () => ({
  default: {
    fromBackend: vi.fn(),
  },
}));

describe("anthropic_messages", () => {
  let mockUpdateTextUI: ReturnType<typeof vi.fn>;
  let mockOnTimingData: ReturnType<typeof vi.fn>;
  let mockOnUsageData: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    vi.clearAllMocks();
    mockUpdateTextUI = vi.fn();
    mockOnTimingData = vi.fn();
    mockOnUsageData = vi.fn();
  });

  it("should throw error when access token is missing", async () => {
    await expect(
      makeAnthropicMessagesRequest(
        [{ role: "user", content: "test" }],
        mockUpdateTextUI,
        "claude-3-opus-20240229",
        null,
      ),
    ).rejects.toThrow("API key is required");
  });

  it("should call updateTextUI with content from stream chunks", async () => {
    const Anthropic = await import("@anthropic-ai/sdk");

    const mockStream = [
      {
        type: "content_block_delta",
        delta: { type: "text_delta", text: "Hello" },
      },
      {
        type: "content_block_delta",
        delta: { type: "text_delta", text: " world" },
      },
    ];

    const mockStreamMethod = vi.fn().mockReturnValue(mockStream);
    (Anthropic.default as any).mockImplementation(() => ({
      messages: {
        stream: mockStreamMethod,
      },
    }));

    await makeAnthropicMessagesRequest(
      [{ role: "user", content: "test" }],
      mockUpdateTextUI,
      "claude-3-opus-20240229",
      "test-token",
    );

    expect(mockUpdateTextUI).toHaveBeenCalledWith("assistant", "Hello", "claude-3-opus-20240229");
    expect(mockUpdateTextUI).toHaveBeenCalledWith("assistant", " world", "claude-3-opus-20240229");
  });

  it("should measure time to first token", async () => {
    const Anthropic = await import("@anthropic-ai/sdk");

    const mockStream = [
      {
        type: "content_block_delta",
        delta: { type: "text_delta", text: "First token" },
      },
    ];

    const mockStreamMethod = vi.fn().mockReturnValue(mockStream);
    (Anthropic.default as any).mockImplementation(() => ({
      messages: {
        stream: mockStreamMethod,
      },
    }));

    await makeAnthropicMessagesRequest(
      [{ role: "user", content: "test" }],
      mockUpdateTextUI,
      "claude-3-opus-20240229",
      "test-token",
      [],
      undefined,
      undefined,
      mockOnTimingData,
    );

    expect(mockOnTimingData).toHaveBeenCalled();
    expect(mockOnTimingData).toHaveBeenCalledWith(expect.any(Number), "claude-3-opus-20240229");
  });

  it("should include tags in headers when provided", async () => {
    const Anthropic = await import("@anthropic-ai/sdk");

    const mockStream = [
      {
        type: "content_block_delta",
        delta: { type: "text_delta", text: "test" },
      },
    ];

    const mockStreamMethod = vi.fn().mockReturnValue(mockStream);
    (Anthropic.default as any).mockImplementation(() => ({
      messages: {
        stream: mockStreamMethod,
      },
    }));

    await makeAnthropicMessagesRequest(
      [{ role: "user", content: "test" }],
      mockUpdateTextUI,
      "claude-3-opus-20240229",
      "test-token",
      ["tag1", "tag2"],
    );

    // Verify Anthropic client was constructed with tags in headers
    expect(Anthropic.default).toHaveBeenCalledWith(
      expect.objectContaining({
        defaultHeaders: expect.objectContaining({
          "x-litellm-tags": "tag1,tag2",
        }),
      }),
    );
  });

  it("should process usage data from message_delta events", async () => {
    const Anthropic = await import("@anthropic-ai/sdk");

    const mockStream = [
      {
        type: "content_block_delta",
        delta: { type: "text_delta", text: "test" },
      },
      {
        type: "message_delta",
        usage: {
          input_tokens: 10,
          output_tokens: 20,
        },
      },
    ];

    const mockStreamMethod = vi.fn().mockReturnValue(mockStream);
    (Anthropic.default as any).mockImplementation(() => ({
      messages: {
        stream: mockStreamMethod,
      },
    }));

    await makeAnthropicMessagesRequest(
      [{ role: "user", content: "test" }],
      mockUpdateTextUI,
      "claude-3-opus-20240229",
      "test-token",
      [],
      undefined,
      undefined,
      undefined,
      mockOnUsageData,
    );

    expect(mockOnUsageData).toHaveBeenCalledWith(
      {
        completionTokens: 20,
        promptTokens: 10,
        totalTokens: 30,
      },
      undefined,
      "claude-3-opus-20240229",
    );
  });

  it("should handle aborted requests", async () => {
    const Anthropic = await import("@anthropic-ai/sdk");
    const abortController = new AbortController();

    const mockStreamMethod = vi.fn().mockImplementation(() => {
      abortController.abort();
      throw new Error("Request aborted");
    });

    (Anthropic.default as any).mockImplementation(() => ({
      messages: {
        stream: mockStreamMethod,
      },
    }));

    await expect(
      makeAnthropicMessagesRequest(
        [{ role: "user", content: "test" }],
        mockUpdateTextUI,
        "claude-3-opus-20240229",
        "test-token",
        [],
        abortController.signal,
      ),
    ).rejects.toThrow();
  });
});
