import { describe, it, vi, expect, beforeEach } from "vitest";
import { makeOpenAIResponsesRequest } from "./responses_api";

// Mock the OpenAI client
vi.mock("openai", () => {
  return {
    default: {
      OpenAI: vi.fn(),
    },
  };
});

// Mock the getProxyBaseUrl function
vi.mock("@/components/networking", () => ({
  getProxyBaseUrl: vi.fn(() => "http://localhost:4000"),
}));

// Mock NotificationManager
vi.mock("@/components/molecules/notifications_manager", () => ({
  default: {
    fromBackend: vi.fn(),
  },
}));

describe("response_api", () => {
  let mockUpdateTextUI: ReturnType<typeof vi.fn>;
  let mockOnTimingData: ReturnType<typeof vi.fn>;
  let mockOnUsageData: ReturnType<typeof vi.fn>;
  let mockOnResponseId: ReturnType<typeof vi.fn>;
  let mockOnMCPEvent: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    vi.clearAllMocks();
    mockUpdateTextUI = vi.fn();
    mockOnTimingData = vi.fn();
    mockOnUsageData = vi.fn();
    mockOnResponseId = vi.fn();
    mockOnMCPEvent = vi.fn();
  });

  it("should call updateTextUI with text deltas from stream", async () => {
    const openai = await import("openai");
    const mockStream = [
      {
        type: "response.output_text.delta",
        delta: "Hello",
      },
      {
        type: "response.output_text.delta",
        delta: " world",
      },
    ];

    const mockCreate = vi.fn().mockResolvedValue(mockStream);
    (openai.default.OpenAI as any).mockImplementation(() => ({
      responses: {
        create: mockCreate,
      },
    }));

    await makeOpenAIResponsesRequest([{ role: "user", content: "test" }], mockUpdateTextUI, "gpt-4", "test-token");

    expect(mockUpdateTextUI).toHaveBeenCalledWith("assistant", "Hello", "gpt-4");
    expect(mockUpdateTextUI).toHaveBeenCalledWith("assistant", " world", "gpt-4");
  });

  it("should measure time to first token", async () => {
    const openai = await import("openai");
    const mockStream = [
      {
        type: "response.output_text.delta",
        delta: "First token",
      },
    ];

    const mockCreate = vi.fn().mockResolvedValue(mockStream);
    (openai.default.OpenAI as any).mockImplementation(() => ({
      responses: {
        create: mockCreate,
      },
    }));

    await makeOpenAIResponsesRequest(
      [{ role: "user", content: "test" }],
      mockUpdateTextUI,
      "gpt-4",
      "test-token",
      [],
      undefined,
      undefined,
      mockOnTimingData,
    );

    expect(mockOnTimingData).toHaveBeenCalled();
    expect(mockOnTimingData).toHaveBeenCalledWith(expect.any(Number), "gpt-4");
  });

  it("should include tags in headers when provided", async () => {
    const openai = await import("openai");
    const mockStream = [
      {
        type: "response.output_text.delta",
        delta: "test",
      },
    ];

    const mockCreate = vi.fn().mockResolvedValue(mockStream);
    (openai.default.OpenAI as any).mockImplementation(() => ({
      responses: {
        create: mockCreate,
      },
    }));

    await makeOpenAIResponsesRequest([{ role: "user", content: "test" }], mockUpdateTextUI, "gpt-4", "test-token", [
      "tag1",
      "tag2",
    ]);

    // Verify OpenAI client was constructed with tags in headers
    expect(openai.default.OpenAI).toHaveBeenCalledWith(
      expect.objectContaining({
        defaultHeaders: expect.objectContaining({
          "x-litellm-tags": "tag1,tag2",
        }),
      }),
    );
  });

  it("should handle response ID from completed event", async () => {
    const openai = await import("openai");
    const mockStream = [
      {
        type: "response.completed",
        response: {
          id: "resp_123",
          usage: {
            input_tokens: 10,
            output_tokens: 20,
            total_tokens: 30,
          },
        },
      },
    ];

    const mockCreate = vi.fn().mockResolvedValue(mockStream);
    (openai.default.OpenAI as any).mockImplementation(() => ({
      responses: {
        create: mockCreate,
      },
    }));

    await makeOpenAIResponsesRequest(
      [{ role: "user", content: "test" }],
      mockUpdateTextUI,
      "gpt-4",
      "test-token",
      [],
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      mockOnResponseId,
    );

    expect(mockOnResponseId).toHaveBeenCalledWith("resp_123");
  });

  it("should handle usage data from completed event", async () => {
    const openai = await import("openai");
    const mockStream = [
      {
        type: "response.completed",
        response: {
          id: "resp_123",
          usage: {
            input_tokens: 10,
            output_tokens: 20,
            total_tokens: 30,
          },
        },
      },
    ];

    const mockCreate = vi.fn().mockResolvedValue(mockStream);
    (openai.default.OpenAI as any).mockImplementation(() => ({
      responses: {
        create: mockCreate,
      },
    }));

    await makeOpenAIResponsesRequest(
      [{ role: "user", content: "test" }],
      mockUpdateTextUI,
      "gpt-4",
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
      "",
      "gpt-4",
    );
  });

  it("should handle MCP events", async () => {
    const openai = await import("openai");
    const mockStream = [
      {
        type: "response.mcp_call.started",
        sequence_number: 1,
        output_index: 0,
        item_id: "item_123",
      },
    ];

    const mockCreate = vi.fn().mockResolvedValue(mockStream);
    (openai.default.OpenAI as any).mockImplementation(() => ({
      responses: {
        create: mockCreate,
      },
    }));

    await makeOpenAIResponsesRequest(
      [{ role: "user", content: "test" }],
      mockUpdateTextUI,
      "gpt-4",
      "test-token",
      [],
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      mockOnMCPEvent,
    );

    expect(mockOnMCPEvent).toHaveBeenCalledWith(
      expect.objectContaining({
        type: "response.mcp_call.started",
        sequence_number: 1,
        output_index: 0,
        item_id: "item_123",
      }),
    );
  });

  it("should handle aborted requests", async () => {
    const openai = await import("openai");
    const abortController = new AbortController();

    const mockCreate = vi.fn().mockImplementation(() => {
      abortController.abort();
      throw new Error("Request aborted");
    });

    (openai.default.OpenAI as any).mockImplementation(() => ({
      responses: {
        create: mockCreate,
      },
    }));

    await expect(
      makeOpenAIResponsesRequest(
        [{ role: "user", content: "test" }],
        mockUpdateTextUI,
        "gpt-4",
        "test-token",
        [],
        abortController.signal,
      ),
    ).rejects.toThrow();
  });

  it("should throw error when access token is missing", async () => {
    await expect(
      makeOpenAIResponsesRequest([{ role: "user", content: "test" }], mockUpdateTextUI, "gpt-4", null),
    ).rejects.toThrow("API key is required");
  });

  it("should skip whitespace-only deltas", async () => {
    const openai = await import("openai");
    const mockStream = [
      {
        type: "response.output_text.delta",
        delta: "   ",
      },
      {
        type: "response.output_text.delta",
        delta: "actual content",
      },
    ];

    const mockCreate = vi.fn().mockResolvedValue(mockStream);
    (openai.default.OpenAI as any).mockImplementation(() => ({
      responses: {
        create: mockCreate,
      },
    }));

    await makeOpenAIResponsesRequest([{ role: "user", content: "test" }], mockUpdateTextUI, "gpt-4", "test-token");

    // Should only be called once for "actual content", not for whitespace
    expect(mockUpdateTextUI).toHaveBeenCalledTimes(1);
    expect(mockUpdateTextUI).toHaveBeenCalledWith("assistant", "actual content", "gpt-4");
  });
});
