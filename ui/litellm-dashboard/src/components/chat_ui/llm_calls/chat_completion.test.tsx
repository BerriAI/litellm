import { describe, it, vi, expect, beforeEach } from "vitest";
import { makeOpenAIChatCompletionRequest } from "./chat_completion";

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

describe("chat_completion", () => {
  let mockUpdateUI: ReturnType<typeof vi.fn>;
  let mockOnTimingData: ReturnType<typeof vi.fn>;
  let mockOnUsageData: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    vi.clearAllMocks();
    mockUpdateUI = vi.fn();
    mockOnTimingData = vi.fn();
    mockOnUsageData = vi.fn();
  });

  it("should call updateUI with content from stream chunks", async () => {
    const openai = await import("openai");
    const mockStream = [
      {
        choices: [{ delta: { content: "Hello" } }],
      },
      {
        choices: [{ delta: { content: " world" } }],
      },
    ];

    const mockCreate = vi.fn().mockResolvedValue(mockStream);
    (openai.default.OpenAI as any).mockImplementation(() => ({
      chat: {
        completions: {
          create: mockCreate,
        },
      },
    }));

    await makeOpenAIChatCompletionRequest([{ role: "user", content: "test" }], mockUpdateUI, "gpt-4", "test-token");

    expect(mockUpdateUI).toHaveBeenCalledWith("Hello", "gpt-4");
    expect(mockUpdateUI).toHaveBeenCalledWith(" world", "gpt-4");
  });

  it("should measure time to first token", async () => {
    const openai = await import("openai");
    const mockStream = [
      {
        choices: [{ delta: { content: "First token" } }],
      },
    ];

    const mockCreate = vi.fn().mockResolvedValue(mockStream);
    (openai.default.OpenAI as any).mockImplementation(() => ({
      chat: {
        completions: {
          create: mockCreate,
        },
      },
    }));

    await makeOpenAIChatCompletionRequest(
      [{ role: "user", content: "test" }],
      mockUpdateUI,
      "gpt-4",
      "test-token",
      undefined,
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
        choices: [{ delta: { content: "test" } }],
      },
    ];

    const mockCreate = vi.fn().mockResolvedValue(mockStream);
    (openai.default.OpenAI as any).mockImplementation(() => ({
      chat: {
        completions: {
          create: mockCreate,
        },
      },
    }));

    await makeOpenAIChatCompletionRequest([{ role: "user", content: "test" }], mockUpdateUI, "gpt-4", "test-token", [
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

  it("should handle aborted requests", async () => {
    const openai = await import("openai");
    const abortController = new AbortController();

    const mockCreate = vi.fn().mockImplementation(() => {
      abortController.abort();
      throw new Error("Request aborted");
    });

    (openai.default.OpenAI as any).mockImplementation(() => ({
      chat: {
        completions: {
          create: mockCreate,
        },
      },
    }));

    await expect(
      makeOpenAIChatCompletionRequest(
        [{ role: "user", content: "test" }],
        mockUpdateUI,
        "gpt-4",
        "test-token",
        undefined,
        abortController.signal,
      ),
    ).rejects.toThrow();
  });
});
