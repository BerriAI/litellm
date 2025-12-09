import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { makeOpenAIChatCompletionRequest } from "./chat_completion";

vi.mock("@/components/networking", () => ({
  getProxyBaseUrl: vi.fn(() => "https://example.com"),
}));

// Mock the OpenAI client
const mockCreate = vi.fn();
const mockChatCompletions = {
  create: mockCreate,
};
const mockChat = {
  completions: mockChatCompletions,
};
const mockClient = {
  chat: mockChat,
};

vi.mock("openai", () => ({
  default: {
    OpenAI: vi.fn(() => mockClient),
  },
}));

describe("chat_completion", () => {
  const mockUpdateUI = vi.fn();
  const mockChatHistory = [{ role: "user", content: "Hello" }];

  beforeEach(() => {
    // Create a mock async iterator for streaming response
    const mockChunks = [
      {
        choices: [{ delta: { content: "Hello" }, index: 0 }],
        model: "gpt-4",
      },
      {
        choices: [{ delta: { content: " there" }, index: 0 }],
        model: "gpt-4",
      },
      {
        choices: [{ delta: {}, index: 0 }],
        model: "gpt-4",
        usage: {
          completion_tokens: 2,
          prompt_tokens: 5,
          total_tokens: 7,
        },
      },
    ];

    async function* mockStream() {
      for (const chunk of mockChunks) {
        yield chunk;
      }
    }

    mockCreate.mockResolvedValue(mockStream());
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("should make a basic chat completion request", async () => {
    await makeOpenAIChatCompletionRequest(mockChatHistory, mockUpdateUI, "gpt-4", "test-token");

    expect(mockCreate).toHaveBeenCalledTimes(1);
    expect(mockCreate).toHaveBeenCalledWith(
      {
        model: "gpt-4",
        stream: true,
        stream_options: {
          include_usage: true,
        },
        messages: mockChatHistory,
      },
      { signal: undefined },
    );
    expect(mockUpdateUI).toHaveBeenCalledWith("Hello", "gpt-4");
    expect(mockUpdateUI).toHaveBeenCalledWith(" there", "gpt-4");
  });

  it("should include temperature and max_tokens when provided", async () => {
    await makeOpenAIChatCompletionRequest(
      mockChatHistory,
      mockUpdateUI,
      "gpt-4",
      "test-token",
      undefined, // tags
      undefined, // signal
      undefined, // onReasoningContent
      undefined, // onTimingData
      undefined, // onUsageData
      undefined, // traceId
      undefined, // vector_store_ids
      undefined, // guardrails
      undefined, // selectedMCPTools
      undefined, // onImageGenerated
      undefined, // onSearchResults
      0.7, // temperature
      100, // max_tokens
    );

    expect(mockCreate).toHaveBeenCalledTimes(1);
    const callArgs = mockCreate.mock.calls[0][0];
    expect(callArgs).toMatchObject({
      model: "gpt-4",
      stream: true,
      stream_options: {
        include_usage: true,
      },
      messages: mockChatHistory,
      temperature: 0.7,
      max_tokens: 100,
    });
  });
});
