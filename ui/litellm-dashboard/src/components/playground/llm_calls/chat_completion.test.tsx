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
      undefined, // policies
      undefined, // selectedMCPServers
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

  it("should configure MCP tools per server with restrictions", async () => {
    const selectedMCPServers = ["server-1", "server-2"];
    const mcpServers = [
      {
        server_id: "server-1",
        alias: "alpha",
        server_name: "Alpha",
        url: "http://example.com",
        created_at: "2024-01-01",
        created_by: "test",
        updated_at: "2024-01-01",
        updated_by: "test",
      },
      {
        server_id: "server-2",
        server_name: "Beta",
        url: "http://example.com",
        created_at: "2024-01-01",
        created_by: "test",
        updated_at: "2024-01-01",
        updated_by: "test",
      },
    ];
    const mcpServerToolRestrictions = {
      "server-1": ["toolA", "toolB"],
      "server-2": ["toolC"],
    } as Record<string, string[]>;

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
      undefined, // policies
      selectedMCPServers,
      undefined, // onImageGenerated
      undefined, // onSearchResults
      undefined, // temperature
      undefined, // max_tokens
      undefined, // onTotalLatency
      undefined, // customBaseUrl
      mcpServers,
      mcpServerToolRestrictions,
    );

    const callArgs = mockCreate.mock.calls[0][0];
    expect(callArgs.tool_choice).toBe("auto");
    expect(callArgs.tools).toHaveLength(2);

    // Check first tool
    const firstTool = callArgs.tools[0];
    expect(firstTool.type).toBe("mcp");
    expect(firstTool.server_label).toBe("litellm");
    expect(firstTool.server_url).toBe("litellm_proxy/mcp/alpha");
    expect(firstTool.require_approval).toBe("never");
    expect(firstTool.allowed_tools).toEqual(["toolA", "toolB"]);

    // Check second tool
    const secondTool = callArgs.tools[1];
    expect(secondTool.type).toBe("mcp");
    expect(secondTool.server_label).toBe("litellm");
    expect(secondTool.server_url).toBe("litellm_proxy/mcp/Beta");
    expect(secondTool.require_approval).toBe("never");
    expect(secondTool.allowed_tools).toEqual(["toolC"]);
  });

  it("should include mock_testing_fallbacks in request body when mockTestFallbacks is true", async () => {
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
      undefined, // policies
      undefined, // selectedMCPServers
      undefined, // onImageGenerated
      undefined, // onSearchResults
      undefined, // temperature
      undefined, // max_tokens
      undefined, // onTotalLatency
      undefined, // customBaseUrl
      undefined, // mcpServers
      undefined, // mcpServerToolRestrictions
      undefined, // onMCPEvent
      true, // mockTestFallbacks
    );

    expect(mockCreate).toHaveBeenCalledTimes(1);
    const callArgs = mockCreate.mock.calls[0][0];
    expect(callArgs.mock_testing_fallbacks).toBe(true);
  });

  it("should not include mock_testing_fallbacks in request body when mockTestFallbacks is false or undefined", async () => {
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
      undefined, // policies
      undefined, // selectedMCPServers
      undefined, // onImageGenerated
      undefined, // onSearchResults
      undefined, // temperature
      undefined, // max_tokens
      undefined, // onTotalLatency
      undefined, // customBaseUrl
      undefined, // mcpServers
      undefined, // mcpServerToolRestrictions
      undefined, // onMCPEvent
      false, // mockTestFallbacks
    );

    expect(mockCreate).toHaveBeenCalledTimes(1);
    const callArgs = mockCreate.mock.calls[0][0];
    expect(callArgs).not.toHaveProperty("mock_testing_fallbacks");
  });
});
