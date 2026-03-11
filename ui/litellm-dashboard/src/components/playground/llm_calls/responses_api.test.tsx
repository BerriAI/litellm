import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { makeOpenAIResponsesRequest } from "./responses_api";
import { MessageType } from "../chat_ui/types";

vi.mock("@/components/networking", () => ({
  getProxyBaseUrl: vi.fn(() => "https://example.com"),
}));

const mockResponsesCreate = vi.fn();
const mockClient = {
  responses: {
    create: mockResponsesCreate,
  },
};

vi.mock("openai", () => ({
  default: {
    OpenAI: vi.fn(() => mockClient),
  },
}));

describe("responses_api", () => {
  const mockUpdateTextUI = vi.fn();
  const messages: MessageType[] = [{ role: "user", content: "Hello" }];

  beforeEach(() => {
    const mockEvents = [
      { type: "response.output_text.delta", delta: "Hi" },
      {
        type: "response.completed",
        response: {
          id: "resp_123",
          usage: { output_tokens: 2, input_tokens: 5, total_tokens: 7 },
        },
      },
    ];

    async function* mockStream() {
      for (const event of mockEvents) {
        yield event;
      }
    }

    mockResponsesCreate.mockResolvedValue(mockStream());
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("should send a basic responses request", async () => {
    await makeOpenAIResponsesRequest(messages, mockUpdateTextUI, "gpt-4", "test-token");

    expect(mockResponsesCreate).toHaveBeenCalledTimes(1);
    expect(mockResponsesCreate).toHaveBeenCalledWith(
      expect.objectContaining({
        model: "gpt-4",
        input: [
          {
            role: "user",
            content: "Hello",
            type: "message",
          },
        ],
        stream: true,
      }),
      { signal: undefined },
    );
    expect(mockUpdateTextUI).toHaveBeenCalledWith("assistant", "Hi", "gpt-4");
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
    const mcpServerToolRestrictions: Record<string, string[]> = {
      "server-1": ["toolA"],
      "server-2": ["toolB", "toolC"],
    };

    await makeOpenAIResponsesRequest(
      messages,
      mockUpdateTextUI,
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
      undefined, // previousResponseId
      undefined, // onResponseId
      undefined, // onMCPEvent
      undefined, // codeInterpreterEnabled
      undefined, // onCodeInterpreterResult
      undefined, // customBaseUrl
      mcpServers,
      mcpServerToolRestrictions,
    );

    const callArgs = mockResponsesCreate.mock.calls[0][0];
    expect(callArgs.tool_choice).toBe("auto");
    expect(callArgs.tools).toEqual([
      {
        type: "mcp",
        server_label: "litellm",
        server_url: "litellm_proxy/mcp/alpha",
        require_approval: "never",
        allowed_tools: ["toolA"],
      },
      {
        type: "mcp",
        server_label: "litellm",
        server_url: "litellm_proxy/mcp/Beta",
        require_approval: "never",
        allowed_tools: ["toolB", "toolC"],
      },
    ]);
  });
});
