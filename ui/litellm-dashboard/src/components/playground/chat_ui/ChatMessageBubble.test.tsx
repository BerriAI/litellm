import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import ChatMessageBubble from "./ChatMessageBubble";
import { EndpointType } from "./mode_endpoint_mapping";
import { MessageType } from "./types";

// Mock child components to isolate bubble rendering logic
vi.mock("react-markdown", () => ({
  default: ({ children }: { children: string }) => <div data-testid="react-markdown">{children}</div>,
}));

vi.mock("react-syntax-highlighter", () => ({
  Prism: ({ children }: { children: string }) => <pre data-testid="syntax-highlighter">{children}</pre>,
}));

vi.mock("react-syntax-highlighter/dist/esm/styles/prism", () => ({
  coy: {},
}));

vi.mock("./ReasoningContent", () => ({
  default: ({ reasoningContent }: { reasoningContent: string }) => (
    <div data-testid="reasoning-content">{reasoningContent}</div>
  ),
}));

vi.mock("./MCPEventsDisplay", () => ({
  default: ({ events }: { events: unknown[] }) => (
    <div data-testid="mcp-events-display">{events.length} events</div>
  ),
}));

vi.mock("./SearchResultsDisplay", () => ({
  SearchResultsDisplay: ({ searchResults }: { searchResults: unknown[] }) => (
    <div data-testid="search-results-display">{searchResults.length} results</div>
  ),
}));

vi.mock("./ResponseMetrics", () => ({
  default: ({ timeToFirstToken }: { timeToFirstToken?: number }) => (
    <div data-testid="response-metrics">TTFT: {timeToFirstToken}</div>
  ),
}));

vi.mock("./A2AMetrics", () => ({
  default: ({ a2aMetadata }: { a2aMetadata: unknown }) => (
    <div data-testid="a2a-metrics">A2A</div>
  ),
}));

vi.mock("./CodeInterpreterOutput", () => ({
  default: ({ code }: { code: string }) => <div data-testid="code-interpreter-output">{code}</div>,
}));

vi.mock("./AudioRenderer", () => ({
  default: ({ message }: { message: MessageType }) => (
    <div data-testid="audio-renderer">{typeof message.content === "string" ? message.content : ""}</div>
  ),
}));

vi.mock("./ResponsesImageRenderer", () => ({
  default: () => <div data-testid="responses-image-renderer" />,
}));

vi.mock("./ChatImageRenderer", () => ({
  default: () => <div data-testid="chat-image-renderer" />,
}));

const defaultProps = {
  isLastMessage: false,
  endpointType: EndpointType.CHAT,
  mcpEvents: [],
  codeInterpreterResult: null,
  accessToken: "test-token",
};

describe("ChatMessageBubble", () => {
  it("should render a user message with right-aligned text", () => {
    render(
      <ChatMessageBubble
        {...defaultProps}
        message={{ role: "user", content: "Hello" }}
      />,
    );

    expect(screen.getByText("user")).toBeInTheDocument();
    expect(screen.getByText("Hello")).toBeInTheDocument();
  });

  it("should render an assistant message with left-aligned text", () => {
    render(
      <ChatMessageBubble
        {...defaultProps}
        message={{ role: "assistant", content: "Hi there" }}
      />,
    );

    expect(screen.getByText("assistant")).toBeInTheDocument();
    expect(screen.getByText("Hi there")).toBeInTheDocument();
  });

  it("should show model badge for assistant messages when model is provided", () => {
    render(
      <ChatMessageBubble
        {...defaultProps}
        message={{ role: "assistant", content: "Reply", model: "gpt-4" }}
      />,
    );

    expect(screen.getByText("gpt-4")).toBeInTheDocument();
  });

  it("should not show model badge for user messages even when model is set", () => {
    render(
      <ChatMessageBubble
        {...defaultProps}
        message={{ role: "user", content: "Hello", model: "gpt-4" }}
      />,
    );

    expect(screen.queryByText("gpt-4")).not.toBeInTheDocument();
  });

  it("should render markdown content via ReactMarkdown", () => {
    render(
      <ChatMessageBubble
        {...defaultProps}
        message={{ role: "assistant", content: "**bold text**" }}
      />,
    );

    expect(screen.getByTestId("react-markdown")).toHaveTextContent("**bold text**");
  });

  it("should render an image when isImage is true", () => {
    render(
      <ChatMessageBubble
        {...defaultProps}
        message={{ role: "assistant", content: "https://example.com/img.png", isImage: true }}
      />,
    );

    expect(screen.getByAltText("Generated image")).toHaveAttribute("src", "https://example.com/img.png");
  });

  it("should render AudioRenderer when isAudio is true", () => {
    render(
      <ChatMessageBubble
        {...defaultProps}
        message={{ role: "assistant", content: "audio-url", isAudio: true }}
      />,
    );

    expect(screen.getByTestId("audio-renderer")).toBeInTheDocument();
  });

  it("should show ReasoningContent when reasoningContent is present", () => {
    render(
      <ChatMessageBubble
        {...defaultProps}
        message={{ role: "assistant", content: "answer", reasoningContent: "thinking..." }}
      />,
    );

    expect(screen.getByTestId("reasoning-content")).toHaveTextContent("thinking...");
  });

  it("should show MCP events on the last assistant message for RESPONSES endpoint", () => {
    const mcpEvents = [{ type: "tool_call", item_id: "1" }];

    render(
      <ChatMessageBubble
        {...defaultProps}
        isLastMessage={true}
        endpointType={EndpointType.RESPONSES}
        mcpEvents={mcpEvents as any}
        message={{ role: "assistant", content: "response" }}
      />,
    );

    expect(screen.getByTestId("mcp-events-display")).toHaveTextContent("1 events");
  });

  it("should show MCP events on the last assistant message for CHAT endpoint", () => {
    const mcpEvents = [{ type: "tool_call", item_id: "1" }];

    render(
      <ChatMessageBubble
        {...defaultProps}
        isLastMessage={true}
        endpointType={EndpointType.CHAT}
        mcpEvents={mcpEvents as any}
        message={{ role: "assistant", content: "response" }}
      />,
    );

    expect(screen.getByTestId("mcp-events-display")).toHaveTextContent("1 events");
  });

  it("should not show MCP events when isLastMessage is false", () => {
    const mcpEvents = [{ type: "tool_call", item_id: "1" }];

    render(
      <ChatMessageBubble
        {...defaultProps}
        isLastMessage={false}
        endpointType={EndpointType.RESPONSES}
        mcpEvents={mcpEvents as any}
        message={{ role: "assistant", content: "response" }}
      />,
    );

    expect(screen.queryByTestId("mcp-events-display")).not.toBeInTheDocument();
  });

  it("should show SearchResultsDisplay when searchResults are present", () => {
    render(
      <ChatMessageBubble
        {...defaultProps}
        message={{
          role: "assistant",
          content: "found results",
          searchResults: [{ object: "search", search_query: "q", data: [] }],
        }}
      />,
    );

    expect(screen.getByTestId("search-results-display")).toBeInTheDocument();
  });

  it("should show ResponseMetrics when usage data is present and no a2aMetadata", () => {
    render(
      <ChatMessageBubble
        {...defaultProps}
        message={{
          role: "assistant",
          content: "response",
          timeToFirstToken: 150,
          usage: { completionTokens: 10, promptTokens: 5, totalTokens: 15 },
        }}
      />,
    );

    expect(screen.getByTestId("response-metrics")).toBeInTheDocument();
  });

  it("should show A2AMetrics when a2aMetadata is present instead of ResponseMetrics", () => {
    render(
      <ChatMessageBubble
        {...defaultProps}
        message={{
          role: "assistant",
          content: "agent response",
          timeToFirstToken: 100,
          a2aMetadata: { taskId: "task-1", status: { state: "completed" } },
        }}
      />,
    );

    expect(screen.getByTestId("a2a-metrics")).toBeInTheDocument();
    expect(screen.queryByTestId("response-metrics")).not.toBeInTheDocument();
  });

  it("should show CodeInterpreterOutput on the last assistant message for RESPONSES endpoint", () => {
    render(
      <ChatMessageBubble
        {...defaultProps}
        isLastMessage={true}
        endpointType={EndpointType.RESPONSES}
        codeInterpreterResult={{
          code: "print('hello')",
          containerId: "container-1",
          annotations: [],
        }}
        message={{ role: "assistant", content: "result" }}
      />,
    );

    expect(screen.getByTestId("code-interpreter-output")).toHaveTextContent("print('hello')");
  });

  it("should render generated image from chat completions via message.image", () => {
    render(
      <ChatMessageBubble
        {...defaultProps}
        message={{
          role: "assistant",
          content: "Here is your image",
          image: { url: "https://example.com/generated.png", detail: "auto" },
        }}
      />,
    );

    const images = screen.getAllByAltText("Generated image");
    expect(images.some((img) => img.getAttribute("src") === "https://example.com/generated.png")).toBe(true);
  });
});
