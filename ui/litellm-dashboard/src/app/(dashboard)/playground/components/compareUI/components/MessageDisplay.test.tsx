import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { MessageType } from "@/components/chat_ui/types";
import { MessageDisplay } from "./MessageDisplay";

vi.mock("@/components/chat_ui/ReasoningContent", () => ({
  default: ({ reasoningContent }: { reasoningContent: string }) => (
    <div data-testid="reasoning-content">{reasoningContent}</div>
  ),
}));

vi.mock("@/components/chat_ui/ResponseMetrics", () => ({
  default: () => <div data-testid="response-metrics">ResponseMetrics</div>,
}));

vi.mock("../../chat_ui/SearchResultsDisplay", () => ({
  SearchResultsDisplay: () => <div data-testid="search-results">SearchResultsDisplay</div>,
}));

vi.mock("../../chat_ui/ChatImageRenderer", () => ({
  default: ({ message }: { message: any }) =>
    message.imagePreviewUrl ? (
      <div data-testid="chat-image-renderer">
        <img src={message.imagePreviewUrl} alt="User uploaded image" />
      </div>
    ) : null,
}));

describe("MessageDisplay", () => {
  it("should render", () => {
    const messages: MessageType[] = [
      {
        role: "user",
        content: "Hello",
      },
      {
        role: "assistant",
        content: "Hi there!",
        model: "gpt-4",
      },
    ];
    render(<MessageDisplay messages={messages} isLoading={false} />);
    expect(screen.getByText("Hello")).toBeInTheDocument();
    expect(screen.getByText("Hi there!")).toBeInTheDocument();
  });

  it("displays user and assistant messages with proper grouping and shows loading state", () => {
    const messages: MessageType[] = [
      {
        role: "user",
        content: "What is 2+2?",
      },
      {
        role: "assistant",
        content: "2+2 equals 4",
        model: "gpt-4",
        toolName: "calculator",
        timeToFirstToken: 100,
        totalLatency: 500,
        usage: {
          completionTokens: 10,
          promptTokens: 20,
          totalTokens: 30,
        },
      },
    ];
    render(<MessageDisplay messages={messages} isLoading={false} />);
    expect(screen.getByText("You")).toBeInTheDocument();
    expect(screen.getByText("What is 2+2?")).toBeInTheDocument();
    expect(screen.getByText("gpt-4")).toBeInTheDocument();
    expect(screen.getByText("calculator")).toBeInTheDocument();
    expect(screen.getByText("2+2 equals 4")).toBeInTheDocument();
    expect(screen.getByTestId("response-metrics")).toBeInTheDocument();
  });

  it("should display image attachment in user message", () => {
    const messages: MessageType[] = [
      {
        role: "user",
        content: "What is in this image? [Image attached]",
        imagePreviewUrl: "blob:test-image-url",
      },
      {
        role: "assistant",
        content: "This is a test image",
        model: "gpt-4",
      },
    ];
    render(<MessageDisplay messages={messages} isLoading={false} />);
    expect(screen.getByText("What is in this image? [Image attached]")).toBeInTheDocument();
    expect(screen.getByTestId("chat-image-renderer")).toBeInTheDocument();
    const image = screen.getByTestId("chat-image-renderer").querySelector("img");
    expect(image).toHaveAttribute("src", "blob:test-image-url");
  });
});
