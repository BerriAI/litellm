import { render } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { MessageType } from "../../chat_ui/types";
import { MessageDisplay } from "./MessageDisplay";

vi.mock("../../chat_ui/ReasoningContent", () => ({
  default: ({ reasoningContent }: { reasoningContent: string }) => (
    <div data-testid="reasoning-content">{reasoningContent}</div>
  ),
}));

vi.mock("../../chat_ui/ResponseMetrics", () => ({
  default: () => <div data-testid="response-metrics">ResponseMetrics</div>,
}));

vi.mock("../../chat_ui/SearchResultsDisplay", () => ({
  SearchResultsDisplay: () => <div data-testid="search-results">SearchResultsDisplay</div>,
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
    const { getByText } = render(<MessageDisplay messages={messages} isLoading={false} />);
    expect(getByText("Hello")).toBeInTheDocument();
    expect(getByText("Hi there!")).toBeInTheDocument();
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
    const { getByText, getByTestId } = render(<MessageDisplay messages={messages} isLoading={false} />);
    expect(getByText("You")).toBeInTheDocument();
    expect(getByText("What is 2+2?")).toBeInTheDocument();
    expect(getByText("gpt-4")).toBeInTheDocument();
    expect(getByText("calculator")).toBeInTheDocument();
    expect(getByText("2+2 equals 4")).toBeInTheDocument();
    expect(getByTestId("response-metrics")).toBeInTheDocument();
  });
});
