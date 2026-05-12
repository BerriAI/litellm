import React from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect } from "vitest";
import { HistoryTree } from "./HistoryTree";
import { ParsedMessage } from "./prettyMessagesTypes";

describe("HistoryTree", () => {
  it("should return null when messages array is empty", () => {
    const { container } = render(<HistoryTree messages={[]} />);
    expect(container.innerHTML).toBe("");
  });

  it('should render message count with plural "messages" for multiple messages', () => {
    const messages: ParsedMessage[] = [
      { role: "user", content: "Hello" },
      { role: "assistant", content: "Hi there" },
      { role: "user", content: "How are you?" },
    ];
    render(<HistoryTree messages={messages} />);
    expect(
      screen.getByText("HISTORY (3 messages)")
    ).toBeInTheDocument();
  });

  it('should render message count with singular "message" for one message', () => {
    const messages: ParsedMessage[] = [
      { role: "user", content: "Hello" },
    ];
    render(<HistoryTree messages={messages} />);
    expect(
      screen.getByText("HISTORY (1 message)")
    ).toBeInTheDocument();
  });

  it("should expand and show messages when header is clicked", async () => {
    const user = userEvent.setup();
    const messages: ParsedMessage[] = [
      { role: "user", content: "Hello" },
      { role: "assistant", content: "Hi there" },
    ];
    render(<HistoryTree messages={messages} />);

    // Click to expand
    await user.click(screen.getByText("HISTORY (2 messages)"));

    expect(screen.getByText("Hello")).toBeInTheDocument();
    expect(screen.getByText("Hi there")).toBeInTheDocument();
  });
});
