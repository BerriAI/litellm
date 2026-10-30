import { act, fireEvent, render, screen, within } from "@testing-library/react";
import type { ComponentProps } from "react";
import { describe, expect, it, vi } from "vitest";
import MoyaiChatHome from "./MoyaiChatHome";
import type { Conversation } from "../types";

function homeProps(overrides: Partial<ComponentProps<typeof MoyaiChatHome>> = {}) {
  return {
    children: <span>Chat composer</span>,
    imageMode: false,
    draft: "",
    busy: false,
    conversations: [],
    onSelectConversation: vi.fn(),
    onPrompt: vi.fn(),
    ...overrides,
  };
}

function conversation(id: string, updatedAt: number): Conversation {
  return { id, title: `Chat ${id}`, model: "chat-model", messages: [], mcpServerNames: [], createdAt: 1, updatedAt };
}

describe("MoyaiChatHome", () => {
  it("should render the Moyai welcome and composer", () => {
    render(<MoyaiChatHome {...homeProps()} />);
    expect(screen.getByRole("heading", { level: 1, name: "What are we building?" })).toBeInTheDocument();
    expect(screen.getByText("Chat composer")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Recent chats" })).toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent("No recent chats in this browser yet.");
  });

  it("should show the three most recently updated chats and open the selected one", () => {
    const conversations = [
      conversation("first", 1),
      conversation("second", 4),
      conversation("third", 3),
      conversation("fourth", 2),
    ];
    const props = homeProps({ conversations });
    render(<MoyaiChatHome {...props} />);
    const recent = screen.getByRole("region", { name: "Recent chats" });
    const buttons = within(recent).getAllByRole("button");
    expect(buttons.map((button) => button.textContent)).toEqual(["Chat second", "Chat third", "Chat fourth"]);
    expect(screen.queryByRole("button", { name: "Chat first" })).not.toBeInTheDocument();
    act(() => {
      fireEvent.click(screen.getByRole("button", { name: "Chat third" }));
    });
    expect(props.onSelectConversation).toHaveBeenCalledWith("third");
    expect(conversations.map((item) => item.id)).toEqual(["first", "second", "third", "fourth"]);
  });

  it("should pass a starter prompt to the composer without submitting it", () => {
    const props = homeProps();
    render(<MoyaiChatHome {...props} />);
    act(() => {
      fireEvent.click(screen.getByRole("button", { name: /Review code/ }));
    });
    expect(props.onPrompt).toHaveBeenCalledWith("Review my changes for bugs and improvements.");
    expect(props.onSelectConversation).not.toHaveBeenCalled();
  });

  it("should disable starters during a request", () => {
    const props = homeProps({ busy: true });
    render(<MoyaiChatHome {...props} />);
    const starter = screen.getByRole("button", { name: /Review code/ });
    expect(starter).toBeDisabled();
    act(() => {
      fireEvent.click(starter);
    });
    expect(props.onPrompt).not.toHaveBeenCalled();
  });

  it("should hide starter choices from assistive technology once a draft is present", () => {
    render(<MoyaiChatHome {...homeProps({ draft: "My current prompt" })} />);
    expect(screen.queryByRole("group", { name: "Start with a prompt" })).not.toBeInTheDocument();
    for (const button of screen.getAllByRole("button", { hidden: true })) expect(button).toBeDisabled();
  });

  it("should show the image welcome without code starter prompts in image mode", () => {
    render(<MoyaiChatHome {...homeProps({ imageMode: true })} />);
    expect(screen.getByRole("heading", { level: 1, name: "What are we creating?" })).toBeInTheDocument();
    expect(screen.queryByRole("group", { name: "Start with a prompt" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Review code/ })).not.toBeInTheDocument();
  });
});
