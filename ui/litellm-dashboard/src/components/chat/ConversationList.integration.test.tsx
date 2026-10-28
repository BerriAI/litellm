import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import ConversationList from "./ConversationList";
import type { Conversation } from "./types";

const conversations: Conversation[] = [
  {
    id: "design",
    title: "Design review",
    model: "test-model",
    messages: [],
    mcpServerNames: [],
    createdAt: 1,
    updatedAt: 1,
  },
  {
    id: "photos",
    title: "Photo ideas",
    model: "test-model",
    messages: [],
    mcpServerNames: [],
    createdAt: 2,
    updatedAt: 2,
  },
];

describe("ConversationList navigation", () => {
  it("should render named conversation actions and the active selection", () => {
    render(
      <ConversationList
        conversations={conversations}
        activeConversationId="design"
        onSelect={vi.fn()}
        onDelete={vi.fn()}
        onRename={vi.fn()}
      />,
    );

    expect(screen.getByRole("button", { name: "Design review", exact: true })).toHaveAttribute("aria-current", "page");
    expect(screen.getByRole("button", { name: "Rename Design review" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Delete Design review" })).toBeInTheDocument();
  });

  it("should select a conversation with the keyboard", async () => {
    const onSelect = vi.fn();
    const user = userEvent.setup();
    render(
      <ConversationList
        conversations={conversations}
        activeConversationId={null}
        onSelect={onSelect}
        onDelete={vi.fn()}
        onRename={vi.fn()}
      />,
    );

    await user.tab();
    expect(screen.getByRole("button", { name: "Search chats" })).toHaveFocus();
    await user.tab();
    expect(screen.getByRole("button", { name: "Design review", exact: true })).toHaveFocus();
    await user.keyboard("{Enter}");
    expect(onSelect).toHaveBeenCalledWith("design");
  });

  it("should filter chats from visible search and select a result", async () => {
    const onSelect = vi.fn();
    const user = userEvent.setup();
    render(
      <ConversationList
        conversations={conversations}
        activeConversationId={null}
        onSelect={onSelect}
        onDelete={vi.fn()}
        onRename={vi.fn()}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Search chats" }));
    const dialog = await screen.findByRole("dialog", { name: "Search conversations" });
    act(() =>
      fireEvent.change(screen.getByRole("textbox", { name: "Search conversations" }), { target: { value: "photo" } }),
    );
    expect(within(dialog).queryByRole("button", { name: /Design review/ })).not.toBeInTheDocument();
    await user.click(within(dialog).getByRole("button", { name: /Photo ideas/ }));
    expect(onSelect).toHaveBeenCalledWith("photos");
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
  });
});
