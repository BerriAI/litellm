import { act, fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import ConversationSearchDialog from "./ConversationSearchDialog";
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

describe("ConversationSearchDialog", () => {
  it("should filter chats by title and select a result", async () => {
    const onSelect = vi.fn();
    const user = userEvent.setup();
    const onClose = vi.fn();
    render(<ConversationSearchDialog open conversations={conversations} onSelect={onSelect} onClose={onClose} />);

    const dialog = await screen.findByRole("dialog", { name: "Search conversations" });
    act(() =>
      fireEvent.change(screen.getByRole("textbox", { name: "Search conversations" }), { target: { value: "photo" } }),
    );
    expect(within(dialog).queryByRole("button", { name: /Design review/ })).not.toBeInTheDocument();
    await user.click(within(dialog).getByRole("button", { name: /Photo ideas/ }));
    expect(onSelect).toHaveBeenCalledWith("photos");
    expect(onClose).toHaveBeenCalled();
  });
});
