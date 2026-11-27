import { act, fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { I18nextProvider } from "react-i18next";
import i18n from "@/i18n/i18n";
import zh from "@/i18n/locales/zh-CN.json";
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
  it("should search persisted default titles in the active language without translating custom titles", async () => {
    const locale = i18n.cloneInstance({ forkResourceStore: true });
    locale.addResourceBundle("zh-CN", "translation", zh);
    await locale.changeLanguage("zh-CN");
    const defaultTitle = { ...conversations[0], title: "New conversation" };
    const customTitle = { ...conversations[1], title: "New conversation", hasCustomTitle: true };
    render(
      <I18nextProvider i18n={locale}>
        <ConversationSearchDialog
          open
          conversations={[defaultTitle, customTitle]}
          onSelect={vi.fn()}
          onClose={vi.fn()}
        />
      </I18nextProvider>,
    );

    expect(screen.getByRole("button", { name: /新对话/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /New conversation/ })).toBeInTheDocument();
    fireEvent.change(screen.getByRole("textbox", { name: "搜索对话" }), { target: { value: "新对话" } });
    expect(screen.getByRole("button", { name: /新对话/ })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /New conversation/ })).not.toBeInTheDocument();

    await act(() => locale.changeLanguage("en"));
    expect(screen.queryByRole("button", { name: /新对话/ })).not.toBeInTheDocument();
    fireEvent.change(screen.getByRole("textbox", { name: "Search conversations" }), {
      target: { value: "New conversation" },
    });
    expect(screen.getAllByRole("button", { name: /New conversation/ })).toHaveLength(2);
  });

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
