import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { I18nextProvider } from "react-i18next";
import i18n from "@/i18n/i18n";
import zh from "@/i18n/locales/zh-CN.json";
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
  it("keeps the localized default title out of the rename field and saved title", async () => {
    const locale = i18n.cloneInstance({ forkResourceStore: true });
    locale.addResourceBundle("zh-CN", "translation", zh);
    await locale.changeLanguage("zh-CN");
    const onRename = vi.fn();
    const user = userEvent.setup();
    render(
      <I18nextProvider i18n={locale}>
        <ConversationList
          conversations={[{ ...conversations[0], title: "New conversation" }]}
          activeConversationId="design"
          onSelect={vi.fn()}
          onDelete={vi.fn()}
          onRename={onRename}
        />
      </I18nextProvider>,
    );

    await user.click(screen.getByRole("button", { name: "重命名 新对话" }));
    const input = screen.getByRole("textbox");
    expect(input).toHaveValue("New conversation");
    await user.keyboard("{Enter}");
    expect(onRename).not.toHaveBeenCalled();
    expect(screen.getByRole("button", { name: "新对话" })).toBeInTheDocument();
  });

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
    expect(screen.getByRole("button", { name: "Design review", exact: true })).toHaveFocus();
    await user.keyboard("{Enter}");
    expect(onSelect).toHaveBeenCalledWith("design");
  });
});
