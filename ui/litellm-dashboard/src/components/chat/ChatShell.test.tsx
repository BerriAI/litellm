import { afterEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import ChatShell from "./ChatShell";

const translationState = vi.hoisted(() => ({ language: "en" as "en" | "ru" }));

vi.mock("react-i18next", async () => {
  const { resources } = await import("@/i18n/catalog");
  return {
    useTranslation: () => ({
      t: (key: string) =>
        key.split(".").reduce<unknown>((copy, segment) => {
          if (typeof copy !== "object" || copy === null) return undefined;
          return (copy as Record<string, unknown>)[segment];
        }, resources[translationState.language].chat) ?? key,
    }),
  };
});

const { mockPush, mockUsePathname, mockUseChatShell } = vi.hoisted(() => ({
  mockPush: vi.fn(),
  mockUsePathname: vi.fn(() => "/ui/chat"),
  mockUseChatShell: vi.fn(() => ({
    conversations: [],
    activeConversationId: null,
    deleteConversation: vi.fn(),
    renameConversation: vi.fn(),
  })),
}));

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: mockPush }),
  usePathname: mockUsePathname,
}));
// Deterministic hrefs so navigation/active-state assertions don't depend on server_root_path.
vi.mock("@/utils/migratedPages", () => ({ migratedHref: (seg: string) => `/ui/${seg}`.replace(/\/$/, "") || "/ui" }));
vi.mock("@/contexts/ChatShellContext", () => ({ useChatShell: mockUseChatShell }));
vi.mock("./ConversationList", () => ({ default: () => <div data-testid="conversation-list" /> }));

describe("ChatShell", () => {
  afterEach(() => {
    translationState.language = "en";
    mockPush.mockClear();
    mockUsePathname.mockReturnValue("/ui/chat");
  });

  it("renders the chat navigation in Russian", () => {
    translationState.language = "ru";
    render(
      <ChatShell>
        <div />
      </ChatShell>,
    );

    expect(screen.getByRole("button", { name: "Новый чат" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Интеграции" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Использование" })).toBeInTheDocument();
  });

  it("marks Chats active and shows the conversation list on the base chat route", () => {
    render(
      <ChatShell>
        <div />
      </ChatShell>,
    );
    expect(screen.getByRole("button", { name: "Chats" })).toHaveAttribute("aria-current", "page");
    expect(screen.getByRole("button", { name: "API Keys" })).not.toHaveAttribute("aria-current");
    expect(screen.getByTestId("conversation-list")).toBeInTheDocument();
  });

  it("marks API Keys active while still showing the conversation list", () => {
    mockUsePathname.mockReturnValue("/ui/chat/api-keys");
    render(
      <ChatShell>
        <div />
      </ChatShell>,
    );
    expect(screen.getByRole("button", { name: "API Keys" })).toHaveAttribute("aria-current", "page");
    expect(screen.getByRole("button", { name: "Chats" })).not.toHaveAttribute("aria-current");
    expect(screen.getByTestId("conversation-list")).toBeInTheDocument();
  });

  it("navigates to the dedicated route for each nav item", () => {
    render(
      <ChatShell>
        <div />
      </ChatShell>,
    );
    fireEvent.click(screen.getByRole("button", { name: "Integrations" }));
    expect(mockPush).toHaveBeenCalledWith("/ui/chat/integrations");

    fireEvent.click(screen.getByRole("button", { name: "Usage" }));
    expect(mockPush).toHaveBeenCalledWith("/ui/chat/usage");

    fireEvent.click(screen.getByRole("button", { name: "Logs" }));
    expect(mockPush).toHaveBeenCalledWith("/ui/chat/logs");
  });

  it("marks Logs active on the logs route", () => {
    mockUsePathname.mockReturnValue("/ui/chat/logs");
    render(
      <ChatShell>
        <div />
      </ChatShell>,
    );
    expect(screen.getByRole("button", { name: "Logs" })).toHaveAttribute("aria-current", "page");
    expect(screen.getByRole("button", { name: "Usage" })).not.toHaveAttribute("aria-current");
  });

  it("tolerates a trailing slash on the current pathname when matching the active route", () => {
    mockUsePathname.mockReturnValue("/ui/chat/usage/");
    render(
      <ChatShell>
        <div />
      </ChatShell>,
    );
    expect(screen.getByRole("button", { name: "Usage" })).toHaveAttribute("aria-current", "page");
  });
});
