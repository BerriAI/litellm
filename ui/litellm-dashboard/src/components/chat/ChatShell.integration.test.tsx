import { afterEach, describe, expect, it, vi } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders as render } from "@/../tests/test-utils";
import ChatShell from "./ChatShell";

const { mockPush, mockUsePathname, mockUseChatShell, mockUseMediaQuery } = vi.hoisted(() => ({
  mockPush: vi.fn(),
  mockUsePathname: vi.fn(() => "/ui/chat"),
  mockUseMediaQuery: vi.fn(() => true),
  mockUseChatShell: vi.fn(() => ({
    accessToken: "test-token",
    userEmail: "developer@example.com",
    userId: "test-user",
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
vi.mock("@/utils/uiHref", () => ({ uiHref: (seg: string) => `/ui/${seg}`.replace(/\/$/, "") || "/ui" }));
vi.mock("@/contexts/ChatShellContext", () => ({ useChatShell: mockUseChatShell }));
vi.mock("@/app/(dashboard)/hooks/uiSettings/useUISettings", () => ({
  useUISettings: () => ({ data: { values: { enable_chat_ui: true } } }),
}));
vi.mock("@/app/(dashboard)/hooks/useLogout", () => ({ useLogout: () => vi.fn() }));
vi.mock("usehooks-ts", () => ({ useMediaQuery: mockUseMediaQuery }));
vi.mock("./ConversationList", () => ({ default: () => <div data-testid="conversation-list" /> }));

describe("ChatShell", () => {
  afterEach(() => {
    mockPush.mockClear();
    mockUsePathname.mockReturnValue("/ui/chat");
    mockUseMediaQuery.mockReturnValue(true);
  });

  it("should render the workspace and mark Chats active on the base chat route", () => {
    render(
      <ChatShell>
        <div />
      </ChatShell>,
    );
    expect(screen.getByRole("button", { name: "Chats" })).toHaveAttribute("aria-current", "page");
    expect(screen.getByRole("button", { name: "Models" })).not.toHaveAttribute("aria-current");
    expect(screen.getByRole("button", { name: "Workspace" })).not.toHaveAttribute("aria-current");
    expect(screen.getByRole("button", { name: "Settings" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Sign out" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Switch workspace, current: Chat" })).toBeInTheDocument();
    expect(screen.getByTestId("conversation-list")).toBeInTheDocument();
  });

  it("should mark Models active on its route and tolerate a trailing slash", () => {
    mockUsePathname.mockReturnValue("/ui/chat/models/");
    render(
      <ChatShell>
        <div />
      </ChatShell>,
    );

    expect(screen.getByRole("button", { name: "Models" })).toHaveAttribute("aria-current", "page");
    expect(screen.getByRole("button", { name: "Chats" })).not.toHaveAttribute("aria-current");
    expect(screen.getByRole("button", { name: "Workspace" })).not.toHaveAttribute("aria-current");
  });

  it("should mark API Keys active while still showing the conversation list", () => {
    mockUsePathname.mockReturnValue("/ui/chat/api-keys");
    render(
      <ChatShell>
        <div />
      </ChatShell>,
    );
    expect(screen.getByRole("button", { name: "Workspace" })).toHaveAttribute("aria-current", "page");
    expect(screen.getByRole("button", { name: "Chats" })).not.toHaveAttribute("aria-current");
    expect(screen.getByTestId("conversation-list")).toBeInTheDocument();
  });

  it("should navigate through the workspace menu and retain all chat destinations", async () => {
    const user = userEvent.setup();
    render(
      <ChatShell>
        <div />
      </ChatShell>,
    );
    await user.click(screen.getByRole("button", { name: "Integrations" }));
    expect(mockPush).toHaveBeenCalledWith("/ui/chat/integrations");
    await user.click(screen.getByRole("button", { name: "Models" }));
    expect(mockPush).toHaveBeenCalledWith("/ui/chat/models");
    for (const [label, href] of [
      ["Credentials", "/ui/chat/credentials"],
      ["API Keys", "/ui/chat/api-keys"],
      ["Usage", "/ui/chat/usage"],
      ["Logs", "/ui/chat/logs"],
    ]) {
      await user.click(screen.getByRole("button", { name: "Workspace" }));
      await user.click(await screen.findByRole("menuitem", { name: label, exact: true }));
      expect(mockPush).toHaveBeenCalledWith(href);
    }
  });

  it("should mark the active workspace destination and tolerate a trailing slash", async () => {
    mockUsePathname.mockReturnValue("/ui/chat/usage/");
    const user = userEvent.setup();
    render(
      <ChatShell>
        <div />
      </ChatShell>,
    );
    expect(screen.getByRole("button", { name: "Workspace" })).toHaveAttribute("aria-current", "page");
    await user.click(screen.getByRole("button", { name: "Workspace" }));
    expect(await screen.findByRole("menuitem", { name: "Usage", exact: true })).toHaveAttribute("aria-current", "page");
    expect(screen.getByRole("menuitem", { name: "Logs", exact: true })).not.toHaveAttribute("aria-current");
  });

  it("should collapse and reopen the desktop rail", async () => {
    const user = userEvent.setup();
    render(
      <ChatShell>
        <div />
      </ChatShell>,
    );
    await user.click(screen.getByRole("button", { name: "Close sidebar" }));
    expect(screen.getByRole("button", { name: "Open sidebar" })).toHaveAttribute("aria-expanded", "false");
    await user.click(screen.getByRole("button", { name: "Switch workspace, current: Chat" }));
    expect(await screen.findByRole("menuitem", { name: "AI Gateway", exact: true })).toBeInTheDocument();
    await user.keyboard("{Escape}");
    await user.click(screen.getByRole("button", { name: "Open sidebar" }));
    expect(screen.getByRole("button", { name: "Close sidebar" })).toHaveAttribute("aria-expanded", "true");
  });

  it("should open mobile navigation and close it after selecting Models", async () => {
    mockUseMediaQuery.mockReturnValue(false);
    const user = userEvent.setup();
    render(
      <ChatShell>
        <div>Conversation content</div>
      </ChatShell>,
    );

    expect(screen.queryByRole("navigation", { name: "Chat workspace" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Switch workspace, current: Chat" })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Open chat navigation" }));
    expect(await screen.findByRole("dialog", { name: "LiteLLM Chat" })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Models" }));

    expect(mockPush).toHaveBeenCalledWith("/ui/chat/models");
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    expect(screen.getByText("Conversation content")).toBeInTheDocument();
  });

  it("should close mobile navigation with Escape and restore trigger focus", async () => {
    mockUseMediaQuery.mockReturnValue(false);
    const user = userEvent.setup();
    render(
      <ChatShell>
        <div />
      </ChatShell>,
    );

    const trigger = screen.getByRole("button", { name: "Open chat navigation" });
    await user.click(trigger);
    expect(await screen.findByRole("dialog", { name: "LiteLLM Chat" })).toBeInTheDocument();
    await user.keyboard("{Escape}");

    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    await waitFor(() => expect(trigger).toHaveFocus());
  });
});
