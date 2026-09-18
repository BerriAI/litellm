import { afterEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import ChatShell from "./ChatShell";

const { mockPush, mockUsePathname, mockUseChatShell } = vi.hoisted(() => ({
  mockPush: vi.fn(),
  mockUsePathname: vi.fn(() => "/chat"),
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
vi.mock("@/contexts/ChatShellContext", () => ({ useChatShell: mockUseChatShell }));
vi.mock("./ConversationList", () => ({ default: () => <div data-testid="conversation-list" /> }));

describe("ChatShell", () => {
  afterEach(() => {
    mockPush.mockClear();
    mockUsePathname.mockReturnValue("/chat");
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
    mockUsePathname.mockReturnValue("/chat/api-keys");
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
    expect(mockPush).toHaveBeenCalledWith("/chat/integrations");

    fireEvent.click(screen.getByRole("button", { name: "Usage" }));
    expect(mockPush).toHaveBeenCalledWith("/chat/usage");

    fireEvent.click(screen.getByRole("button", { name: "Logs" }));
    expect(mockPush).toHaveBeenCalledWith("/chat/logs");
  });

  it("marks Logs active on the logs route", () => {
    mockUsePathname.mockReturnValue("/chat/logs");
    render(
      <ChatShell>
        <div />
      </ChatShell>,
    );
    expect(screen.getByRole("button", { name: "Logs" })).toHaveAttribute("aria-current", "page");
    expect(screen.getByRole("button", { name: "Usage" })).not.toHaveAttribute("aria-current");
  });

  it("tolerates a trailing slash on the current pathname when matching the active route", () => {
    mockUsePathname.mockReturnValue("/chat/usage/");
    render(
      <ChatShell>
        <div />
      </ChatShell>,
    );
    expect(screen.getByRole("button", { name: "Usage" })).toHaveAttribute("aria-current", "page");
  });
});
