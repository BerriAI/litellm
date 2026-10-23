import { afterEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, screen } from "@testing-library/react";
import { renderWithProviders as render } from "@/../tests/test-utils";
import ChatLayout from "./layout";

const { mockUseAuthorized, mockUseUISettings, mockReplace, mockUiHref, recordChatActive, state } = vi.hoisted(() => {
  const state = {
    enableChatUI: false,
    isUISettingsLoading: false,
    isIdentityLoading: false,
    isAuthorized: true,
    userId: "user-1" as string | null,
  };
  return {
    state,
    mockReplace: vi.fn(),
    recordChatActive: vi.fn(async () => {}),
    mockUiHref: vi.fn((segment: string) => `/mocked-ui/${segment}`),
    mockUseAuthorized: vi.fn(() => ({
      accessToken: "token-123",
      userRole: "Internal User",
      userId: state.userId,
      userEmail: "user@example.com",
      premiumUser: false,
      isLoading: state.isIdentityLoading,
      isAuthorized: state.isAuthorized,
    })),
    mockUseUISettings: vi.fn(() => ({
      data: { values: { enable_chat_ui: state.enableChatUI } },
      isLoading: state.isUISettingsLoading,
    })),
  };
});

vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace: mockReplace }),
}));
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({ default: mockUseAuthorized }));
vi.mock("@/app/(dashboard)/hooks/uiSettings/useUISettings", () => ({ useUISettings: mockUseUISettings }));
vi.mock("@/utils/uiHref", () => ({ uiHref: mockUiHref }));
vi.mock("@/lib/telemetry/chat", () => ({ recordChatActive }));
vi.mock("@/components/navbar", () => ({ default: () => <div data-testid="navbar" /> }));
vi.mock("@/contexts/ThemeContext", () => ({
  ThemeProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));
vi.mock("@/contexts/ChatShellContext", () => ({
  ChatShellProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));
vi.mock("@/contexts/PluginModeContext", () => ({
  PluginModeProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));
vi.mock("@/components/chat/ChatShell", () => ({
  default: ({ children }: { children: React.ReactNode }) => <div data-testid="chat-shell">{children}</div>,
}));

describe("ChatLayout", () => {
  afterEach(() => {
    state.enableChatUI = false;
    state.isUISettingsLoading = false;
    state.isIdentityLoading = false;
    state.isAuthorized = true;
    state.userId = "user-1";
    mockReplace.mockClear();
    mockUiHref.mockClear();
    recordChatActive.mockClear();
    vi.restoreAllMocks();
  });

  it("should render the chat shell and record activity when enable_chat_ui is on", () => {
    state.enableChatUI = true;
    render(
      <ChatLayout>
        <div data-testid="page-content" />
      </ChatLayout>,
    );
    expect(screen.getByTestId("chat-shell")).toBeInTheDocument();
    expect(screen.getByTestId("page-content")).toBeInTheDocument();
    expect(mockReplace).not.toHaveBeenCalled();
    expect(recordChatActive).toHaveBeenCalledExactlyOnceWith("user-1");
  });

  it("should redirect without recording activity when enable_chat_ui is off", () => {
    state.enableChatUI = false;
    render(
      <ChatLayout>
        <div data-testid="page-content" />
      </ChatLayout>,
    );
    expect(screen.queryByTestId("chat-shell")).not.toBeInTheDocument();
    expect(mockReplace).toHaveBeenCalledWith("/mocked-ui/");
    expect(recordChatActive).not.toHaveBeenCalled();
  });

  it("should wait for UI settings without recording activity", () => {
    state.isUISettingsLoading = true;
    render(
      <ChatLayout>
        <div data-testid="page-content" />
      </ChatLayout>,
    );
    expect(screen.queryByTestId("chat-shell")).not.toBeInTheDocument();
    expect(mockReplace).not.toHaveBeenCalled();
    expect(recordChatActive).not.toHaveBeenCalled();
  });

  it("should wait for the signed-in user's identity without recording activity", () => {
    state.enableChatUI = true;
    state.isIdentityLoading = true;
    render(
      <ChatLayout>
        <div data-testid="page-content" />
      </ChatLayout>,
    );
    expect(screen.queryByTestId("chat-shell")).not.toBeInTheDocument();
    expect(screen.queryByTestId("page-content")).not.toBeInTheDocument();
    expect(recordChatActive).not.toHaveBeenCalled();
  });

  it("should hide Chat without recording activity when the signed-in user's identity cannot be confirmed", () => {
    state.enableChatUI = true;
    state.isAuthorized = false;
    render(
      <ChatLayout>
        <div data-testid="page-content" />
      </ChatLayout>,
    );
    expect(screen.queryByTestId("chat-shell")).not.toBeInTheDocument();
    expect(screen.queryByTestId("page-content")).not.toBeInTheDocument();
    expect(recordChatActive).not.toHaveBeenCalled();
  });

  it("should record foreground activity and stop listening when Chat is unmounted", () => {
    state.enableChatUI = true;
    const visibility = vi.spyOn(document, "visibilityState", "get").mockReturnValue("hidden");
    const view = render(<ChatLayout>Chat content</ChatLayout>);

    act(() => fireEvent.focus(window));
    expect(recordChatActive).not.toHaveBeenCalled();

    visibility.mockReturnValue("visible");
    act(() => fireEvent(document, new Event("visibilitychange")));
    expect(recordChatActive).toHaveBeenCalledExactlyOnceWith("user-1");

    view.rerender(<ChatLayout>Another Chat page</ChatLayout>);
    expect(recordChatActive).toHaveBeenCalledTimes(1);
    act(() => fireEvent.focus(window));
    expect(recordChatActive).toHaveBeenCalledTimes(2);

    view.unmount();
    act(() => {
      fireEvent.focus(window);
      fireEvent(document, new Event("visibilitychange"));
    });
    expect(recordChatActive).toHaveBeenCalledTimes(2);
  });

  it("should wait for a known user and remove activity listeners when access is revoked", () => {
    state.enableChatUI = true;
    state.userId = null;
    const view = render(<ChatLayout>Chat content</ChatLayout>);
    expect(recordChatActive).not.toHaveBeenCalled();

    state.userId = "user-2";
    view.rerender(<ChatLayout>Chat content</ChatLayout>);
    expect(recordChatActive).toHaveBeenCalledExactlyOnceWith("user-2");

    state.isAuthorized = false;
    view.rerender(<ChatLayout>Chat content</ChatLayout>);
    act(() => fireEvent.focus(window));
    expect(recordChatActive).toHaveBeenCalledTimes(1);
  });
});
