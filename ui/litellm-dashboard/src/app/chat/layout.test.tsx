import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, screen } from "@testing-library/react";
import { renderWithProviders as render } from "@/../tests/test-utils";
import ChatLayout from "./layout";

const { mockUseAuthorized, mockUseUISettings, mockReplace, mockUiHref, state } = vi.hoisted(() => {
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
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn());
  });

  afterEach(() => {
    state.enableChatUI = false;
    state.isUISettingsLoading = false;
    state.isIdentityLoading = false;
    state.isAuthorized = true;
    state.userId = "user-1";
    mockReplace.mockClear();
    mockUiHref.mockClear();
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("should render the chat shell without recording a query when enable_chat_ui is on", () => {
    state.enableChatUI = true;
    render(
      <ChatLayout>
        <div data-testid="page-content" />
      </ChatLayout>,
    );
    expect(screen.getByTestId("chat-shell")).toBeInTheDocument();
    expect(screen.getByTestId("page-content")).toBeInTheDocument();
    expect(mockReplace).not.toHaveBeenCalled();
  });

  it("should redirect without recording a query when enable_chat_ui is off", () => {
    state.enableChatUI = false;
    render(
      <ChatLayout>
        <div data-testid="page-content" />
      </ChatLayout>,
    );
    expect(screen.queryByTestId("chat-shell")).not.toBeInTheDocument();
    expect(mockReplace).toHaveBeenCalledWith("/mocked-ui/");
  });

  it("should wait for UI settings without recording a query", () => {
    state.isUISettingsLoading = true;
    render(
      <ChatLayout>
        <div data-testid="page-content" />
      </ChatLayout>,
    );
    expect(screen.queryByTestId("chat-shell")).not.toBeInTheDocument();
    expect(mockReplace).not.toHaveBeenCalled();
  });

  it("should wait for the signed-in user's identity without recording a query", () => {
    state.enableChatUI = true;
    state.isIdentityLoading = true;
    render(
      <ChatLayout>
        <div data-testid="page-content" />
      </ChatLayout>,
    );
    expect(screen.queryByTestId("chat-shell")).not.toBeInTheDocument();
    expect(screen.queryByTestId("page-content")).not.toBeInTheDocument();
  });

  it("should hide Chat without recording a query when the signed-in user's identity cannot be confirmed", () => {
    state.enableChatUI = true;
    state.isAuthorized = false;
    render(
      <ChatLayout>
        <div data-testid="page-content" />
      </ChatLayout>,
    );
    expect(screen.queryByTestId("chat-shell")).not.toBeInTheDocument();
    expect(screen.queryByTestId("page-content")).not.toBeInTheDocument();
  });

  it("should not record a query or fetch telemetry when opening, focusing, or revisiting Chat", () => {
    state.enableChatUI = true;
    const visibility = vi.spyOn(document, "visibilityState", "get").mockReturnValue("hidden");
    const view = render(<ChatLayout>Chat content</ChatLayout>);

    act(() => fireEvent.focus(window));

    visibility.mockReturnValue("visible");
    act(() => fireEvent(document, new Event("visibilitychange")));

    view.rerender(<ChatLayout>Another Chat page</ChatLayout>);
    act(() => fireEvent.focus(window));

    view.unmount();
    render(<ChatLayout>Chat revisited</ChatLayout>);
    act(() => {
      fireEvent.focus(window);
      fireEvent(document, new Event("visibilitychange"));
    });
    expect(fetch).not.toHaveBeenCalled();
  });

  it("should not record a query or fetch telemetry when the signed-in account changes", () => {
    state.enableChatUI = true;
    const view = render(<ChatLayout>Chat content</ChatLayout>);

    state.userId = "user-2";
    view.rerender(<ChatLayout>Chat content</ChatLayout>);
    expect(screen.getByTestId("chat-shell")).toBeInTheDocument();

    state.userId = null;
    state.isAuthorized = false;
    view.rerender(<ChatLayout>Chat content</ChatLayout>);
    act(() => fireEvent.focus(window));
    expect(fetch).not.toHaveBeenCalled();
  });
});
