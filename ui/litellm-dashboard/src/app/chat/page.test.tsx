import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import ChatPageRoute from "./page";

const { mockUseAuthorized, mockUseUISettings, mockUseUIConfig, mockReplace, state } = vi.hoisted(() => {
  const state = {
    userRole: "Internal User" as string,
    enableChatUI: false,
    isUISettingsLoading: false,
    serverRootPath: undefined as string | undefined,
  };
  return {
    state,
    mockReplace: vi.fn(),
    mockUseAuthorized: vi.fn(() => ({
      accessToken: "token-123",
      userRole: state.userRole,
      userId: "user-1",
      userEmail: "user@example.com",
    })),
    mockUseUISettings: vi.fn(() => ({
      data: { values: { enable_chat_ui: state.enableChatUI } },
      isLoading: state.isUISettingsLoading,
    })),
    mockUseUIConfig: vi.fn(() => ({ data: { server_root_path: state.serverRootPath } })),
  };
});

vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace: mockReplace }),
}));
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({ default: mockUseAuthorized }));
vi.mock("@/app/(dashboard)/hooks/uiSettings/useUISettings", () => ({ useUISettings: mockUseUISettings }));
vi.mock("@/app/(dashboard)/hooks/uiConfig/useUIConfig", () => ({ useUIConfig: mockUseUIConfig }));
vi.mock("@/components/chat/ChatPage", () => ({ default: () => <div data-testid="chat-page" /> }));

describe("ChatPageRoute", () => {
  afterEach(() => {
    state.userRole = "Internal User";
    state.enableChatUI = false;
    state.isUISettingsLoading = false;
    state.serverRootPath = undefined;
    mockReplace.mockClear();
  });

  it("renders the chat page when enable_chat_ui is on", () => {
    state.enableChatUI = true;
    render(<ChatPageRoute />);
    expect(screen.getByTestId("chat-page")).toBeInTheDocument();
    expect(mockReplace).not.toHaveBeenCalled();
  });

  it("redirects non-admins to the dashboard when enable_chat_ui is off", () => {
    state.enableChatUI = false;
    state.userRole = "Internal User";
    render(<ChatPageRoute />);
    expect(screen.queryByTestId("chat-page")).not.toBeInTheDocument();
    expect(mockReplace).toHaveBeenCalledWith("/ui/");
  });

  it("redirects admins to the dashboard when enable_chat_ui is off", () => {
    state.enableChatUI = false;
    state.userRole = "Admin";
    render(<ChatPageRoute />);
    expect(screen.queryByTestId("chat-page")).not.toBeInTheDocument();
    expect(mockReplace).toHaveBeenCalledWith("/ui/");
  });

  it("renders nothing while UI settings are still loading", () => {
    state.isUISettingsLoading = true;
    render(<ChatPageRoute />);
    expect(screen.queryByTestId("chat-page")).not.toBeInTheDocument();
    expect(mockReplace).not.toHaveBeenCalled();
  });

  it("respects a custom server_root_path when redirecting", () => {
    state.enableChatUI = false;
    state.userRole = "Internal User";
    state.serverRootPath = "/api/v1";
    render(<ChatPageRoute />);
    expect(mockReplace).toHaveBeenCalledWith("/api/v1/ui/");
  });
});
