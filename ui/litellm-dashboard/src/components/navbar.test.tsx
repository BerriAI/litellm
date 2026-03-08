import userEvent from "@testing-library/user-event";
import React, { useState } from "react";
import { describe, expect, it, vi } from "vitest";
import { renderWithProviders, screen, waitFor } from "../../tests/test-utils";
import Navbar from "./navbar";

// Mock the hooks and utilities
vi.mock("@/components/networking", () => ({
  getProxyBaseUrl: vi.fn(() => "http://localhost:4000"),
  serverRootPath: "",
}));

let mockUseDisableBouncingIconImpl = () => false;
vi.mock("@/app/(dashboard)/hooks/useDisableBouncingIcon", () => ({
  useDisableBouncingIcon: () => mockUseDisableBouncingIconImpl(),
}));

let mockUseUIConfigImpl = () => ({
  data: {
    server_root_path: "/",
    proxy_base_url: null,
    auto_redirect_to_sso: false,
    admin_ui_disabled: false,
    sso_configured: false,
    disable_bouncing_icon: false,
  },
});
vi.mock("@/app/(dashboard)/hooks/uiConfig/useUIConfig", () => ({
  useUIConfig: () => mockUseUIConfigImpl(),
}));

vi.mock("./Navbar/BlogDropdown/BlogDropdown", () => ({
  BlogDropdown: () => <div data-testid="blog-dropdown">Blog</div>,
}));

const mockUserDropdownData = vi.hoisted(() => ({
  current: () => ({
    userId: "test-user",
    userEmail: "test@example.com",
    userRole: "Admin",
    premiumUser: false,
  }),
}));

vi.mock("./Navbar/UserDropdown/UserDropdown", async (importOriginal) => {
  const React = await import("react");
  const { useState } = React;
  const localStorageUtils = await import("@/utils/localStorageUtils");
  return {
    default: function MockUserDropdown({
      onLogout,
      disableBouncingIconByEnv,
    }: {
      onLogout: () => void;
      disableBouncingIconByEnv?: boolean;
    }) {
      const { userId, userEmail, userRole, premiumUser } = mockUserDropdownData.current();
      const [open, setOpen] = useState(false);
      return (
        <div>
          <button type="button" onClick={() => setOpen(!open)}>
            User
          </button>
          {open && (
            <div data-testid="user-dropdown-content">
              <span>{userId}</span>
              <span>{userRole}</span>
              <span>{userEmail}</span>
              {premiumUser && <span>Premium</span>}
              {disableBouncingIconByEnv && <span>Bouncing Icon Disabled by Env</span>}
              <button type="button" onClick={() => onLogout()}>
                Logout
              </button>
              <label>
                <input
                  type="checkbox"
                  aria-label="Toggle hide new feature indicators"
                  onChange={(e) => {
                    if (e.target.checked) {
                      localStorageUtils.setLocalStorageItem("disableShowNewBadge", "true");
                      localStorageUtils.emitLocalStorageChange("disableShowNewBadge");
                    }
                  }}
                />
                Toggle hide new feature indicators
              </label>
            </div>
          )}
        </div>
      );
    },
  };
});

vi.mock("@/utils/proxyUtils", () => ({
  fetchProxySettings: vi.fn(),
}));

// Mock CommunityEngagementButtons component
vi.mock("./Navbar/CommunityEngagementButtons/CommunityEngagementButtons", () => ({
  CommunityEngagementButtons: () => (
    <div data-testid="community-engagement-buttons">
      <a href="https://www.litellm.ai/support" target="_blank" rel="noopener noreferrer">
        Join Slack
      </a>
      <a href="https://github.com/BerriAI/litellm" target="_blank" rel="noopener noreferrer">
        Star us on GitHub
      </a>
    </div>
  ),
}));

// Create mock functions that can be controlled in tests
let mockUseThemeImpl = () => ({ logoUrl: null as string | null });
let mockUseHealthReadinessImpl = () => ({ data: null as any });
let mockGetLocalStorageItemImpl = (key: string) => null as string | null;
let mockUseAuthorizedImpl = () => ({
  userId: "test-user",
  userEmail: "test@example.com",
  userRole: "Admin",
  premiumUser: false,
});

vi.mock("@/contexts/ThemeContext", () => ({
  useTheme: () => mockUseThemeImpl(),
}));

vi.mock("@/app/(dashboard)/hooks/healthReadiness/useHealthReadiness", () => ({
  useHealthReadiness: () => mockUseHealthReadinessImpl(),
}));

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => mockUseAuthorizedImpl(),
}));

vi.mock("@/utils/localStorageUtils", () => ({
  LOCAL_STORAGE_EVENT: "local-storage-change",
  getLocalStorageItem: (key: string) => mockGetLocalStorageItemImpl(key),
  setLocalStorageItem: vi.fn(),
  removeLocalStorageItem: vi.fn(),
  emitLocalStorageChange: vi.fn(),
}));

vi.mock("@/utils/cookieUtils", () => ({
  clearTokenCookies: vi.fn(),
}));

// Mock window.location.href for logout testing
Object.defineProperty(window, "location", {
  value: { href: "" },
  writable: true,
});

describe("Navbar", () => {
  const defaultProps = {
    userID: "test-user",
    userEmail: "test@example.com",
    userRole: "Admin",
    premiumUser: false,
    proxySettings: {},
    setProxySettings: vi.fn(),
    accessToken: "test-token",
    isPublicPage: false,
    isDarkMode: false,
    toggleDarkMode: vi.fn(),
  };

  it("should render without crashing", () => {
    renderWithProviders(<Navbar {...defaultProps} />);

    expect(screen.getByText("Docs")).toBeInTheDocument();
    expect(screen.getByText("User")).toBeInTheDocument();
  });

  it("should display user information in dropdown", async () => {
    const user = userEvent.setup();
    renderWithProviders(<Navbar {...defaultProps} />);

    await user.click(screen.getByText("User"));

    await waitFor(() => {
      expect(screen.getByText("test-user")).toBeInTheDocument();
    });
    expect(screen.getByText("Admin")).toBeInTheDocument();
    expect(screen.getByText("test@example.com")).toBeInTheDocument();
  });

  it("should show sidebar toggle button when onToggleSidebar is provided", () => {
    const mockToggle = vi.fn();
    renderWithProviders(<Navbar {...defaultProps} onToggleSidebar={mockToggle} />);

    const toggleButton = screen.getByTitle("Collapse sidebar");
    expect(toggleButton).toBeInTheDocument();
  });

  it("should call onToggleSidebar when sidebar button is clicked", async () => {
    const mockToggle = vi.fn();
    const user = userEvent.setup();
    renderWithProviders(<Navbar {...defaultProps} onToggleSidebar={mockToggle} />);

    const toggleButton = screen.getByTitle("Collapse sidebar");
    await user.click(toggleButton);

    expect(mockToggle).toHaveBeenCalledTimes(1);
  });

  it("should show premium user badge when premiumUser is true", async () => {
    const user = userEvent.setup();
    const originalCurrent = mockUserDropdownData.current;
    mockUserDropdownData.current = () => ({
      userId: "test-user",
      userEmail: "test@example.com",
      userRole: "Admin",
      premiumUser: true,
    });
    renderWithProviders(<Navbar {...defaultProps} />);

    await user.click(screen.getByText("User"));

    await waitFor(() => {
      expect(screen.getByText("Premium")).toBeInTheDocument();
    });

    // Reset mock
    mockUserDropdownData.current = originalCurrent;
  });

  it("should show version badge when health data contains version", () => {
    mockUseHealthReadinessImpl = () => ({ data: { litellm_version: "1.0.0" } });

    renderWithProviders(<Navbar {...defaultProps} />);

    expect(screen.getByText("v1.0.0")).toBeInTheDocument();
    expect(screen.getByTitle("Thanks for using LiteLLM!")).toBeInTheDocument();

    // Reset mock
    mockUseHealthReadinessImpl = () => ({ data: null });
  });

  it("should hide bouncing icon when disable_bouncing_icon is set from ui config", () => {
    mockUseHealthReadinessImpl = () => ({ data: { litellm_version: "1.0.0" } });
    mockUseUIConfigImpl = () => ({
      data: {
        server_root_path: "/",
        proxy_base_url: null,
        auto_redirect_to_sso: false,
        admin_ui_disabled: false,
        sso_configured: false,
        disable_bouncing_icon: true,
      },
    });

    renderWithProviders(<Navbar {...defaultProps} />);

    expect(screen.getByText("v1.0.0")).toBeInTheDocument();
    expect(screen.queryByTitle("Thanks for using LiteLLM!")).not.toBeInTheDocument();

    mockUseHealthReadinessImpl = () => ({ data: null });
    mockUseUIConfigImpl = () => ({
      data: {
        server_root_path: "/",
        proxy_base_url: null,
        auto_redirect_to_sso: false,
        admin_ui_disabled: false,
        sso_configured: false,
        disable_bouncing_icon: false,
      },
    });
  });

  it("should pass env disable flag to user dropdown", async () => {
    const user = userEvent.setup();
    mockUseUIConfigImpl = () => ({
      data: {
        server_root_path: "/",
        proxy_base_url: null,
        auto_redirect_to_sso: false,
        admin_ui_disabled: false,
        sso_configured: false,
        disable_bouncing_icon: true,
      },
    });

    renderWithProviders(<Navbar {...defaultProps} />);
    await user.click(screen.getByText("User"));

    expect(screen.getByText("Bouncing Icon Disabled by Env")).toBeInTheDocument();

    mockUseUIConfigImpl = () => ({
      data: {
        server_root_path: "/",
        proxy_base_url: null,
        auto_redirect_to_sso: false,
        admin_ui_disabled: false,
        sso_configured: false,
        disable_bouncing_icon: false,
      },
    });
  });

  it("should use custom logo from theme context", () => {
    mockUseThemeImpl = () => ({ logoUrl: "https://example.com/custom-logo.png" });

    renderWithProviders(<Navbar {...defaultProps} />);

    const logoImg = screen.getByAltText("LiteLLM Brand");
    expect(logoImg).toHaveAttribute("src", "https://example.com/custom-logo.png");

    // Reset mock
    mockUseThemeImpl = () => ({ logoUrl: null });
  });

  it("should hide user dropdown on public pages", () => {
    const publicPageProps = { ...defaultProps, isPublicPage: true };
    renderWithProviders(<Navbar {...publicPageProps} />);

    expect(screen.queryByText("User")).not.toBeInTheDocument();
  });

  it("should handle hide new features toggle", async () => {
    const user = userEvent.setup();

    // Initially disabled
    mockGetLocalStorageItemImpl = (key: string) => {
      if (key === "disableShowNewBadge") return "false";
      return null;
    };

    renderWithProviders(<Navbar {...defaultProps} />);

    await user.click(screen.getByText("User"));

    await waitFor(() => {
      expect(screen.getByText("test-user")).toBeInTheDocument();
    });

    // Find and click the toggle switch
    const toggleSwitch = screen.getByLabelText("Toggle hide new feature indicators");
    await user.click(toggleSwitch);

    // The functions are mocked globally, so we can check if they were called
    // by accessing them through the mock registry
    const localStorageUtils = vi.mocked(await import("@/utils/localStorageUtils"));
    expect(localStorageUtils.setLocalStorageItem).toHaveBeenCalledWith("disableShowNewBadge", "true");
    expect(localStorageUtils.emitLocalStorageChange).toHaveBeenCalledWith("disableShowNewBadge");

    // Reset mock
    mockGetLocalStorageItemImpl = (key: string) => null;
  });

  it("should handle logout functionality", async () => {
    const user = userEvent.setup();

    renderWithProviders(<Navbar {...defaultProps} />);

    await user.click(screen.getByText("User"));

    await waitFor(() => {
      expect(screen.getByText("test-user")).toBeInTheDocument();
    });

    // Click logout
    await user.click(screen.getByText("Logout"));

    const cookieUtils = vi.mocked(await import("@/utils/cookieUtils"));
    expect(cookieUtils.clearTokenCookies).toHaveBeenCalled();
    expect(window.location.href).toBe("");
  });

  it("should not render dark mode toggle slider", () => {
    renderWithProviders(<Navbar {...defaultProps} />);

    // DO NOT RENDER THIS UNTIL ALL COMPONENTS ARE CONFIRMED TO SUPPORT DARK MODE STYLES. IT IS AN ISSUE IF THIS TEST FAILS.
    expect(screen.queryByTestId("dark-mode-toggle")).not.toBeInTheDocument();
  });
});
