import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { renderWithProviders, screen, waitFor } from "../../tests/test-utils";
import Navbar from "./navbar";

// Mock the hooks and utilities
vi.mock("@/components/networking", () => ({
  getProxyBaseUrl: vi.fn(() => "http://localhost:4000"),
}));

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
    mockUseAuthorizedImpl = () => ({
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
    mockUseAuthorizedImpl = () => ({
      userId: "test-user",
      userEmail: "test@example.com",
      userRole: "Admin",
      premiumUser: false,
    });
  });

  it("should show version badge when health data contains version", () => {
    mockUseHealthReadinessImpl = () => ({ data: { litellm_version: "1.0.0" } });

    renderWithProviders(<Navbar {...defaultProps} />);

    expect(screen.getByText("v1.0.0")).toBeInTheDocument();

    // Reset mock
    mockUseHealthReadinessImpl = () => ({ data: null });
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
