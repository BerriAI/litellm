import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders, screen, waitFor } from "../../../tests/test-utils";
import SidebarAccountMenu from "./SidebarAccountMenu";

interface AuthMock {
  userId: string | null;
  userEmail: string | null;
  userRole: string;
  premiumUser: boolean;
  accessToken: string;
}

let mockUseAuthorizedImpl: () => AuthMock = () => ({
  userId: "test-user-id",
  userEmail: "test@example.com",
  userRole: "Admin",
  premiumUser: false,
  accessToken: "test-token",
});

let mockUseDisableShowPromptsImpl = () => false;
let mockUseDisableBouncingIconImpl = () => false;
let mockHealthDataImpl = (): { litellm_version?: string } | undefined => ({ litellm_version: "1.99.0" });

let mockGetLocalStorageItemImpl = (key: string): string | null => {
  if (key === "disableShowNewBadge") return null;
  if (key === "disableShowPrompts") return null;
  return null;
};

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => mockUseAuthorizedImpl(),
}));

vi.mock("@/app/(dashboard)/hooks/useDisableShowPrompts", () => ({
  useDisableShowPrompts: () => mockUseDisableShowPromptsImpl(),
}));

vi.mock("@/app/(dashboard)/hooks/useDisableUsageIndicator", () => ({
  useDisableUsageIndicator: () => false,
}));

vi.mock("@/app/(dashboard)/hooks/useDisableBlogPosts", () => ({
  useDisableBlogPosts: () => false,
}));

vi.mock("@/app/(dashboard)/hooks/useDisableBouncingIcon", () => ({
  useDisableBouncingIcon: () => mockUseDisableBouncingIconImpl(),
}));

vi.mock("@/app/(dashboard)/hooks/healthReadiness/useHealthReadinessDetails", () => ({
  useHealthReadinessDetails: () => ({ data: mockHealthDataImpl() }),
}));

vi.mock("@/utils/localStorageUtils", () => ({
  LOCAL_STORAGE_EVENT: "local-storage-change",
  getLocalStorageItem: (key: string) => mockGetLocalStorageItemImpl(key),
  setLocalStorageItem: vi.fn(),
  removeLocalStorageItem: vi.fn(),
  emitLocalStorageChange: vi.fn(),
}));

describe("SidebarAccountMenu", () => {
  const mockOnLogout = vi.fn();

  const getAccountTrigger = () => screen.getByRole("button", { name: /account menu/i });

  const openMenu = async (user: ReturnType<typeof userEvent.setup>) => {
    await user.click(getAccountTrigger());
    await waitFor(() => {
      expect(screen.getByTestId("sidebar-account-menu-panel")).toBeInTheDocument();
    });
  };

  beforeEach(() => {
    vi.clearAllMocks();
    mockUseAuthorizedImpl = () => ({
      userId: "test-user-id",
      userEmail: "test@example.com",
      userRole: "Admin",
      premiumUser: false,
      accessToken: "test-token",
    });
    mockUseDisableShowPromptsImpl = () => false;
    mockUseDisableBouncingIconImpl = () => false;
    mockHealthDataImpl = () => ({ litellm_version: "1.99.0" });
    mockGetLocalStorageItemImpl = (key: string): string | null => {
      if (key === "disableShowNewBadge") return null;
      if (key === "disableShowPrompts") return null;
      return null;
    };
  });

  it("should render the account trigger with initials", () => {
    renderWithProviders(<SidebarAccountMenu onLogout={mockOnLogout} />);
    expect(getAccountTrigger()).toBeInTheDocument();
    expect(screen.getByText("TE")).toBeInTheDocument();
  });

  it("should render only the avatar (no name/role) when collapsed", () => {
    renderWithProviders(<SidebarAccountMenu onLogout={mockOnLogout} collapsed />);
    expect(getAccountTrigger()).toBeInTheDocument();
    expect(screen.getByText("TE")).toBeInTheDocument();
    expect(screen.queryByText("Admin")).not.toBeInTheDocument();
    expect(screen.queryByText("test@example.com")).not.toBeInTheDocument();
  });

  it("should show email, user ID, and role when the menu is opened", async () => {
    const user = userEvent.setup();
    renderWithProviders(<SidebarAccountMenu onLogout={mockOnLogout} />);

    await openMenu(user);

    expect(screen.getAllByText("test@example.com").length).toBeGreaterThan(0);
    expect(screen.getByText("test-user-id")).toBeInTheDocument();
    expect(screen.getAllByText("Admin").length).toBeGreaterThan(0);
  });

  it("should display Standard tier for non-premium users", async () => {
    const user = userEvent.setup();
    renderWithProviders(<SidebarAccountMenu onLogout={mockOnLogout} />);

    await openMenu(user);

    expect(screen.getByText("Standard")).toBeInTheDocument();
  });

  it("should display Premium tier for premium users", async () => {
    const user = userEvent.setup();
    mockUseAuthorizedImpl = () => ({
      userId: "test-user-id",
      userEmail: "test@example.com",
      userRole: "Admin",
      premiumUser: true,
      accessToken: "test-token",
    });

    renderWithProviders(<SidebarAccountMenu onLogout={mockOnLogout} />);

    await openMenu(user);

    expect(screen.getByText("Premium")).toBeInTheDocument();
  });

  it("should render a clickable version badge linking to the release notes", async () => {
    const user = userEvent.setup();
    renderWithProviders(<SidebarAccountMenu onLogout={mockOnLogout} />);

    await openMenu(user);

    const versionLink = screen.getByRole("link", { name: /v1\.99\.0/ });
    expect(versionLink).toHaveAttribute("href", "https://docs.litellm.ai/release_notes");
    expect(versionLink).toHaveAttribute("target", "_blank");
  });

  it("should not render the version badge when the version is unavailable", async () => {
    const user = userEvent.setup();
    mockHealthDataImpl = () => undefined;
    renderWithProviders(<SidebarAccountMenu onLogout={mockOnLogout} />);

    await openMenu(user);

    expect(screen.queryByRole("link", { name: /^v/ })).not.toBeInTheDocument();
  });

  it("should show the bouncing icon by default", async () => {
    const user = userEvent.setup();
    renderWithProviders(<SidebarAccountMenu onLogout={mockOnLogout} />);
    await openMenu(user);
    expect(screen.getByTitle("Thanks for using LiteLLM!")).toBeInTheDocument();
  });

  it("should hide the bouncing icon when Hide Bouncing Icon is enabled", async () => {
    const user = userEvent.setup();
    mockUseDisableBouncingIconImpl = () => true;
    renderWithProviders(<SidebarAccountMenu onLogout={mockOnLogout} />);
    await openMenu(user);
    expect(screen.queryByTitle("Thanks for using LiteLLM!")).not.toBeInTheDocument();
  });

  it("should copy the email and show the confirmation checkmark on success", async () => {
    const user = userEvent.setup();
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", { value: { writeText }, configurable: true });

    renderWithProviders(<SidebarAccountMenu onLogout={mockOnLogout} />);
    await openMenu(user);

    const copyButton = screen.getByRole("button", { name: "Copy email" });
    expect(copyButton.querySelector(".lucide-copy")).toBeInTheDocument();

    await user.click(copyButton);

    expect(writeText).toHaveBeenCalledWith("test@example.com");
    await waitFor(() => expect(copyButton.querySelector(".lucide-check")).toBeInTheDocument());
  });

  it("should not show the checkmark when the clipboard write is rejected", async () => {
    const user = userEvent.setup();
    const writeText = vi.fn().mockRejectedValue(new Error("permission denied"));
    Object.defineProperty(navigator, "clipboard", { value: { writeText }, configurable: true });

    renderWithProviders(<SidebarAccountMenu onLogout={mockOnLogout} />);
    await openMenu(user);

    const copyButton = screen.getByRole("button", { name: "Copy email" });
    await user.click(copyButton);

    await waitFor(() => expect(writeText).toHaveBeenCalledWith("test@example.com"));
    expect(copyButton.querySelector(".lucide-check")).not.toBeInTheDocument();
    expect(copyButton.querySelector(".lucide-copy")).toBeInTheDocument();
  });

  it("should not show the checkmark when the clipboard API is unavailable", async () => {
    const user = userEvent.setup();
    Object.defineProperty(navigator, "clipboard", { value: undefined, configurable: true });

    renderWithProviders(<SidebarAccountMenu onLogout={mockOnLogout} />);
    await openMenu(user);

    const copyButton = screen.getByRole("button", { name: "Copy email" });
    await user.click(copyButton);

    expect(copyButton.querySelector(".lucide-check")).not.toBeInTheDocument();
    expect(copyButton.querySelector(".lucide-copy")).toBeInTheDocument();
  });

  it("should call onLogout when logout is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(<SidebarAccountMenu onLogout={mockOnLogout} />);

    await openMenu(user);

    await user.click(screen.getByRole("button", { name: /logout/i }));

    expect(mockOnLogout).toHaveBeenCalledTimes(1);
  });

  it("should toggle hide new feature indicators on", async () => {
    const user = userEvent.setup();
    renderWithProviders(<SidebarAccountMenu onLogout={mockOnLogout} />);

    await openMenu(user);

    const toggle = screen.getByLabelText("Toggle hide new feature indicators");
    expect(toggle).not.toBeChecked();

    await user.click(toggle);

    const localStorageUtils = vi.mocked(await import("@/utils/localStorageUtils"));
    expect(localStorageUtils.setLocalStorageItem).toHaveBeenCalledWith("disableShowNewBadge", "true");
    expect(localStorageUtils.emitLocalStorageChange).toHaveBeenCalledWith("disableShowNewBadge");
  });

  it("should toggle hide new feature indicators off", async () => {
    const user = userEvent.setup();
    mockGetLocalStorageItemImpl = (key: string): string | null => {
      if (key === "disableShowNewBadge") return "true";
      return null;
    };

    renderWithProviders(<SidebarAccountMenu onLogout={mockOnLogout} />);

    await openMenu(user);

    const toggle = screen.getByLabelText("Toggle hide new feature indicators");
    expect(toggle).toBeChecked();

    await user.click(toggle);

    const localStorageUtils = vi.mocked(await import("@/utils/localStorageUtils"));
    expect(localStorageUtils.removeLocalStorageItem).toHaveBeenCalledWith("disableShowNewBadge");
    expect(localStorageUtils.emitLocalStorageChange).toHaveBeenCalledWith("disableShowNewBadge");
  });

  it("should toggle hide all prompts on", async () => {
    const user = userEvent.setup();
    renderWithProviders(<SidebarAccountMenu onLogout={mockOnLogout} />);

    await openMenu(user);

    const toggle = screen.getByLabelText("Toggle hide all prompts");
    expect(toggle).not.toBeChecked();

    await user.click(toggle);

    const localStorageUtils = vi.mocked(await import("@/utils/localStorageUtils"));
    expect(localStorageUtils.setLocalStorageItem).toHaveBeenCalledWith("disableShowPrompts", "true");
    expect(localStorageUtils.emitLocalStorageChange).toHaveBeenCalledWith("disableShowPrompts");
  });

  it("should initialize hide new feature indicators from localStorage", async () => {
    const user = userEvent.setup();
    mockGetLocalStorageItemImpl = (key: string): string | null => {
      if (key === "disableShowNewBadge") return "true";
      return null;
    };

    renderWithProviders(<SidebarAccountMenu onLogout={mockOnLogout} />);

    await openMenu(user);

    const toggle = screen.getByLabelText("Toggle hide new feature indicators");
    expect(toggle).toBeChecked();
  });

  it("should show Account in the trigger for the default placeholder user id", () => {
    mockUseAuthorizedImpl = () => ({
      userId: "default_user_id",
      userEmail: null,
      userRole: "Admin",
      premiumUser: false,
      accessToken: "test-token",
    });
    renderWithProviders(<SidebarAccountMenu onLogout={mockOnLogout} />);
    expect(screen.getByText("Account")).toBeInTheDocument();
  });

  it("should display a dash when email is unavailable", async () => {
    const user = userEvent.setup();
    mockUseAuthorizedImpl = () => ({
      userId: "test-user-id",
      userEmail: null,
      userRole: "Admin",
      premiumUser: false,
      accessToken: "test-token",
    });

    renderWithProviders(<SidebarAccountMenu onLogout={mockOnLogout} />);

    await openMenu(user);

    expect(screen.getByText("-")).toBeInTheDocument();
  });
});
