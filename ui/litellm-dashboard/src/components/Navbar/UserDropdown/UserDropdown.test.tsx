import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders, screen, waitFor } from "../../../../tests/test-utils";
import UserDropdown from "./UserDropdown";
import { LanguageProvider } from "@/contexts/LanguageContext";

let mockUseAuthorizedImpl: () => {
  userId: string | null;
  userEmail: string | null;
  userRoleLabel: string;
  premiumUser: boolean;
  loginMethod?: string | null;
} = () => ({
  userId: "test-user-id",
  userEmail: "test@example.com",
  userRoleLabel: "Admin",
  premiumUser: false,
});

const mockRouterPush = vi.fn();

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: mockRouterPush }),
}));

let mockUseDisableShowPromptsImpl = () => false;

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

vi.mock("@/utils/localStorageUtils", () => ({
  LOCAL_STORAGE_EVENT: "local-storage-change",
  getLocalStorageItem: (key: string) => mockGetLocalStorageItemImpl(key),
  setLocalStorageItem: vi.fn(),
  removeLocalStorageItem: vi.fn(),
  emitLocalStorageChange: vi.fn(),
}));

describe("UserDropdown", () => {
  const mockOnLogout = vi.fn();

  const getAccountTrigger = () => screen.getByRole("button", { name: /account menu/i });

  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
    mockUseAuthorizedImpl = () => ({
      userId: "test-user-id",
      userEmail: "test@example.com",
      userRoleLabel: "Admin",
      premiumUser: false,
    });
    mockUseDisableShowPromptsImpl = () => false;
    mockGetLocalStorageItemImpl = (key: string): string | null => {
      if (key === "disableShowNewBadge") return null;
      if (key === "disableShowPrompts") return null;
      return null;
    };
  });

  it("should render", () => {
    renderWithProviders(<UserDropdown onLogout={mockOnLogout} />);
    expect(getAccountTrigger()).toBeInTheDocument();
  });

  it("汉化账户占位名和固定角色，保留真实账户标识", async () => {
    mockUseAuthorizedImpl = () => ({
      userId: "default_user_id",
      userEmail: null,
      userRoleLabel: "Admin",
      premiumUser: false,
    });
    const user = userEvent.setup();
    renderWithProviders(
      <LanguageProvider defaultLanguage="zh-CN" userId="navbar-test" storageScope="navbar-test">
        <UserDropdown onLogout={mockOnLogout} variant="sidebar" />
      </LanguageProvider>,
    );
    const trigger = screen.getByRole("button", { name: "账户菜单，管理员，当前登录账户 default_user_id" });
    expect(trigger).toHaveTextContent("账户");
    await user.click(trigger);
    expect(screen.getByText("default_user_id")).toBeInTheDocument();
    expect(screen.getAllByText("管理员")).toHaveLength(2);
  });

  it("不把自定义角色和用户 ID 当作翻译键", async () => {
    mockUseAuthorizedImpl = () => ({
      userId: "Account",
      userEmail: null,
      userRoleLabel: "Home",
      premiumUser: false,
    });
    const user = userEvent.setup();
    renderWithProviders(
      <LanguageProvider defaultLanguage="zh-CN" userId="navbar-test" storageScope="navbar-test">
        <UserDropdown onLogout={mockOnLogout} variant="sidebar" />
      </LanguageProvider>,
    );
    const trigger = screen.getByRole("button", { name: "账户菜单，Home，当前登录账户 Account" });
    expect(trigger).toHaveTextContent("Account");
    await user.click(trigger);
    expect(screen.getAllByText("Home")).toHaveLength(2);
  });

  it("should surface initials and account menu affordance", () => {
    renderWithProviders(<UserDropdown onLogout={mockOnLogout} />);
    expect(getAccountTrigger()).toBeInTheDocument();
    expect(screen.getByText("TE")).toBeInTheDocument();
  });

  it("should show user email when dropdown is opened", async () => {
    const user = userEvent.setup();
    renderWithProviders(<UserDropdown onLogout={mockOnLogout} />);

    await user.click(getAccountTrigger());

    await waitFor(() => {
      expect(screen.getAllByText("test@example.com").length).toBeGreaterThan(0);
    });
  });

  it("should show user ID when dropdown is opened", async () => {
    const user = userEvent.setup();
    renderWithProviders(<UserDropdown onLogout={mockOnLogout} />);

    await user.click(getAccountTrigger());

    await waitFor(() => {
      expect(screen.getByText("test-user-id")).toBeInTheDocument();
    });
  });

  it("should show user role when dropdown is opened", async () => {
    const user = userEvent.setup();
    renderWithProviders(<UserDropdown onLogout={mockOnLogout} />);

    await user.click(getAccountTrigger());

    await waitFor(() => {
      expect(screen.getAllByText("Admin").length).toBeGreaterThan(0);
    });
  });

  it("should display Standard badge for non-premium users", async () => {
    const user = userEvent.setup();
    renderWithProviders(<UserDropdown onLogout={mockOnLogout} />);

    await user.click(getAccountTrigger());

    await waitFor(() => {
      expect(screen.getByText("Standard")).toBeInTheDocument();
    });
  });

  it("should display Premium badge for premium users", async () => {
    const user = userEvent.setup();
    mockUseAuthorizedImpl = () => ({
      userId: "test-user-id",
      userEmail: "test@example.com",
      userRoleLabel: "Admin",
      premiumUser: true,
    });

    renderWithProviders(<UserDropdown onLogout={mockOnLogout} />);

    await user.click(getAccountTrigger());

    await waitFor(() => {
      expect(screen.getByText("Premium")).toBeInTheDocument();
    });
  });

  it("should call onLogout when logout is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(<UserDropdown onLogout={mockOnLogout} />);

    await user.click(getAccountTrigger());

    await waitFor(() => {
      expect(screen.getAllByText("test@example.com").length).toBeGreaterThan(0);
    });

    await user.click(screen.getByText("Logout"));

    expect(mockOnLogout).toHaveBeenCalledTimes(1);
  });

  it("should navigate to the change-password page for username/password sessions", async () => {
    mockUseAuthorizedImpl = () => ({
      userId: "test-user-id",
      userEmail: "test@example.com",
      userRoleLabel: "Admin",
      premiumUser: false,
      loginMethod: "username_password",
    });
    const user = userEvent.setup();
    renderWithProviders(<UserDropdown onLogout={mockOnLogout} />);

    await user.click(getAccountTrigger());

    await user.click(await screen.findByText("Change Password"));

    expect(mockRouterPush).toHaveBeenCalledWith(expect.stringContaining("change-password"));
  });

  it("should hide the change-password entry for SSO sessions", async () => {
    mockUseAuthorizedImpl = () => ({
      userId: "test-user-id",
      userEmail: "test@example.com",
      userRoleLabel: "Admin",
      premiumUser: false,
      loginMethod: "sso",
    });
    const user = userEvent.setup();
    renderWithProviders(<UserDropdown onLogout={mockOnLogout} />);

    await user.click(getAccountTrigger());

    await waitFor(() => {
      expect(screen.getAllByText("test@example.com").length).toBeGreaterThan(0);
    });

    expect(screen.queryByText("Change Password")).not.toBeInTheDocument();
  });

  it("should toggle hide new feature indicators switch", async () => {
    const user = userEvent.setup();
    renderWithProviders(<UserDropdown onLogout={mockOnLogout} />);

    await user.click(getAccountTrigger());

    await waitFor(() => {
      expect(screen.getAllByText("test@example.com").length).toBeGreaterThan(0);
    });

    const toggle = screen.getByLabelText("Toggle hide new feature indicators");
    expect(toggle).not.toBeChecked();

    await user.click(toggle);

    const localStorageUtils = vi.mocked(await import("@/utils/localStorageUtils"));
    expect(localStorageUtils.setLocalStorageItem).toHaveBeenCalledWith("disableShowNewBadge", "true");
    expect(localStorageUtils.emitLocalStorageChange).toHaveBeenCalledWith("disableShowNewBadge");
  });

  it("should toggle hide new feature indicators switch off", async () => {
    const user = userEvent.setup();
    mockGetLocalStorageItemImpl = (key: string): string | null => {
      if (key === "disableShowNewBadge") return "true";
      return null;
    };

    renderWithProviders(<UserDropdown onLogout={mockOnLogout} />);

    await user.click(getAccountTrigger());

    await waitFor(() => {
      expect(screen.getAllByText("test@example.com").length).toBeGreaterThan(0);
    });

    const toggle = screen.getByLabelText("Toggle hide new feature indicators");
    expect(toggle).toBeChecked();

    await user.click(toggle);

    const localStorageUtils = vi.mocked(await import("@/utils/localStorageUtils"));
    expect(localStorageUtils.removeLocalStorageItem).toHaveBeenCalledWith("disableShowNewBadge");
    expect(localStorageUtils.emitLocalStorageChange).toHaveBeenCalledWith("disableShowNewBadge");
  });

  it("should toggle hide all prompts switch", async () => {
    const user = userEvent.setup();
    renderWithProviders(<UserDropdown onLogout={mockOnLogout} />);

    await user.click(getAccountTrigger());

    await waitFor(() => {
      expect(screen.getAllByText("test@example.com").length).toBeGreaterThan(0);
    });

    const toggle = screen.getByLabelText("Toggle hide all prompts");
    expect(toggle).not.toBeChecked();

    await user.click(toggle);

    const localStorageUtils = vi.mocked(await import("@/utils/localStorageUtils"));
    expect(localStorageUtils.setLocalStorageItem).toHaveBeenCalledWith("disableShowPrompts", "true");
    expect(localStorageUtils.emitLocalStorageChange).toHaveBeenCalledWith("disableShowPrompts");
  });

  it("should toggle hide all prompts switch off", async () => {
    const user = userEvent.setup();
    mockUseDisableShowPromptsImpl = () => true;
    mockGetLocalStorageItemImpl = (key: string): string | null => {
      if (key === "disableShowPrompts") return "true";
      return null;
    };

    renderWithProviders(<UserDropdown onLogout={mockOnLogout} />);

    await user.click(getAccountTrigger());

    await waitFor(() => {
      expect(screen.getAllByText("test@example.com").length).toBeGreaterThan(0);
    });

    const toggle = screen.getByLabelText("Toggle hide all prompts");
    expect(toggle).toBeChecked();

    await user.click(toggle);

    const localStorageUtils = vi.mocked(await import("@/utils/localStorageUtils"));
    expect(localStorageUtils.removeLocalStorageItem).toHaveBeenCalledWith("disableShowPrompts");
    expect(localStorageUtils.emitLocalStorageChange).toHaveBeenCalledWith("disableShowPrompts");
  });

  it("should show Account in the trigger when user id is the default placeholder", () => {
    mockUseAuthorizedImpl = () => ({
      userId: "default_user_id",
      userEmail: null as any,
      userRoleLabel: "Admin",
      premiumUser: false,
    });
    renderWithProviders(<UserDropdown onLogout={mockOnLogout} />);
    expect(screen.getByText("Account")).toBeInTheDocument();
  });

  it("should display dash when user email is not available", async () => {
    const user = userEvent.setup();
    mockUseAuthorizedImpl = () => ({
      userId: "test-user-id",
      userEmail: null as any,
      userRoleLabel: "Admin",
      premiumUser: false,
    });

    renderWithProviders(<UserDropdown onLogout={mockOnLogout} />);

    await user.click(getAccountTrigger());

    await waitFor(() => {
      expect(screen.getByText("-")).toBeInTheDocument();
    });
  });

  it("should display dash when user ID is not available", async () => {
    const user = userEvent.setup();
    mockUseAuthorizedImpl = () => ({
      userId: null as any,
      userEmail: "test@example.com",
      userRoleLabel: "Admin",
      premiumUser: false,
    });

    renderWithProviders(<UserDropdown onLogout={mockOnLogout} />);

    await user.click(getAccountTrigger());

    await waitFor(() => {
      const dashElements = screen.getAllByText("-");
      expect(dashElements.length).toBeGreaterThan(0);
    });
  });

  it("should initialize hide new feature indicators from localStorage", async () => {
    const user = userEvent.setup();
    mockGetLocalStorageItemImpl = (key: string): string | null => {
      if (key === "disableShowNewBadge") return "true";
      return null;
    };

    renderWithProviders(<UserDropdown onLogout={mockOnLogout} />);

    await user.click(getAccountTrigger());

    await waitFor(() => {
      expect(screen.getAllByText("test@example.com").length).toBeGreaterThan(0);
    });

    const toggle = screen.getByLabelText("Toggle hide new feature indicators");
    expect(toggle).toBeChecked();
  });
});
