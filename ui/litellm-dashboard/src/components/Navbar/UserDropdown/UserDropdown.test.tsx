import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders, screen, waitFor } from "../../../../tests/test-utils";
import UserDropdown from "./UserDropdown";

let mockUseAuthorizedImpl = () => ({
  userId: "test-user-id",
  userEmail: "test@example.com",
  userRole: "Admin",
  premiumUser: false,
});

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

  beforeEach(() => {
    vi.clearAllMocks();
    mockUseAuthorizedImpl = () => ({
      userId: "test-user-id",
      userEmail: "test@example.com",
      userRole: "Admin",
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
    expect(screen.getByRole("button")).toBeInTheDocument();
  });

  it("should display user button with User text", () => {
    renderWithProviders(<UserDropdown onLogout={mockOnLogout} />);
    expect(screen.getByText("User")).toBeInTheDocument();
  });

  it("should show user email when dropdown is opened", async () => {
    const user = userEvent.setup();
    renderWithProviders(<UserDropdown onLogout={mockOnLogout} />);

    await user.click(screen.getByText("User"));

    await waitFor(() => {
      expect(screen.getByText("test@example.com")).toBeInTheDocument();
    });
  });

  it("should show user ID when dropdown is opened", async () => {
    const user = userEvent.setup();
    renderWithProviders(<UserDropdown onLogout={mockOnLogout} />);

    await user.click(screen.getByText("User"));

    await waitFor(() => {
      expect(screen.getByText("test-user-id")).toBeInTheDocument();
    });
  });

  it("should show user role when dropdown is opened", async () => {
    const user = userEvent.setup();
    renderWithProviders(<UserDropdown onLogout={mockOnLogout} />);

    await user.click(screen.getByText("User"));

    await waitFor(() => {
      expect(screen.getByText("Admin")).toBeInTheDocument();
    });
  });

  it("should display Standard badge for non-premium users", async () => {
    const user = userEvent.setup();
    renderWithProviders(<UserDropdown onLogout={mockOnLogout} />);

    await user.click(screen.getByText("User"));

    await waitFor(() => {
      expect(screen.getByText("Standard")).toBeInTheDocument();
    });
  });

  it("should display Premium badge for premium users", async () => {
    const user = userEvent.setup();
    mockUseAuthorizedImpl = () => ({
      userId: "test-user-id",
      userEmail: "test@example.com",
      userRole: "Admin",
      premiumUser: true,
    });

    renderWithProviders(<UserDropdown onLogout={mockOnLogout} />);

    await user.click(screen.getByText("User"));

    await waitFor(() => {
      expect(screen.getByText("Premium")).toBeInTheDocument();
    });
  });

  it("should call onLogout when logout is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(<UserDropdown onLogout={mockOnLogout} />);

    await user.click(screen.getByText("User"));

    await waitFor(() => {
      expect(screen.getByText("test@example.com")).toBeInTheDocument();
    });

    await user.click(screen.getByText("Logout"));

    expect(mockOnLogout).toHaveBeenCalledTimes(1);
  });

  it("should toggle hide new feature indicators switch", async () => {
    const user = userEvent.setup();
    renderWithProviders(<UserDropdown onLogout={mockOnLogout} />);

    await user.click(screen.getByText("User"));

    await waitFor(() => {
      expect(screen.getByText("test@example.com")).toBeInTheDocument();
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

    await user.click(screen.getByText("User"));

    await waitFor(() => {
      expect(screen.getByText("test@example.com")).toBeInTheDocument();
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

    await user.click(screen.getByText("User"));

    await waitFor(() => {
      expect(screen.getByText("test@example.com")).toBeInTheDocument();
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

    await user.click(screen.getByText("User"));

    await waitFor(() => {
      expect(screen.getByText("test@example.com")).toBeInTheDocument();
    });

    const toggle = screen.getByLabelText("Toggle hide all prompts");
    expect(toggle).toBeChecked();

    await user.click(toggle);

    const localStorageUtils = vi.mocked(await import("@/utils/localStorageUtils"));
    expect(localStorageUtils.removeLocalStorageItem).toHaveBeenCalledWith("disableShowPrompts");
    expect(localStorageUtils.emitLocalStorageChange).toHaveBeenCalledWith("disableShowPrompts");
  });

  it("should display dash when user email is not available", async () => {
    const user = userEvent.setup();
    mockUseAuthorizedImpl = () => ({
      userId: "test-user-id",
      userEmail: null as any,
      userRole: "Admin",
      premiumUser: false,
    });

    renderWithProviders(<UserDropdown onLogout={mockOnLogout} />);

    await user.click(screen.getByText("User"));

    await waitFor(() => {
      expect(screen.getByText("-")).toBeInTheDocument();
    });
  });

  it("should display dash when user ID is not available", async () => {
    const user = userEvent.setup();
    mockUseAuthorizedImpl = () => ({
      userId: null as any,
      userEmail: "test@example.com",
      userRole: "Admin",
      premiumUser: false,
    });

    renderWithProviders(<UserDropdown onLogout={mockOnLogout} />);

    await user.click(screen.getByText("User"));

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

    await user.click(screen.getByText("User"));

    await waitFor(() => {
      expect(screen.getByText("test@example.com")).toBeInTheDocument();
    });

    const toggle = screen.getByLabelText("Toggle hide new feature indicators");
    expect(toggle).toBeChecked();
  });
});
