import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { CreateUserButton } from "./CreateUserButton";
import * as networking from "./networking";
import NotificationsManager from "./molecules/notifications_manager";

vi.mock("./networking", () => ({
  userCreateCall: vi.fn(),
  modelAvailableCall: vi.fn().mockResolvedValue({ data: [] }),
  invitationCreateCall: vi.fn(),
  getProxyUISettings: vi.fn().mockResolvedValue({
    PROXY_BASE_URL: null,
    PROXY_LOGOUT_URL: null,
    DEFAULT_TEAM_DISABLED: false,
    SSO_ENABLED: false,
  }),
  getProxyBaseUrl: vi.fn().mockReturnValue("http://localhost"),
}));

vi.mock("./bulk_create_users_button", () => ({
  default: () => <div data-testid="bulk-create-users">Bulk Create Users</div>,
}));

const mockUserCreateCall = vi.mocked(networking.userCreateCall);
const mockInvitationCreateCall = vi.mocked(networking.invitationCreateCall);
const mockGetProxyUISettings = vi.mocked(networking.getProxyUISettings);
const mockNotificationsManager = vi.mocked(NotificationsManager);

const createQueryClient = () =>
  new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 } },
  });

const defaultProps = {
  userID: "123",
  accessToken: "token",
  teams: [],
  possibleUIRoles: null as Record<string, Record<string, string>> | null,
};

function renderWithProviders(ui: React.ReactElement) {
  const qc = createQueryClient();
  return render(<QueryClientProvider client={qc}>{ui}</QueryClientProvider>);
}

describe("CreateUserButton", { timeout: 20000 }, () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockGetProxyUISettings.mockResolvedValue({
      PROXY_BASE_URL: null,
      PROXY_LOGOUT_URL: null,
      DEFAULT_TEAM_DISABLED: false,
      SSO_ENABLED: false,
    });
  });

  it("should render the create user form when embedded", () => {
    renderWithProviders(
      <CreateUserButton {...defaultProps} isEmbedded />,
    );
    expect(screen.getByRole("button", { name: /create user/i })).toBeInTheDocument();
  });

  it("should render the invite user button when not embedded", async () => {
    renderWithProviders(<CreateUserButton {...defaultProps} />);
    await waitFor(() => {
      expect(screen.getByRole("button", { name: /\+ invite user/i })).toBeInTheDocument();
    });
  });

  it("should open the invite modal when invite user button is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(<CreateUserButton {...defaultProps} />);
    await waitFor(() => {
      expect(screen.getByRole("button", { name: /\+ invite user/i })).toBeInTheDocument();
    });
    await user.click(screen.getByRole("button", { name: /\+ invite user/i }));
    const dialog = screen.getByRole("dialog", { name: /invite user/i });
    expect(dialog).toBeInTheDocument();
    expect(within(dialog).getByRole("button", { name: /invite user/i })).toBeInTheDocument();
  });

  it("should display email invitations info message in embedded mode", () => {
    renderWithProviders(<CreateUserButton {...defaultProps} isEmbedded />);
    expect(screen.getByText("Email invitations")).toBeInTheDocument();
  });

  it("should display user role options when possibleUIRoles is provided", async () => {
    const possibleUIRoles = {
      proxy_admin: { ui_label: "Admin", description: "Full access" },
      proxy_user: { ui_label: "User", description: "Limited access" },
    };
    renderWithProviders(
      <CreateUserButton {...defaultProps} possibleUIRoles={possibleUIRoles} isEmbedded />,
    );
    await userEvent.click(screen.getByRole("combobox", { name: /user role/i }));
    expect(screen.getByText("Admin")).toBeInTheDocument();
    expect(screen.getByText("User")).toBeInTheDocument();
  });

  it("should call userCreateCall when form is submitted in embedded mode", async () => {
    const user = userEvent.setup();
    mockUserCreateCall.mockResolvedValue({ data: { user_id: "new-user-123" } });
    mockInvitationCreateCall.mockResolvedValue({
      id: "inv-1",
      user_id: "new-user-123",
      has_user_setup_sso: false,
    } as any);

    renderWithProviders(
      <CreateUserButton {...defaultProps} possibleUIRoles={{ proxy_user: { ui_label: "User", description: "" } }} isEmbedded />,
    );

    await user.type(screen.getByLabelText(/user email/i), "test@example.com");
    await user.click(screen.getByRole("combobox", { name: /user role/i }));
    await user.click(screen.getByText("User"));
    await user.click(screen.getByRole("button", { name: /create user/i }));

    await waitFor(() => {
      expect(mockUserCreateCall).toHaveBeenCalledWith("token", null, expect.objectContaining({
        user_email: "test@example.com",
        user_role: "proxy_user",
      }));
    });
  });

  it("should call onUserCreated callback when user is created in embedded mode", async () => {
    const user = userEvent.setup();
    const onUserCreated = vi.fn();
    mockUserCreateCall.mockResolvedValue({ data: { user_id: "new-user-456" } });

    renderWithProviders(
      <CreateUserButton {...defaultProps} onUserCreated={onUserCreated} possibleUIRoles={{ proxy_user: { ui_label: "User", description: "" } }} isEmbedded />,
    );

    await user.type(screen.getByLabelText(/user email/i), "embedded@example.com");
    await user.click(screen.getByRole("combobox", { name: /user role/i }));
    await user.click(screen.getByText("User"));
    await user.click(screen.getByRole("button", { name: /create user/i }));

    await waitFor(() => {
      expect(onUserCreated).toHaveBeenCalledWith("new-user-456");
    });
  });

  it("should show success notification when user is created successfully in standalone mode", async () => {
    const user = userEvent.setup();
    mockUserCreateCall.mockResolvedValue({ data: { user_id: "new-user-789" } });
    mockInvitationCreateCall.mockResolvedValue({
      id: "inv-2",
      user_id: "new-user-789",
      has_user_setup_sso: false,
    } as any);

    renderWithProviders(
      <CreateUserButton {...defaultProps} possibleUIRoles={{ proxy_user: { ui_label: "User", description: "" } }} />,
    );

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /\+ invite user/i })).toBeInTheDocument();
    });
    await user.click(screen.getByRole("button", { name: /\+ invite user/i }));

    const dialog = screen.getByRole("dialog", { name: /invite user/i });
    await user.type(within(dialog).getByLabelText(/user email/i), "standalone@example.com");
    await user.click(within(dialog).getByRole("combobox", { name: /global proxy role/i }));
    await user.click(screen.getByText("User"));
    await user.click(within(dialog).getByRole("button", { name: /invite user/i }));

    await waitFor(() => {
      expect(mockNotificationsManager.success).toHaveBeenCalledWith("API user Created");
    });
  });

  it("should show error notification when user creation fails", async () => {
    const user = userEvent.setup();
    mockUserCreateCall.mockRejectedValue({ response: { data: { detail: "Email already exists" } } });

    renderWithProviders(
      <CreateUserButton {...defaultProps} possibleUIRoles={{ proxy_user: { ui_label: "User", description: "" } }} isEmbedded />,
    );

    await user.type(screen.getByLabelText(/user email/i), "duplicate@example.com");
    await user.click(screen.getByRole("combobox", { name: /user role/i }));
    await user.click(screen.getByText("User"));
    await user.click(screen.getByRole("button", { name: /create user/i }));

    await waitFor(() => {
      expect(mockNotificationsManager.fromBackend).toHaveBeenCalledWith("Email already exists");
    });
  });

  it("should show info notification when making API call", async () => {
    const user = userEvent.setup();
    mockUserCreateCall.mockResolvedValue({ data: { user_id: "new-user" } });
    mockInvitationCreateCall.mockResolvedValue({
      id: "inv-3",
      user_id: "new-user",
      has_user_setup_sso: false,
    } as any);

    renderWithProviders(
      <CreateUserButton {...defaultProps} possibleUIRoles={{ proxy_user: { ui_label: "User", description: "" } }} isEmbedded />,
    );

    await user.type(screen.getByLabelText(/user email/i), "info@example.com");
    await user.click(screen.getByRole("combobox", { name: /user role/i }));
    await user.click(screen.getByText("User"));
    await user.click(screen.getByRole("button", { name: /create user/i }));

    await waitFor(() => {
      expect(mockNotificationsManager.info).toHaveBeenCalledWith("Making API Call");
    });
  });

  it("should close modal when cancel is clicked in standalone mode", async () => {
    const user = userEvent.setup();
    renderWithProviders(<CreateUserButton {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /\+ invite user/i })).toBeInTheDocument();
    });
    await user.click(screen.getByRole("button", { name: /\+ invite user/i }));
    expect(screen.getByRole("dialog", { name: /invite user/i })).toBeInTheDocument();

    const dialog = screen.getByRole("dialog", { name: /invite user/i });
    await user.click(within(dialog).getByRole("button", { name: /close/i }));
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("should show onboarding modal when user is created and SSO is disabled", async () => {
    const user = userEvent.setup();
    mockUserCreateCall.mockResolvedValue({ data: { user_id: "sso-user" } });
    mockInvitationCreateCall.mockResolvedValue({
      id: "inv-sso",
      user_id: "sso-user",
      has_user_setup_sso: false,
    } as any);

    renderWithProviders(
      <CreateUserButton {...defaultProps} possibleUIRoles={{ proxy_user: { ui_label: "User", description: "" } }} />,
    );

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /\+ invite user/i })).toBeInTheDocument();
    });
    await user.click(screen.getByRole("button", { name: /\+ invite user/i }));

    const dialog = screen.getByRole("dialog", { name: /invite user/i });
    await user.type(within(dialog).getByLabelText(/user email/i), "sso@example.com");
    await user.click(within(dialog).getByRole("combobox", { name: /global proxy role/i }));
    await user.click(screen.getByText("User"));
    await user.click(within(dialog).getByRole("button", { name: /invite user/i }));

    await waitFor(() => {
      expect(mockInvitationCreateCall).toHaveBeenCalledWith("token", "sso-user");
    });
    await waitFor(() => {
      expect(mockNotificationsManager.success).toHaveBeenCalledWith("API user Created");
    });
  });
});
