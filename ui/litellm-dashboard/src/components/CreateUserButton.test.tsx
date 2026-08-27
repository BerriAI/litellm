import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent, { PointerEventsCheckLevel } from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { CreateUserButton } from "./CreateUserButton";
import * as networking from "./networking";
import { toast } from "@/lib/toast";

vi.mock("./networking", () => ({
  userCreateCall: vi.fn(),
  modelAvailableCall: vi.fn().mockResolvedValue({ data: [] }),
  invitationCreateCall: vi.fn(),
  organizationMemberAddCall: vi.fn(),
  getProxyUISettings: vi.fn().mockResolvedValue({
    PROXY_BASE_URL: null,
    PROXY_LOGOUT_URL: null,
    DEFAULT_TEAM_DISABLED: false,
    SSO_ENABLED: false,
  }),
  getProxyBaseUrl: vi.fn().mockReturnValue("http://localhost"),
}));

vi.mock("@/app/(dashboard)/hooks/organizations/useOrganizations", () => ({
  useOrganizations: vi.fn().mockReturnValue({ data: [], isLoading: false }),
}));

const mockUserCreateCall = vi.mocked(networking.userCreateCall);
const mockInvitationCreateCall = vi.mocked(networking.invitationCreateCall);
const mockGetProxyUISettings = vi.mocked(networking.getProxyUISettings);
const mockOrganizationMemberAddCall = vi.mocked(networking.organizationMemberAddCall);
const mockToast = vi.mocked(toast);

const createQueryClient = () =>
  new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 } },
  });

const defaultProps = {
  userID: "123",
  accessToken: "token",
  possibleUIRoles: null as Record<string, Record<string, string>> | null,
};

function renderWithProviders(ui: React.ReactElement) {
  const qc = createQueryClient();
  return render(<QueryClientProvider client={qc}>{ui}</QueryClientProvider>);
}

describe("CreateUserButton", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockGetProxyUISettings.mockResolvedValue({
      PROXY_BASE_URL: null,
      PROXY_LOGOUT_URL: null,
      DEFAULT_TEAM_DISABLED: false,
      SSO_ENABLED: false,
    });
  });

  describe("rendering and visibility", () => {
    it("should render the create user form when embedded", () => {
      renderWithProviders(<CreateUserButton {...defaultProps} isEmbedded />);
      expect(screen.getByRole("button", { name: /create user/i })).toBeInTheDocument();
    });

    it("should render the invite user button when not embedded", async () => {
      renderWithProviders(<CreateUserButton {...defaultProps} />);
      await waitFor(() => {
        expect(screen.getByRole("button", { name: /\+ invite user/i })).toBeInTheDocument();
      });
    });

    it("should not render the bulk invite button", async () => {
      renderWithProviders(<CreateUserButton {...defaultProps} />);
      await waitFor(() => {
        expect(screen.getByRole("button", { name: /\+ invite user/i })).toBeInTheDocument();
      });
      expect(screen.queryByRole("button", { name: /bulk invite users/i })).not.toBeInTheDocument();
    });

    it("should open the invite modal when invite user button is clicked", async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
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
      renderWithProviders(<CreateUserButton {...defaultProps} possibleUIRoles={possibleUIRoles} isEmbedded />);
      await userEvent.click(screen.getByRole("combobox", { name: /user role/i }));
      expect(screen.getByText("Admin")).toBeInTheDocument();
      expect(screen.getByText("User")).toBeInTheDocument();
    });

    it("should close modal when cancel is clicked in standalone mode", async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
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
  });

  describe("embedded mode submission", () => {
    it("should call userCreateCall when form is submitted in embedded mode", async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      mockUserCreateCall.mockResolvedValue({ data: { user_id: "new-user-123" } });
      mockInvitationCreateCall.mockResolvedValue({
        id: "inv-1",
        user_id: "new-user-123",
        has_user_setup_sso: false,
      } as any);

      renderWithProviders(
        <CreateUserButton
          {...defaultProps}
          possibleUIRoles={{ proxy_user: { ui_label: "User", description: "" } }}
          isEmbedded
        />,
      );

      await user.type(screen.getByLabelText(/user email/i), "test@example.com");
      await user.click(screen.getByRole("combobox", { name: /user role/i }));
      await user.click(screen.getByText("User"));
      await user.click(screen.getByRole("button", { name: /create user/i }));

      await waitFor(() => {
        expect(mockUserCreateCall).toHaveBeenCalledWith(
          "token",
          null,
          expect.objectContaining({
            user_email: "test@example.com",
            user_role: "proxy_user",
          }),
        );
      });
    });

    it("should call onUserCreated callback when user is created in embedded mode", async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      const onUserCreated = vi.fn();
      mockUserCreateCall.mockResolvedValue({ data: { user_id: "new-user-456" } });

      renderWithProviders(
        <CreateUserButton
          {...defaultProps}
          onUserCreated={onUserCreated}
          possibleUIRoles={{ proxy_user: { ui_label: "User", description: "" } }}
          isEmbedded
        />,
      );

      await user.type(screen.getByLabelText(/user email/i), "embedded@example.com");
      await user.click(screen.getByRole("combobox", { name: /user role/i }));
      await user.click(screen.getByText("User"));
      await user.click(screen.getByRole("button", { name: /create user/i }));

      await waitFor(() => {
        expect(onUserCreated).toHaveBeenCalledWith("new-user-456");
      });
    });

    it("should show error notification when user creation fails", async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      mockUserCreateCall.mockRejectedValue({ response: { data: { detail: "Email already exists" } } });

      renderWithProviders(
        <CreateUserButton
          {...defaultProps}
          possibleUIRoles={{ proxy_user: { ui_label: "User", description: "" } }}
          isEmbedded
        />,
      );

      await user.type(screen.getByLabelText(/user email/i), "duplicate@example.com");
      await user.click(screen.getByRole("combobox", { name: /user role/i }));
      await user.click(screen.getByText("User"));
      await user.click(screen.getByRole("button", { name: /create user/i }));

      await waitFor(() => {
        expect(mockToast.fromError).toHaveBeenCalledWith("Email already exists");
      });
    });

    it("should show info notification when making API call", async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      mockUserCreateCall.mockResolvedValue({ data: { user_id: "new-user" } });
      mockInvitationCreateCall.mockResolvedValue({
        id: "inv-3",
        user_id: "new-user",
        has_user_setup_sso: false,
      } as any);

      renderWithProviders(
        <CreateUserButton
          {...defaultProps}
          possibleUIRoles={{ proxy_user: { ui_label: "User", description: "" } }}
          isEmbedded
        />,
      );

      await user.type(screen.getByLabelText(/user email/i), "info@example.com");
      await user.click(screen.getByRole("combobox", { name: /user role/i }));
      await user.click(screen.getByText("User"));
      await user.click(screen.getByRole("button", { name: /create user/i }));

      await waitFor(() => {
        expect(mockToast.info).toHaveBeenCalledWith("Making API Call");
      });
    });
  });

  describe("standalone mode submission", () => {
    it("should show success notification when user is created successfully in standalone mode", async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
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
        expect(mockToast.success).toHaveBeenCalledWith("API user Created");
      });
    });

    it("should show onboarding modal when user is created and SSO is disabled", async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
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
        expect(mockToast.success).toHaveBeenCalledWith("API user Created");
      });
    });
  });

  describe("organizations", () => {
    it("should send organizations list in POST body when organizations are selected", async () => {
      const { useOrganizations } = await import("@/app/(dashboard)/hooks/organizations/useOrganizations");
      vi.mocked(useOrganizations).mockReturnValue({
        data: [{ organization_id: "org-1", organization_alias: "My Org" }],
        isLoading: false,
      } as any);

      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      mockUserCreateCall.mockResolvedValue({ data: { user_id: "org-user" } });
      mockInvitationCreateCall.mockResolvedValue({
        id: "inv-org",
        user_id: "org-user",
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
      await user.type(within(dialog).getByLabelText(/user email/i), "org@example.com");
      await user.click(within(dialog).getByRole("combobox", { name: /global proxy role/i }));
      await user.click(screen.getByText("User"));

      // Select org from the dropdown
      const orgSelect = within(dialog).getByRole("combobox", { name: /organization/i });
      await user.click(orgSelect);
      await user.click(screen.getByText("My Org (org-1)"));

      await user.click(within(dialog).getByRole("button", { name: /invite user/i }));

      await waitFor(() => {
        expect(mockUserCreateCall).toHaveBeenCalledWith(
          "token",
          null,
          expect.objectContaining({
            organizations: ["org-1"],
          }),
        );
      });
    });

    it("should not call organizationMemberAddCall after user creation", async () => {
      const { useOrganizations } = await import("@/app/(dashboard)/hooks/organizations/useOrganizations");
      vi.mocked(useOrganizations).mockReturnValue({
        data: [{ organization_id: "org-1", organization_alias: "My Org" }],
        isLoading: false,
      } as any);

      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      mockUserCreateCall.mockResolvedValue({ data: { user_id: "no-member-add-user" } });
      mockInvitationCreateCall.mockResolvedValue({
        id: "inv-nma",
        user_id: "no-member-add-user",
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
      await user.type(within(dialog).getByLabelText(/user email/i), "nomemberadd@example.com");
      await user.click(within(dialog).getByRole("combobox", { name: /global proxy role/i }));
      await user.click(screen.getByText("User"));
      await user.click(within(dialog).getByRole("button", { name: /invite user/i }));

      await waitFor(() => {
        expect(mockUserCreateCall).toHaveBeenCalled();
      });
      expect(mockOrganizationMemberAddCall).not.toHaveBeenCalled();
    });
  });

  describe("send invitation email toggle", () => {
    it("should send send_invite_email true by default in embedded mode", async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      mockUserCreateCall.mockResolvedValue({ data: { user_id: "default-on-user" } });

      renderWithProviders(
        <CreateUserButton
          {...defaultProps}
          possibleUIRoles={{ proxy_user: { ui_label: "User", description: "" } }}
          isEmbedded
        />,
      );

      await user.type(screen.getByLabelText(/user email/i), "default@example.com");
      await user.click(screen.getByRole("combobox", { name: /user role/i }));
      await user.click(screen.getByText("User"));
      await user.click(screen.getByRole("button", { name: /create user/i }));

      await waitFor(() => {
        expect(mockUserCreateCall).toHaveBeenCalledWith(
          "token",
          null,
          expect.objectContaining({
            send_invite_email: true,
          }),
        );
      });
    });

    it("should send send_invite_email false when the checkbox is unchecked in embedded mode", async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      mockUserCreateCall.mockResolvedValue({ data: { user_id: "unchecked-user" } });

      renderWithProviders(
        <CreateUserButton
          {...defaultProps}
          possibleUIRoles={{ proxy_user: { ui_label: "User", description: "" } }}
          isEmbedded
        />,
      );

      await user.type(screen.getByLabelText(/user email/i), "off@example.com");
      await user.click(screen.getByRole("combobox", { name: /user role/i }));
      await user.click(screen.getByText("User"));
      await user.click(screen.getByRole("checkbox", { name: /send invitation email/i }));
      await user.click(screen.getByRole("button", { name: /create user/i }));

      await waitFor(() => {
        expect(mockUserCreateCall).toHaveBeenCalledWith(
          "token",
          null,
          expect.objectContaining({
            send_invite_email: false,
          }),
        );
      });
    });

    it("should send send_invite_email true by default in standalone mode", async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      mockUserCreateCall.mockResolvedValue({ data: { user_id: "standalone-default-user" } });
      mockInvitationCreateCall.mockResolvedValue({
        id: "inv-default",
        user_id: "standalone-default-user",
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
      await user.type(within(dialog).getByLabelText(/user email/i), "standalone-default@example.com");
      await user.click(within(dialog).getByRole("combobox", { name: /global proxy role/i }));
      await user.click(screen.getByText("User"));
      await user.click(within(dialog).getByRole("button", { name: /invite user/i }));

      await waitFor(() => {
        expect(mockUserCreateCall).toHaveBeenCalledWith(
          "token",
          null,
          expect.objectContaining({
            send_invite_email: true,
          }),
        );
      });
    });

    it("should send send_invite_email false when the checkbox is unchecked in standalone mode", async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      mockUserCreateCall.mockResolvedValue({ data: { user_id: "standalone-off-user" } });
      mockInvitationCreateCall.mockResolvedValue({
        id: "inv-off",
        user_id: "standalone-off-user",
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
      await user.type(within(dialog).getByLabelText(/user email/i), "standalone-off@example.com");
      await user.click(within(dialog).getByRole("combobox", { name: /global proxy role/i }));
      await user.click(screen.getByText("User"));
      await user.click(within(dialog).getByRole("checkbox", { name: /send invitation email/i }));
      await user.click(within(dialog).getByRole("button", { name: /invite user/i }));

      await waitFor(() => {
        expect(mockUserCreateCall).toHaveBeenCalledWith(
          "token",
          null,
          expect.objectContaining({
            send_invite_email: false,
          }),
        );
      });
    });

    it("should keep the checkbox checked by default when the modal is opened in standalone mode", async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });

      renderWithProviders(
        <CreateUserButton {...defaultProps} possibleUIRoles={{ proxy_user: { ui_label: "User", description: "" } }} />,
      );

      await waitFor(() => {
        expect(screen.getByRole("button", { name: /\+ invite user/i })).toBeInTheDocument();
      });
      await user.click(screen.getByRole("button", { name: /\+ invite user/i }));

      const dialog = screen.getByRole("dialog", { name: /invite user/i });
      expect(within(dialog).getByRole("checkbox", { name: /send invitation email/i })).toBeChecked();
    });
  });
  describe("submit payload parity", () => {
    const ROLES = {
      proxy_user: { ui_label: "User", description: "" },
      proxy_admin: { ui_label: "Admin", description: "" },
    };

    const openStandaloneModal = async (user: ReturnType<typeof userEvent.setup>) => {
      expect(await screen.findByRole("button", { name: /\+ invite user/i })).toBeInTheDocument();
      await user.click(screen.getByRole("button", { name: /\+ invite user/i }));
      return screen.getByRole("dialog", { name: /invite user/i });
    };

    const submittedPayload = () => mockUserCreateCall.mock.calls[0][2];

    it("should send exactly seven keys from the standalone modal, with the untouched ones undefined", async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      mockUserCreateCall.mockResolvedValue({ data: { user_id: "u1" } });
      mockInvitationCreateCall.mockResolvedValue({ id: "i1", user_id: "u1", has_user_setup_sso: false } as any);

      renderWithProviders(<CreateUserButton {...defaultProps} possibleUIRoles={ROLES} />);
      const dialog = await openStandaloneModal(user);

      await user.type(within(dialog).getByLabelText(/user email/i), "parity@example.com");
      await user.click(within(dialog).getByRole("combobox", { name: /global proxy role/i }));
      await user.click(screen.getByText("User"));
      await user.click(within(dialog).getByRole("button", { name: /invite user/i }));

      await waitFor(() => {
        expect(mockUserCreateCall).toHaveBeenCalled();
      });
      expect(Object.keys(submittedPayload()).sort()).toEqual([
        "metadata",
        "models",
        "organization_ids",
        "send_invite_email",
        "team_id",
        "user_email",
        "user_role",
      ]);
      expect(submittedPayload()).toStrictEqual({
        user_email: "parity@example.com",
        user_role: "proxy_user",
        team_id: undefined,
        organization_ids: undefined,
        metadata: undefined,
        send_invite_email: true,
        models: ["no-default-models"],
      });
    });

    it("should send exactly six keys from the embedded form, with no organization_ids", async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      mockUserCreateCall.mockResolvedValue({ data: { user_id: "u2" } });

      renderWithProviders(<CreateUserButton {...defaultProps} possibleUIRoles={ROLES} isEmbedded />);

      await user.type(screen.getByLabelText(/user email/i), "embedded-parity@example.com");
      await user.click(screen.getByRole("combobox", { name: /user role/i }));
      await user.click(screen.getByText("User"));
      await user.click(screen.getByRole("button", { name: /create user/i }));

      await waitFor(() => {
        expect(mockUserCreateCall).toHaveBeenCalled();
      });
      expect(Object.keys(submittedPayload()).sort()).toEqual([
        "metadata",
        "models",
        "send_invite_email",
        "team_id",
        "user_email",
        "user_role",
      ]);
      expect(submittedPayload()).toStrictEqual({
        user_email: "embedded-parity@example.com",
        user_role: "proxy_user",
        team_id: undefined,
        metadata: undefined,
        send_invite_email: true,
        models: ["no-default-models"],
      });
    });

    it("should not default models to no-default-models for a proxy admin", async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      mockUserCreateCall.mockResolvedValue({ data: { user_id: "u3" } });

      renderWithProviders(<CreateUserButton {...defaultProps} possibleUIRoles={ROLES} isEmbedded />);

      await user.type(screen.getByLabelText(/user email/i), "admin-parity@example.com");
      await user.click(screen.getByRole("combobox", { name: /user role/i }));
      await user.click(screen.getByText("Admin"));
      await user.click(screen.getByRole("button", { name: /create user/i }));

      await waitFor(() => {
        expect(mockUserCreateCall).toHaveBeenCalled();
      });
      expect(submittedPayload()).not.toHaveProperty("models");
      expect(Object.keys(submittedPayload()).sort()).toEqual([
        "metadata",
        "send_invite_email",
        "team_id",
        "user_email",
        "user_role",
      ]);
    });

    it("should rename organization_ids to organizations and drop the original key", async () => {
      const { useOrganizations } = await import("@/app/(dashboard)/hooks/organizations/useOrganizations");
      vi.mocked(useOrganizations).mockReturnValue({
        data: [{ organization_id: "org-1", organization_alias: "My Org" }],
        isLoading: false,
      } as any);

      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      mockUserCreateCall.mockResolvedValue({ data: { user_id: "u4" } });
      mockInvitationCreateCall.mockResolvedValue({ id: "i4", user_id: "u4", has_user_setup_sso: false } as any);

      renderWithProviders(<CreateUserButton {...defaultProps} possibleUIRoles={ROLES} />);
      const dialog = await openStandaloneModal(user);

      await user.type(within(dialog).getByLabelText(/user email/i), "org-parity@example.com");
      await user.click(within(dialog).getByRole("combobox", { name: /global proxy role/i }));
      await user.click(screen.getByText("User"));
      await user.click(within(dialog).getByRole("combobox", { name: /organization/i }));
      await user.click(screen.getByText("My Org (org-1)"));
      await user.click(within(dialog).getByRole("button", { name: /invite user/i }));

      await waitFor(() => {
        expect(mockUserCreateCall).toHaveBeenCalled();
      });
      expect(submittedPayload()).not.toHaveProperty("organization_ids");
      expect(submittedPayload().organizations).toEqual(["org-1"]);
      expect(Object.keys(submittedPayload()).sort()).toEqual([
        "metadata",
        "models",
        "organizations",
        "send_invite_email",
        "team_id",
        "user_email",
        "user_role",
      ]);
    });

    it("should send metadata as the raw string the user typed", async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      mockUserCreateCall.mockResolvedValue({ data: { user_id: "u5" } });

      renderWithProviders(<CreateUserButton {...defaultProps} possibleUIRoles={ROLES} isEmbedded />);

      await user.type(screen.getByLabelText(/user email/i), "meta-parity@example.com");
      await user.click(screen.getByRole("combobox", { name: /user role/i }));
      await user.click(screen.getByText("User"));
      await user.type(screen.getByLabelText(/metadata/i), '{{"a":1}');
      await user.click(screen.getByRole("button", { name: /create user/i }));

      await waitFor(() => {
        expect(mockUserCreateCall).toHaveBeenCalled();
      });
      expect(submittedPayload().metadata).toBe('{"a":1}');
    });
    it("should leave models out entirely for a proxy admin created from the standalone modal", async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      mockUserCreateCall.mockResolvedValue({ data: { user_id: "u6" } });
      mockInvitationCreateCall.mockResolvedValue({ id: "i6", user_id: "u6", has_user_setup_sso: false } as any);

      renderWithProviders(<CreateUserButton {...defaultProps} possibleUIRoles={ROLES} />);
      const dialog = await openStandaloneModal(user);

      await user.type(within(dialog).getByLabelText(/user email/i), "standalone-admin@example.com");
      await user.click(within(dialog).getByRole("combobox", { name: /global proxy role/i }));
      await user.click(screen.getByText("Admin"));
      await user.click(within(dialog).getByRole("button", { name: /invite user/i }));

      await waitFor(() => {
        expect(mockUserCreateCall).toHaveBeenCalled();
      });
      expect(Object.keys(submittedPayload()).sort()).toEqual([
        "metadata",
        "organization_ids",
        "send_invite_email",
        "team_id",
        "user_email",
        "user_role",
      ]);
    });

    it("should discard models picked in the personal key section once it is collapsed again", async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      mockUserCreateCall.mockResolvedValue({ data: { u: "u7" } });
      mockInvitationCreateCall.mockResolvedValue({ id: "i7", user_id: "u7", has_user_setup_sso: false } as any);

      renderWithProviders(<CreateUserButton {...defaultProps} possibleUIRoles={ROLES} />);
      const dialog = await openStandaloneModal(user);

      await user.type(within(dialog).getByLabelText(/user email/i), "collapsed@example.com");
      await user.click(within(dialog).getByRole("combobox", { name: /global proxy role/i }));
      await user.click(screen.getByText("User"));

      await user.click(within(dialog).getByText("Personal Key Creation"));
      const modelsSelect = within(dialog).getByRole("combobox", { name: /select models/i });
      await user.click(modelsSelect);
      await user.click(await screen.findByText("All Proxy Models"));
      await user.click(within(dialog).getByText("Personal Key Creation"));

      await user.click(within(dialog).getByRole("button", { name: /invite user/i }));

      await waitFor(() => {
        expect(mockUserCreateCall).toHaveBeenCalled();
      });
      expect(submittedPayload().models).toEqual(["no-default-models"]);
    });
  });
});
