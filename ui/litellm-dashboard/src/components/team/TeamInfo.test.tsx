import { useTeamMetadataSchema } from "@/app/(dashboard)/hooks/teams/useTeamMetadataSchema";
import * as networking from "@/components/networking";
import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { chooseSelectOption, renderWithProviders, testQueryClient } from "../../../tests/test-utils";
import TeamInfoView, { type TeamData } from "./TeamInfo";

const authState = vi.hoisted(() => ({ userRole: "Admin" }));

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => ({
    token: "123",
    accessToken: "123",
    userId: "user-1",
    userEmail: "user@example.com",
    userRole: authState.userRole,
    premiumUser: false,
    disabledPersonalKeyCreation: null,
    showSSOBanner: false,
  }),
}));

vi.mock("@/components/networking", () => ({
  teamInfoCall: vi.fn(),
  teamMemberDeleteCall: vi.fn(),
  teamMemberAddCall: vi.fn(),
  teamMemberUpdateCall: vi.fn(),
  teamUpdateCall: vi.fn(),
  getGuardrailsList: vi.fn(),
  getPoliciesList: vi.fn(),
  getPolicyInfoWithGuardrails: vi.fn(),
  fetchMCPAccessGroups: vi.fn(),
  getTeamPermissionsCall: vi.fn(),
  organizationInfoCall: vi.fn(),
  getRouterSettingsCall: vi.fn().mockResolvedValue({ fields: [] }),
  getPassThroughEndpointsCall: vi.fn(),
}));

const can = vi.fn();
vi.mock("@/app/(dashboard)/hooks/useCan", () => ({
  default: (...args: unknown[]) => can(...args),
}));

vi.mock("@/components/utils/dataUtils", () => ({
  copyToClipboard: vi.fn().mockResolvedValue(true),
  formatNumberWithCommas: vi.fn((value: number) => value.toLocaleString()),
}));

vi.mock("@/app/(dashboard)/hooks/teams/useTeamMetadataSchema", () => ({
  useTeamMetadataSchema: vi.fn(() => ({ data: [], isLoading: false })),
}));

vi.mock("@/app/(dashboard)/hooks/models/useModels", () => ({
  useAllProxyModels: vi.fn(),
}));

vi.mock("@/app/(dashboard)/hooks/teams/useTeams", () => ({
  useTeam: vi.fn(),
}));

vi.mock("@/app/(dashboard)/hooks/organizations/useOrganizations", () => ({
  useOrganization: vi.fn(),
  useOrganizations: vi.fn().mockReturnValue({ data: [], isLoading: false }),
}));

vi.mock("@/app/(dashboard)/hooks/users/useCurrentUser", () => ({
  useCurrentUser: vi.fn(),
}));

vi.mock("@/components/team/TeamMemberTab", () => ({
  default: vi.fn(({ setIsAddMemberModalVisible }) => (
    <div>
      <button onClick={() => setIsAddMemberModalVisible(true)}>Add Member</button>
    </div>
  )),
}));

vi.mock("@/components/common_components/user_search_modal", () => ({
  default: vi.fn(({ isVisible, onCancel, onSubmit }) =>
    isVisible ? (
      <div>
        <button onClick={onCancel}>Cancel</button>
        <button onClick={() => onSubmit({ user_email: "new@test.com", user_id: "new-user", role: "user" })}>
          Submit
        </button>
      </div>
    ) : null,
  ),
}));

vi.mock("@/components/team/EditMembership", () => ({
  default: vi.fn(({ visible, onCancel, onSubmit }) =>
    visible ? (
      <div>
        <button onClick={onCancel}>Cancel</button>
        <button onClick={() => onSubmit({ user_email: "edit@test.com", user_id: "edit-user", role: "admin" })}>
          Submit
        </button>
      </div>
    ) : null,
  ),
}));

vi.mock("@/components/common_components/DeleteResourceModal", () => ({
  default: vi.fn(({ isOpen, onCancel, onOk }) =>
    isOpen ? (
      <div>
        <button onClick={onCancel}>Cancel</button>
        <button onClick={onOk}>Confirm Delete</button>
      </div>
    ) : null,
  ),
}));

vi.mock("@/components/team/member_permissions", () => ({
  default: vi.fn(() => <div>Member Permissions</div>),
}));

vi.mock("@/components/common_components/ModelAliasManager", () => ({
  default: vi.fn(({ initialModelAliases, onAliasUpdate }) => (
    <div>
      <div data-testid="alias-editor-initial">{JSON.stringify(initialModelAliases)}</div>
      <button onClick={() => onAliasUpdate({ "gpt-4o": "gpt-4" })}>Set Alias</button>
      <button type="button" onClick={() => onAliasUpdate({})}>
        Clear Aliases
      </button>
    </div>
  )),
}));

vi.mock("@/app/(dashboard)/hooks/accessGroups/useAccessGroups", () => ({
  useAccessGroups: vi.fn().mockReturnValue({
    data: [
      { access_group_id: "ag-1", access_group_name: "Group 1" },
      { access_group_id: "ag-2", access_group_name: "Group 2" },
    ],
    isLoading: false,
    isError: false,
  }),
}));

vi.mock("@/app/(dashboard)/hooks/keys/useKeys", () => ({
  useKeys: vi.fn().mockReturnValue({
    data: { keys: [], total_count: 0, current_page: 1, total_pages: 1 },
    isPending: false,
    isFetching: false,
    refetch: vi.fn(),
  }),
}));

vi.mock("../key_team_helpers/filter_helpers", () => ({
  fetchTeamFilterOptions: vi.fn().mockResolvedValue({
    keyAliases: [],
    organizationIds: [],
    userIds: [],
  }),
  fetchAllKeyAliases: vi.fn().mockResolvedValue([]),
  fetchAllOrganizations: vi.fn().mockResolvedValue([]),
}));

import { useAllProxyModels } from "@/app/(dashboard)/hooks/models/useModels";
import { useKeys } from "@/app/(dashboard)/hooks/keys/useKeys";
import { useOrganization } from "@/app/(dashboard)/hooks/organizations/useOrganizations";
import { useTeam } from "@/app/(dashboard)/hooks/teams/useTeams";
import { useCurrentUser } from "@/app/(dashboard)/hooks/users/useCurrentUser";

const mockUseAllProxyModels = vi.mocked(useAllProxyModels);
const mockUseKeys = vi.mocked(useKeys);
const mockUseTeam = vi.mocked(useTeam);
const mockUseOrganization = vi.mocked(useOrganization);
const mockUseCurrentUser = vi.mocked(useCurrentUser);

const createMockTeamData = (overrides = {}) => ({
  team_id: "123",
  team_info: {
    team_alias: "Test Team",
    team_id: "123",
    organization_id: null,
    admins: ["admin@test.com"],
    members: ["user1@test.com"],
    members_with_roles: [
      {
        user_id: "user1@test.com",
        user_email: "user1@test.com",
        role: "member",
        spend: 0,
        budget_id: "budget1",
      },
    ],
    metadata: {},
    tpm_limit: null,
    rpm_limit: null,
    max_budget: null,
    budget_duration: null,
    models: [],
    blocked: false,
    spend: 0,
    max_parallel_requests: null,
    budget_reset_at: null,
    model_id: null,
    litellm_model_table: null,
    created_at: "2024-01-01T00:00:00Z",
    team_member_budget_table: null,
    guardrails: [],
    policies: [],
    object_permission: null,
    ...overrides,
  },
  keys: [],
  team_memberships: [],
});

const seedDefaultMocks = () => {
  mockUseAllProxyModels.mockReturnValue({
    data: { data: [] },
    isLoading: false,
  } as any);
  mockUseTeam.mockReturnValue({
    data: undefined,
    isLoading: false,
  } as any);
  mockUseOrganization.mockReturnValue({
    data: undefined,
    isLoading: false,
  } as any);
  mockUseCurrentUser.mockReturnValue({
    data: { models: [] },
    isLoading: false,
  } as any);
  mockUseKeys.mockReturnValue({
    data: { keys: [], total_count: 0, current_page: 1, total_pages: 1 },
    isPending: false,
    isFetching: false,
    refetch: vi.fn(),
  } as any);
  vi.mocked(useTeamMetadataSchema).mockReturnValue({ data: [], isLoading: false } as any);

  can.mockReturnValue(true);
  vi.mocked(networking.getGuardrailsList).mockResolvedValue({ guardrails: [] });
  vi.mocked(networking.getPoliciesList).mockResolvedValue({ policies: [] });
  vi.mocked(networking.fetchMCPAccessGroups).mockResolvedValue([]);
  vi.mocked(networking.getTeamPermissionsCall).mockResolvedValue({
    all_available_permissions: [],
    team_member_permissions: [],
  });
};

describe("TeamInfoView", () => {
  const defaultProps = {
    teamId: "123",
    onUpdate: vi.fn(),
    onClose: vi.fn(),
    accessToken: "test-token",
    is_team_admin: true,
    is_proxy_admin: true,
    userModels: ["gpt-4", "gpt-3.5-turbo"],
    editTeam: false,
    premiumUser: false,
  };

  beforeEach(seedDefaultMocks);

  afterEach(() => {
    vi.clearAllMocks();
    authState.userRole = "Admin";
  });

  describe("display and rendering", () => {
    it("should render", async () => {
      vi.mocked(networking.teamInfoCall).mockResolvedValue(createMockTeamData());

      renderWithProviders(<TeamInfoView {...defaultProps} />);

      await waitFor(() => {
        const teamNameElements = screen.queryAllByText("Test Team");
        expect(teamNameElements.length).toBeGreaterThan(0);
      });
    });

    it("should display loading state while fetching team data", () => {
      vi.mocked(networking.teamInfoCall).mockImplementation(() => new Promise(() => {}));

      renderWithProviders(<TeamInfoView {...defaultProps} />);

      expect(screen.getByText("Loading...")).toBeInTheDocument();
    });

    it("should display error message when team is not found", async () => {
      vi.mocked(networking.teamInfoCall).mockResolvedValue({
        team_id: "123",
        team_info: null as any,
        keys: [],
        team_memberships: [],
      });

      renderWithProviders(<TeamInfoView {...defaultProps} />);

      await waitFor(() => {
        expect(screen.getByText("Team not found")).toBeInTheDocument();
      });
    });

    it("should display budget information in overview", async () => {
      vi.mocked(networking.teamInfoCall).mockResolvedValue(
        createMockTeamData({
          max_budget: 1000,
          spend: 250.5,
          budget_duration: "30d",
        }),
      );

      renderWithProviders(<TeamInfoView {...defaultProps} />);

      await waitFor(() => {
        expect(screen.getByText("Budget Status")).toBeInTheDocument();
      });
      expect(screen.getByText("$250.50")).toBeInTheDocument();
      expect(screen.getByText(/of \$1,000\.00/)).toBeInTheDocument();
    });

    it("renders a tpm/rpm/budget limit of 0 as 0 in the overview and settings tabs, never as Unlimited or No Limit", async () => {
      vi.mocked(networking.teamInfoCall).mockResolvedValue(
        createMockTeamData({
          tpm_limit: 0,
          rpm_limit: 0,
          team_member_budget_table: { max_budget: 0, budget_duration: null, tpm_limit: 0, rpm_limit: 0 },
        }),
      );

      renderWithProviders(<TeamInfoView {...defaultProps} />);

      const overview = await screen.findByRole("tabpanel", { name: "Overview" });
      expect(within(overview).getByText("TPM: 0")).toBeInTheDocument();
      expect(within(overview).getByText("RPM: 0")).toBeInTheDocument();

      await userEvent.setup({ delay: null }).click(screen.getByRole("tab", { name: "Settings" }));
      const settings = await screen.findByRole("tabpanel", { name: "Settings" });
      expect(within(settings).getByText("TPM: 0")).toBeInTheDocument();
      expect(within(settings).getByText("RPM: 0")).toBeInTheDocument();
      expect(within(settings).getByText("TPM Limit: 0")).toBeInTheDocument();
      expect(within(settings).getByText("RPM Limit: 0")).toBeInTheDocument();
      expect(within(settings).getByText("Max Budget: 0")).toBeInTheDocument();
      expect(screen.queryByText("TPM: Unlimited")).not.toBeInTheDocument();
      expect(screen.queryByText("RPM: Unlimited")).not.toBeInTheDocument();
      expect(screen.queryByText("TPM Limit: No Limit")).not.toBeInTheDocument();
      expect(screen.queryByText("RPM Limit: No Limit")).not.toBeInTheDocument();
    });

    it("should display guardrails in overview when present", async () => {
      vi.mocked(networking.teamInfoCall).mockResolvedValue(
        createMockTeamData({
          metadata: { guardrails: ["guardrail1", "guardrail2"] },
        }),
      );

      renderWithProviders(<TeamInfoView {...defaultProps} />);

      await waitFor(() => {
        expect(screen.getByText("Guardrails Settings")).toBeInTheDocument();
      });
      expect(screen.getByText("guardrail1")).toBeInTheDocument();
      expect(screen.getByText("guardrail2")).toBeInTheDocument();
    });

    it("should display policies in overview when present", async () => {
      vi.mocked(networking.teamInfoCall).mockResolvedValue(
        createMockTeamData({
          policies: ["policy1"],
        }),
      );
      vi.mocked(networking.getPolicyInfoWithGuardrails).mockResolvedValue({
        resolved_guardrails: ["guardrail1"],
      });

      renderWithProviders(<TeamInfoView {...defaultProps} />);

      await waitFor(() => {
        expect(screen.getByText("Policies")).toBeInTheDocument();
      });
    });

    it("should display team member budget information when present", async () => {
      vi.mocked(networking.teamInfoCall).mockResolvedValue(
        createMockTeamData({
          team_member_budget_table: {
            max_budget: 500,
            budget_duration: "30d",
            tpm_limit: 5000,
            rpm_limit: 50,
          },
        }),
      );

      renderWithProviders(<TeamInfoView {...defaultProps} />);

      await waitFor(() => {
        expect(screen.getByText("Budget Status")).toBeInTheDocument();
      });
      expect(screen.getByText("Team Member Budget: $500.00")).toBeInTheDocument();
    });

    it("should display virtual keys information", async () => {
      vi.mocked(networking.teamInfoCall).mockResolvedValue({
        ...createMockTeamData(),
        keys: [{ user_id: "user1", token: "key1" }, { token: "key2" }],
      });

      renderWithProviders(<TeamInfoView {...defaultProps} />);

      await waitFor(() => {
        expect(screen.getByRole("tab", { name: "Virtual Keys" })).toBeInTheDocument();
      });
    });

    it("should display object permissions when present", async () => {
      vi.mocked(networking.teamInfoCall).mockResolvedValue(
        createMockTeamData({
          object_permission: {
            object_permission_id: "perm-1",
            mcp_servers: ["server1"],
            vector_stores: ["store1"],
          },
        }),
      );

      renderWithProviders(<TeamInfoView {...defaultProps} />);

      await waitFor(() => {
        const teamNameElements = screen.queryAllByText("Test Team");
        expect(teamNameElements.length).toBeGreaterThan(0);
      });
    });

    it("should open Settings tab by default when editTeam is true and user can edit", async () => {
      vi.mocked(networking.teamInfoCall).mockResolvedValue(createMockTeamData());

      renderWithProviders(<TeamInfoView {...defaultProps} editTeam={true} />);

      await waitFor(() => {
        const teamNameElements = screen.queryAllByText("Test Team");
        expect(teamNameElements.length).toBeGreaterThan(0);
      });

      expect(screen.getByText("Team Settings")).toBeInTheDocument();
    });

    it("should open Overview tab by default when editTeam is false", async () => {
      vi.mocked(networking.teamInfoCall).mockResolvedValue(createMockTeamData());

      renderWithProviders(<TeamInfoView {...defaultProps} editTeam={false} />);

      await waitFor(() => {
        const teamNameElements = screen.queryAllByText("Test Team");
        expect(teamNameElements.length).toBeGreaterThan(0);
      });

      expect(screen.getByText("Budget Status")).toBeInTheDocument();
    });

    it("should open Overview tab by default when editTeam is true but user cannot edit", async () => {
      vi.mocked(networking.teamInfoCall).mockResolvedValue(createMockTeamData());

      renderWithProviders(
        <TeamInfoView {...defaultProps} editTeam={true} is_team_admin={false} is_proxy_admin={false} />,
      );

      await waitFor(() => {
        const teamNameElements = screen.queryAllByText("Test Team");
        expect(teamNameElements.length).toBeGreaterThan(0);
      });

      expect(screen.getByText("Budget Status")).toBeInTheDocument();
    });
  });

  describe("tabs and navigation", () => {
    it("should show members tab when user can edit team", async () => {
      vi.mocked(networking.teamInfoCall).mockResolvedValue(createMockTeamData());

      renderWithProviders(<TeamInfoView {...defaultProps} />);

      await waitFor(() => {
        expect(screen.getByRole("tab", { name: "Members" })).toBeInTheDocument();
      });
    });

    it("should not show members tab when user cannot edit team", async () => {
      vi.mocked(networking.teamInfoCall).mockResolvedValue(createMockTeamData());

      renderWithProviders(<TeamInfoView {...defaultProps} is_team_admin={false} is_proxy_admin={false} />);

      await waitFor(() => {
        const teamNameElements = screen.queryAllByText("Test Team");
        expect(teamNameElements.length).toBeGreaterThan(0);
      });

      expect(screen.queryByRole("tab", { name: "Members" })).not.toBeInTheDocument();
    });

    it("should show settings tab when user can edit team", async () => {
      vi.mocked(networking.teamInfoCall).mockResolvedValue(createMockTeamData());

      renderWithProviders(<TeamInfoView {...defaultProps} />);

      await waitFor(() => {
        expect(screen.getByRole("tab", { name: "Settings" })).toBeInTheDocument();
      });
    });

    it("shows edit tabs when the fetched team data marks the session user as team admin, even without the is_team_admin prop", async () => {
      vi.mocked(networking.teamInfoCall).mockResolvedValue(
        createMockTeamData({
          members_with_roles: [
            {
              user_id: "user-1",
              user_email: "admin@test.com",
              role: "admin",
              spend: 0,
              budget_id: "budget1",
            },
          ],
        }),
      );

      renderWithProviders(<TeamInfoView {...defaultProps} is_team_admin={false} is_proxy_admin={false} />);

      await waitFor(() => {
        expect(screen.getByRole("tab", { name: "Settings" })).toBeInTheDocument();
      });
      expect(screen.getByRole("tab", { name: "Members" })).toBeInTheDocument();
    });

    it("should navigate to settings tab when clicked", async () => {
      const user = userEvent.setup({ delay: null });
      vi.mocked(networking.teamInfoCall).mockResolvedValue(createMockTeamData());

      renderWithProviders(<TeamInfoView {...defaultProps} />);

      await waitFor(() => {
        const teamNameElements = screen.queryAllByText("Test Team");
        expect(teamNameElements.length).toBeGreaterThan(0);
      });

      const settingsTab = screen.getByRole("tab", { name: "Settings" });
      await user.click(settingsTab);

      await waitFor(() => {
        expect(screen.getByText("Team Settings")).toBeInTheDocument();
      });
    });

    it("should call onClose when back button is clicked", async () => {
      const user = userEvent.setup({ delay: null });
      const onClose = vi.fn();
      vi.mocked(networking.teamInfoCall).mockResolvedValue(createMockTeamData());

      renderWithProviders(<TeamInfoView {...defaultProps} onClose={onClose} />);

      await waitFor(() => {
        const teamNameElements = screen.queryAllByText("Test Team");
        expect(teamNameElements.length).toBeGreaterThan(0);
      });

      const backButton = screen.getByRole("button", { name: /back to teams/i });
      await user.click(backButton);

      expect(onClose).toHaveBeenCalled();
    });

    it("should copy team ID to clipboard when copy button is clicked", async () => {
      const user = userEvent.setup({ delay: null });
      vi.mocked(networking.teamInfoCall).mockResolvedValue(createMockTeamData());

      renderWithProviders(<TeamInfoView {...defaultProps} />);

      await waitFor(() => {
        const teamNameElements = screen.queryAllByText("Test Team");
        expect(teamNameElements.length).toBeGreaterThan(0);
      });

      const copyButtons = screen.getAllByRole("button");
      const copyButton = copyButtons.find((btn) => btn.querySelector("svg"));
      expect(copyButton).toBeTruthy();

      if (copyButton) {
        await user.click(copyButton);
      }
    });

    it("should show Virtual Keys tab when user cannot edit team", async () => {
      vi.mocked(networking.teamInfoCall).mockResolvedValue(createMockTeamData());

      renderWithProviders(<TeamInfoView {...defaultProps} is_team_admin={false} is_proxy_admin={false} />);

      await waitFor(() => {
        expect(screen.getByRole("tab", { name: "Virtual Keys" })).toBeInTheDocument();
      });
    });

    it("should display X Members in Virtual Keys tab when navigated to", async () => {
      const user = userEvent.setup();
      vi.mocked(networking.teamInfoCall).mockResolvedValue(createMockTeamData());
      const fiveKeys = Array.from({ length: 5 }, (_, i) => ({
        token: `sk-${i}`,
        token_id: `key-${i}`,
        key_alias: `key_${i}`,
        key_name: `sk-...${i}`,
        user_id: `user-${i}`,
        organization_id: null,
        user: { user_id: `user-${i}`, user_email: `user${i}@test.com` },
        created_at: "2024-01-01T00:00:00Z",
        team_id: "123",
        spend: 0,
        max_budget: 100,
        models: ["gpt-4"],
      }));
      mockUseKeys.mockReturnValue({
        data: { keys: fiveKeys, total_count: 5, current_page: 1, total_pages: 1 },
        isPending: false,
        isFetching: false,
        refetch: vi.fn(),
      } as any);

      renderWithProviders(<TeamInfoView {...defaultProps} />);

      await waitFor(() => {
        const teamNameElements = screen.queryAllByText("Test Team");
        expect(teamNameElements.length).toBeGreaterThan(0);
      });

      const virtualKeysTab = screen.getByRole("tab", { name: "Virtual Keys" });
      await user.click(virtualKeysTab);

      await waitFor(() => {
        expect(screen.getByTestId("pagination-range")).toHaveTextContent("Showing 1-5 of 5");
      });
    });

    it("should show Filters and pagination controls in Virtual Keys tab", async () => {
      const user = userEvent.setup();
      vi.mocked(networking.teamInfoCall).mockResolvedValue(createMockTeamData());
      mockUseKeys.mockReturnValue({
        data: {
          keys: [
            {
              token: "sk-1",
              token_id: "key-1",
              key_alias: "key1",
              key_name: "sk-...1",
              user_id: "user-1",
              organization_id: null,
              user: { user_id: "user-1", user_email: "user1@test.com" },
              created_at: "2024-01-01T00:00:00Z",
              team_id: "123",
              spend: 0,
              max_budget: 100,
              models: ["gpt-4"],
            },
          ],
          total_count: 1,
          current_page: 1,
          total_pages: 1,
        },
        isPending: false,
        isFetching: false,
        refetch: vi.fn(),
      } as any);

      renderWithProviders(<TeamInfoView {...defaultProps} />);

      await waitFor(() => {
        const teamNameElements = screen.queryAllByText("Test Team");
        expect(teamNameElements.length).toBeGreaterThan(0);
      });

      const virtualKeysTab = screen.getByRole("tab", { name: "Virtual Keys" });
      await user.click(virtualKeysTab);

      await waitFor(() => {
        expect(screen.getByRole("button", { name: "Filters" })).toBeInTheDocument();
      });
      expect(screen.getByRole("button", { name: "Columns" })).toBeInTheDocument();
      expect(screen.getByTestId("pagination-range")).toHaveTextContent("Showing 1-1 of 1");
      expect(screen.getByTestId("pagination-prev")).toBeInTheDocument();
      expect(screen.getByTestId("pagination-next")).toBeInTheDocument();
    });
  });

  describe("settings and editing", () => {
    const policiesFormFieldLabel = () => screen.queryByText("Policies", { selector: "label" });

    it("should offer the policies field and load it for a caller with the viewPolicies capability", async () => {
      const user = userEvent.setup({ delay: null });
      vi.mocked(networking.teamInfoCall).mockResolvedValue(createMockTeamData());

      renderWithProviders(<TeamInfoView {...defaultProps} />);

      await waitFor(() => {
        expect(networking.getPoliciesList).toHaveBeenCalled();
      });
      expect(can).toHaveBeenCalledWith("viewPolicies");

      await user.click(screen.getByRole("tab", { name: "Settings" }));
      await user.click(await screen.findByRole("button", { name: /edit settings/i }));

      await waitFor(() => {
        expect(policiesFormFieldLabel()).toBeInTheDocument();
      });
    });

    it("should omit the policies field and skip the admin-only list without the capability", async () => {
      can.mockReturnValue(false);
      const user = userEvent.setup({ delay: null });
      vi.mocked(networking.teamInfoCall).mockResolvedValue(createMockTeamData());

      renderWithProviders(<TeamInfoView {...defaultProps} />);

      await user.click(await screen.findByRole("tab", { name: "Settings" }));
      await user.click(await screen.findByRole("button", { name: /edit settings/i }));

      expect(await screen.findByLabelText("Team Name")).toBeInTheDocument();

      expect(networking.getPoliciesList).not.toHaveBeenCalled();
      expect(policiesFormFieldLabel()).not.toBeInTheDocument();
    });

    it("should open edit mode when edit button is clicked", async () => {
      const user = userEvent.setup({ delay: null });
      vi.mocked(networking.teamInfoCall).mockResolvedValue(createMockTeamData());

      renderWithProviders(<TeamInfoView {...defaultProps} />);

      await waitFor(() => {
        const teamNameElements = screen.queryAllByText("Test Team");
        expect(teamNameElements.length).toBeGreaterThan(0);
      });

      const settingsTab = screen.getByRole("tab", { name: "Settings" });
      await user.click(settingsTab);

      await waitFor(() => {
        expect(screen.getByRole("button", { name: /edit settings/i })).toBeInTheDocument();
      });

      const editButton = screen.getByRole("button", { name: /edit settings/i });
      await user.click(editButton);

      await waitFor(() => {
        expect(screen.getByLabelText("Team Name")).toBeInTheDocument();
      });
    });

    it("should close edit mode when cancel button is clicked", async () => {
      const user = userEvent.setup({ delay: null });
      vi.mocked(networking.teamInfoCall).mockResolvedValue(createMockTeamData());

      renderWithProviders(<TeamInfoView {...defaultProps} />);

      await waitFor(() => {
        const teamNameElements = screen.queryAllByText("Test Team");
        expect(teamNameElements.length).toBeGreaterThan(0);
      });

      const settingsTab = screen.getByRole("tab", { name: "Settings" });
      await user.click(settingsTab);

      await waitFor(() => {
        expect(screen.getByRole("button", { name: /edit settings/i })).toBeInTheDocument();
      });

      const editButton = screen.getByRole("button", { name: /edit settings/i });
      await user.click(editButton);

      await waitFor(() => {
        expect(screen.getByLabelText("Team Name")).toBeInTheDocument();
      });

      const cancelButton = screen.getByRole("button", { name: /cancel/i });
      await user.click(cancelButton);

      await waitFor(() => {
        expect(screen.queryByLabelText("Team Name")).not.toBeInTheDocument();
      });
    });

    it("should disable secret manager settings for non-premium users", async () => {
      const user = userEvent.setup({ delay: null });
      vi.mocked(networking.teamInfoCall).mockResolvedValue(
        createMockTeamData({
          metadata: {
            secret_manager_settings: { provider: "aws", secret_id: "abc" },
          },
        }),
      );

      renderWithProviders(<TeamInfoView {...defaultProps} premiumUser={false} />);

      await waitFor(() => {
        const teamNameElements = screen.queryAllByText("Test Team");
        expect(teamNameElements.length).toBeGreaterThan(0);
      });

      const settingsTab = screen.getByRole("tab", { name: "Settings" });
      await user.click(settingsTab);

      await waitFor(() => {
        expect(screen.getByRole("button", { name: /edit settings/i })).toBeInTheDocument();
      });

      const editButton = screen.getByRole("button", { name: /edit settings/i });
      await user.click(editButton);

      const secretField = await screen.findByPlaceholderText(
        '{"namespace": "admin", "mount": "secret", "path_prefix": "litellm"}',
      );
      expect(secretField).toBeDisabled();
    });

    it("should allow premium users to edit secret manager settings", async () => {
      const user = userEvent.setup({ delay: null });
      vi.mocked(networking.teamInfoCall).mockResolvedValue(
        createMockTeamData({
          metadata: {
            secret_manager_settings: { provider: "aws", secret_id: "abc" },
          },
        }),
      );
      vi.mocked(networking.teamUpdateCall).mockResolvedValue({ data: {}, team_id: "123" } as any);

      renderWithProviders(<TeamInfoView {...defaultProps} premiumUser={true} />);

      await waitFor(() => {
        const teamNameElements = screen.queryAllByText("Test Team");
        expect(teamNameElements.length).toBeGreaterThan(0);
      });

      const settingsTab = screen.getByRole("tab", { name: "Settings" });
      await user.click(settingsTab);

      await waitFor(() => {
        expect(screen.getByRole("button", { name: /edit settings/i })).toBeInTheDocument();
      });

      const editButton = screen.getByRole("button", { name: /edit settings/i });
      await user.click(editButton);

      const secretField = await screen.findByPlaceholderText(
        '{"namespace": "admin", "mount": "secret", "path_prefix": "litellm"}',
      );
      expect(secretField).toBeEnabled();
    });

    it("should add team member when form is submitted", async () => {
      const user = userEvent.setup({ delay: null });
      const onUpdate = vi.fn();
      const teamData = createMockTeamData();
      vi.mocked(networking.teamInfoCall).mockResolvedValue(teamData);
      vi.mocked(networking.teamMemberAddCall).mockResolvedValue({} as any);

      renderWithProviders(<TeamInfoView {...defaultProps} onUpdate={onUpdate} />);

      await waitFor(() => {
        const teamNameElements = screen.queryAllByText("Test Team");
        expect(teamNameElements.length).toBeGreaterThan(0);
      });

      const membersTab = screen.getByRole("tab", { name: "Members" });
      await user.click(membersTab);

      await waitFor(() => {
        expect(screen.getByRole("button", { name: /add member/i })).toBeInTheDocument();
      });

      const addButton = screen.getByRole("button", { name: /add member/i });
      await user.click(addButton);

      await waitFor(() => {
        expect(screen.getByRole("button", { name: "Submit" })).toBeInTheDocument();
      });

      const submitButton = screen.getByRole("button", { name: "Submit" });
      await user.click(submitButton);

      await waitFor(() => {
        expect(networking.teamMemberAddCall).toHaveBeenCalled();
      });
    });

    it("should display soft budget in settings view when present", async () => {
      const user = userEvent.setup({ delay: null });
      vi.mocked(networking.teamInfoCall).mockResolvedValue(
        createMockTeamData({
          soft_budget: 500.75,
          max_budget: 1000,
        }),
      );

      renderWithProviders(<TeamInfoView {...defaultProps} />);

      await waitFor(() => {
        const teamNameElements = screen.queryAllByText("Test Team");
        expect(teamNameElements.length).toBeGreaterThan(0);
      });

      const settingsTab = screen.getByRole("tab", { name: "Settings" });
      await user.click(settingsTab);

      await waitFor(() => {
        expect(screen.getByText("Team Settings")).toBeInTheDocument();
      });

      await waitFor(() => {
        expect(screen.getByText(/Soft Budget:/)).toBeInTheDocument();
        expect(screen.getByText(/\$500\.75/)).toBeInTheDocument();
      });
    });

    it("should display soft budget alerting emails in settings view when present", async () => {
      const user = userEvent.setup({ delay: null });
      vi.mocked(networking.teamInfoCall).mockResolvedValue(
        createMockTeamData({
          metadata: {
            soft_budget_alerting_emails: ["alert1@test.com", "alert2@test.com"],
          },
        }),
      );

      renderWithProviders(<TeamInfoView {...defaultProps} />);

      await waitFor(() => {
        const teamNameElements = screen.queryAllByText("Test Team");
        expect(teamNameElements.length).toBeGreaterThan(0);
      });

      const settingsTab = screen.getByRole("tab", { name: "Settings" });
      await user.click(settingsTab);

      await waitFor(() => {
        expect(screen.getByText("Team Settings")).toBeInTheDocument();
      });

      await waitFor(() => {
        expect(screen.getByText(/Soft Budget Alerting Emails:/)).toBeInTheDocument();
        expect(screen.getByText(/alert1@test\.com, alert2@test\.com/)).toBeInTheDocument();
      });
    });

    it("should pass access_group_ids to teamUpdateCall when saving team settings", async () => {
      const user = userEvent.setup({ delay: null });
      const accessGroupIds = ["ag-1", "ag-2"];
      vi.mocked(networking.teamInfoCall).mockResolvedValue(
        createMockTeamData({
          access_group_ids: accessGroupIds,
          models: ["gpt-4"],
        }),
      );
      vi.mocked(networking.teamUpdateCall).mockResolvedValue({ data: {}, team_id: "123" } as any);

      renderWithProviders(<TeamInfoView {...defaultProps} />);

      await waitFor(() => {
        const teamNameElements = screen.queryAllByText("Test Team");
        expect(teamNameElements.length).toBeGreaterThan(0);
      });

      const settingsTab = screen.getByRole("tab", { name: "Settings" });
      await user.click(settingsTab);

      await waitFor(() => {
        expect(screen.getByRole("button", { name: /edit settings/i })).toBeInTheDocument();
      });

      const editButton = screen.getByRole("button", { name: /edit settings/i });
      await user.click(editButton);

      await waitFor(() => {
        expect(screen.getByLabelText("Team Name")).toBeInTheDocument();
      });

      const saveButton = screen.getByRole("button", { name: /save changes/i });
      await user.click(saveButton);

      await waitFor(() => {
        expect(networking.teamUpdateCall).toHaveBeenCalledWith(
          "test-token",
          expect.objectContaining({
            access_group_ids: accessGroupIds,
            team_id: "123",
          }),
        );
      });
    });

    const openSettingsEditorForTeam = async (
      user: ReturnType<typeof userEvent.setup>,
      teamOverrides: Record<string, unknown>,
    ) => {
      vi.mocked(networking.teamInfoCall).mockResolvedValue(createMockTeamData(teamOverrides));
      vi.mocked(networking.teamUpdateCall).mockResolvedValue({ data: {}, team_id: "123" } as any);

      renderWithProviders(<TeamInfoView {...defaultProps} />);

      await waitFor(() => {
        expect(screen.queryAllByText("Test Team").length).toBeGreaterThan(0);
      });

      await user.click(screen.getByRole("tab", { name: "Settings" }));
      await waitFor(() => {
        expect(screen.getByRole("button", { name: /edit settings/i })).toBeInTheDocument();
      });
      await user.click(screen.getByRole("button", { name: /edit settings/i }));
      await waitFor(() => {
        expect(screen.getByLabelText("Team Name")).toBeInTheDocument();
      });

      return screen.getByLabelText("Reset Budget");
    };

    it("should send an explicit null budget_duration when a stored Reset Budget is cleared", async () => {
      const user = userEvent.setup({ delay: null });
      const resetBudgetSelect = await openSettingsEditorForTeam(user, { budget_duration: "30d" });

      await chooseSelectOption(user, resetBudgetSelect, "Never resets");

      await waitFor(() => {
        expect(resetBudgetSelect).toHaveTextContent("Never resets");
      });

      await user.click(screen.getByRole("button", { name: /save changes/i }));

      await waitFor(() => {
        expect(networking.teamUpdateCall).toHaveBeenCalled();
      });

      const updateArg = vi.mocked(networking.teamUpdateCall).mock.calls[0][1];
      expect(updateArg.budget_duration).toBeNull();
      expect(JSON.stringify(updateArg)).toContain('"budget_duration":null');
    });

    it("should keep a stored Reset Budget when the form is saved untouched", async () => {
      const user = userEvent.setup({ delay: null });
      await openSettingsEditorForTeam(user, { budget_duration: "30d" });

      await user.click(screen.getByRole("button", { name: /save changes/i }));

      await waitFor(() => {
        expect(networking.teamUpdateCall).toHaveBeenCalled();
      });

      expect(vi.mocked(networking.teamUpdateCall).mock.calls[0][1].budget_duration).toBe("30d");
    });

    it("should send the newly picked budget_duration when one is selected", async () => {
      const user = userEvent.setup({ delay: null });
      const resetBudgetSelect = await openSettingsEditorForTeam(user, { budget_duration: null });

      await chooseSelectOption(user, resetBudgetSelect, "weekly");

      await user.click(screen.getByRole("button", { name: /save changes/i }));

      await waitFor(() => {
        expect(networking.teamUpdateCall).toHaveBeenCalled();
      });

      expect(vi.mocked(networking.teamUpdateCall).mock.calls[0][1].budget_duration).toBe("7d");
    });
  });

  describe("metadata key-value editing", () => {
    const openSettingsEditor = async (user: ReturnType<typeof userEvent.setup>) => {
      await waitFor(() => {
        const teamNameElements = screen.queryAllByText("Test Team");
        expect(teamNameElements.length).toBeGreaterThan(0);
      });

      await user.click(screen.getByRole("tab", { name: "Settings" }));

      await waitFor(() => {
        expect(screen.getByRole("button", { name: /edit settings/i })).toBeInTheDocument();
      });

      await user.click(screen.getByRole("button", { name: /edit settings/i }));

      await waitFor(() => {
        expect(screen.getByLabelText("Team Name")).toBeInTheDocument();
      });
    };

    it("should preserve metadata types and hide managed keys", async () => {
      const user = userEvent.setup({ delay: null });
      vi.mocked(networking.teamInfoCall).mockResolvedValue(
        createMockTeamData({
          metadata: {
            department: "research",
            tier: 3,
            beta: true,
            config: { region: "us" },
            logging: [{ callback_name: "langfuse", callback_type: "success", callback_vars: {} }],
            guardrails: ["g1"],
            disable_global_guardrails: false,
            model_tpm_limit: { "gpt-4": 100 },
          },
          models: ["gpt-4"],
        }),
      );
      vi.mocked(networking.teamUpdateCall).mockResolvedValue({ data: {}, team_id: "123" } as any);

      renderWithProviders(<TeamInfoView {...defaultProps} />);
      await openSettingsEditor(user);

      const keyValues = screen.getAllByPlaceholderText("Key").map((input) => (input as HTMLInputElement).value);
      expect(keyValues).toEqual(["department", "tier", "beta", "config"]);
      const valueValues = screen.getAllByPlaceholderText("Value").map((input) => (input as HTMLInputElement).value);
      expect(valueValues).toEqual(["research", "3", "true", '{"region":"us"}']);

      await user.click(screen.getByRole("button", { name: /save changes/i }));

      await waitFor(() => {
        expect(networking.teamUpdateCall).toHaveBeenCalled();
      });

      const updateArg = vi.mocked(networking.teamUpdateCall).mock.calls[0][1];
      expect(updateArg.metadata).toMatchObject({
        department: "research",
        tier: 3,
        beta: true,
        config: { region: "us" },
        logging: [{ callback_name: "langfuse", callback_type: "success", callback_vars: {} }],
      });
      expect(updateArg.metadata).not.toHaveProperty("model_tpm_limit");
      expect(updateArg.model_tpm_limit).toEqual({ "gpt-4": 100 });
    });

    it("prefills the estimated output token controls, hides them from the pair editor, and saves edits", async () => {
      const user = userEvent.setup({ delay: null });
      vi.mocked(networking.teamInfoCall).mockResolvedValue(
        createMockTeamData({
          metadata: {
            department: "research",
            default_estimated_output_tokens: 512,
            default_estimated_output_tokens_per_model: { "gpt-4": 4096 },
          },
          models: ["gpt-4"],
        }),
      );
      vi.mocked(networking.teamUpdateCall).mockResolvedValue({ data: {}, team_id: "123" } as any);

      renderWithProviders(<TeamInfoView {...defaultProps} />);
      await openSettingsEditor(user);

      expect(screen.getByLabelText("Estimated Output Tokens")).toHaveValue(512);
      expect(screen.getByLabelText("Estimated Output Tokens Per Model")).toHaveValue('{"gpt-4":4096}');
      const keyValues = screen.queryAllByPlaceholderText("Key").map((input) => (input as HTMLInputElement).value);
      expect(keyValues).toEqual(["department"]);

      fireEvent.change(screen.getByLabelText("Estimated Output Tokens"), { target: { value: "999" } });

      await user.click(screen.getByRole("button", { name: /save changes/i }));

      await waitFor(() => {
        expect(networking.teamUpdateCall).toHaveBeenCalled();
      });

      const updateArg = vi.mocked(networking.teamUpdateCall).mock.calls[0][1];
      expect(updateArg.metadata.default_estimated_output_tokens).toBe(999);
      expect(updateArg.metadata.default_estimated_output_tokens_per_model).toEqual({ "gpt-4": 4096 });
    });

    it("omits the estimated output token settings when both controls are blank", async () => {
      const user = userEvent.setup({ delay: null });
      vi.mocked(networking.teamInfoCall).mockResolvedValue(createMockTeamData({ models: ["gpt-4"] }));
      vi.mocked(networking.teamUpdateCall).mockResolvedValue({ data: {}, team_id: "123" } as any);

      renderWithProviders(<TeamInfoView {...defaultProps} />);
      await openSettingsEditor(user);

      await user.click(screen.getByRole("button", { name: /save changes/i }));

      await waitFor(() => {
        expect(networking.teamUpdateCall).toHaveBeenCalled();
      });

      const updateArg = vi.mocked(networking.teamUpdateCall).mock.calls[0][1];
      expect(updateArg.metadata).not.toHaveProperty("default_estimated_output_tokens");
      expect(updateArg.metadata).not.toHaveProperty("default_estimated_output_tokens_per_model");
    });

    it.each(["Internal User", "Admin Viewer", "org_admin"])(
      "leaves both estimate controls read-only for %s and still resubmits the stored values",
      async (userRole) => {
        authState.userRole = userRole;
        const user = userEvent.setup({ delay: null });
        vi.mocked(networking.teamInfoCall).mockResolvedValue(
          createMockTeamData({
            metadata: {
              default_estimated_output_tokens: 512,
              default_estimated_output_tokens_per_model: { "gpt-4": 4096 },
            },
            models: ["gpt-4"],
          }),
        );
        vi.mocked(networking.teamUpdateCall).mockResolvedValue({ data: {}, team_id: "123" } as any);

        renderWithProviders(<TeamInfoView {...defaultProps} />);
        await openSettingsEditor(user);

        expect(screen.getByLabelText("Estimated Output Tokens")).toBeDisabled();
        expect(screen.getByLabelText("Estimated Output Tokens Per Model")).toBeDisabled();

        await user.click(screen.getByRole("button", { name: /save changes/i }));

        await waitFor(() => {
          expect(networking.teamUpdateCall).toHaveBeenCalled();
        });

        const updateArg = vi.mocked(networking.teamUpdateCall).mock.calls[0][1];
        expect(updateArg.metadata.default_estimated_output_tokens).toBe(512);
        expect(updateArg.metadata.default_estimated_output_tokens_per_model).toEqual({ "gpt-4": 4096 });
      },
    );

    it.each(["Admin", "proxy_admin"])("leaves both estimate controls editable for %s", async (userRole) => {
      authState.userRole = userRole;
      const user = userEvent.setup({ delay: null });
      vi.mocked(networking.teamInfoCall).mockResolvedValue(createMockTeamData({ models: ["gpt-4"] }));

      renderWithProviders(<TeamInfoView {...defaultProps} />);
      await openSettingsEditor(user);

      expect(screen.getByLabelText("Estimated Output Tokens")).toBeEnabled();
      expect(screen.getByLabelText("Estimated Output Tokens Per Model")).toBeEnabled();
    });

    it("should keep declared keys as ordinary prefilled rows and submit the edited value", async () => {
      const user = userEvent.setup({ delay: null });
      vi.mocked(useTeamMetadataSchema).mockReturnValue({
        data: [
          { key: "cost_center", label: "Cost Center" },
          { key: "app_name", label: "Application Name" },
        ],
        isLoading: false,
      } as any);
      vi.mocked(networking.teamInfoCall).mockResolvedValue(
        createMockTeamData({
          metadata: { cost_center: "CC-OLD", department: "research" },
          models: ["gpt-4"],
        }),
      );
      vi.mocked(networking.teamUpdateCall).mockResolvedValue({ data: {}, team_id: "123" } as any);

      renderWithProviders(<TeamInfoView {...defaultProps} />);
      await openSettingsEditor(user);

      await waitFor(() => {
        expect(screen.getAllByPlaceholderText("Key").map((input) => (input as HTMLInputElement).value)).toEqual([
          "cost_center",
          "department",
          "app_name",
        ]);
      });
      expect(screen.getAllByPlaceholderText("Value")[0]).toHaveValue("CC-OLD");

      await user.clear(screen.getAllByPlaceholderText("Value")[0]);
      fireEvent.change(screen.getAllByPlaceholderText("Value")[0], { target: { value: "CC-NEW" } });
      await user.click(screen.getByRole("button", { name: /save changes/i }));

      await waitFor(() => {
        expect(networking.teamUpdateCall).toHaveBeenCalled();
      });

      expect(vi.mocked(networking.teamUpdateCall).mock.calls[0][1].metadata).toMatchObject({
        cost_center: "CC-NEW",
        department: "research",
        app_name: "",
      });
    });
  });

  describe("model aliases", () => {
    const openSettingsEditor = async (user: ReturnType<typeof userEvent.setup>) => {
      await waitFor(() => {
        const teamNameElements = screen.queryAllByText("Test Team");
        expect(teamNameElements.length).toBeGreaterThan(0);
      });

      await user.click(screen.getByRole("tab", { name: "Settings" }));

      await waitFor(() => {
        expect(screen.getByRole("button", { name: /edit settings/i })).toBeInTheDocument();
      });

      await user.click(screen.getByRole("button", { name: /edit settings/i }));

      await waitFor(() => {
        expect(screen.getByLabelText("Team Name")).toBeInTheDocument();
      });
    };

    it("should render existing model aliases in the read-only settings view", async () => {
      const user = userEvent.setup({ delay: null });
      vi.mocked(networking.teamInfoCall).mockResolvedValue(
        createMockTeamData({
          litellm_model_table: { model_aliases: { "my-smart-model": "gpt-4", "my-fast-model": "gpt-3.5-turbo" } },
        }),
      );

      renderWithProviders(<TeamInfoView {...defaultProps} />);

      await waitFor(() => {
        const teamNameElements = screen.queryAllByText("Test Team");
        expect(teamNameElements.length).toBeGreaterThan(0);
      });

      await user.click(screen.getByRole("tab", { name: "Settings" }));

      await waitFor(() => {
        expect(screen.getByText("Team Settings")).toBeInTheDocument();
      });

      expect(screen.getByText("Model Aliases")).toBeInTheDocument();
      expect(screen.getByText("my-smart-model")).toBeInTheDocument();
      expect(screen.getByText("gpt-4")).toBeInTheDocument();
      expect(screen.getByText("my-fast-model")).toBeInTheDocument();
      expect(screen.getByText("gpt-3.5-turbo")).toBeInTheDocument();
    });

    it("should render the estimated output token settings in the overview and read-only settings views", async () => {
      const user = userEvent.setup({ delay: null });
      vi.mocked(networking.teamInfoCall).mockResolvedValue(
        createMockTeamData({
          metadata: {
            default_estimated_output_tokens: 512,
            default_estimated_output_tokens_per_model: { "gpt-4": 4096 },
          },
        }),
      );

      renderWithProviders(<TeamInfoView {...defaultProps} />);

      await waitFor(() => {
        const teamNameElements = screen.queryAllByText("Test Team");
        expect(teamNameElements.length).toBeGreaterThan(0);
      });

      await user.click(screen.getByRole("tab", { name: "Settings" }));

      await waitFor(() => {
        expect(screen.getByText("Team Settings")).toBeInTheDocument();
      });

      expect(screen.getAllByText("Estimated Output Tokens: 512")).toHaveLength(2);
      expect(screen.getAllByText('Estimated Output Tokens Per Model: {"gpt-4":4096}')).toHaveLength(2);
    });

    it("should show an empty state when the team has no model aliases", async () => {
      const user = userEvent.setup({ delay: null });
      vi.mocked(networking.teamInfoCall).mockResolvedValue(createMockTeamData({ litellm_model_table: null }));

      renderWithProviders(<TeamInfoView {...defaultProps} />);

      await waitFor(() => {
        const teamNameElements = screen.queryAllByText("Test Team");
        expect(teamNameElements.length).toBeGreaterThan(0);
      });

      await user.click(screen.getByRole("tab", { name: "Settings" }));

      await waitFor(() => {
        expect(screen.getByText("Team Settings")).toBeInTheDocument();
      });

      expect(screen.getByText("No model aliases configured")).toBeInTheDocument();
    });

    it("should seed the alias editor from existing team aliases", async () => {
      const user = userEvent.setup({ delay: null });
      vi.mocked(networking.teamInfoCall).mockResolvedValue(
        createMockTeamData({
          models: ["gpt-4"],
          litellm_model_table: { model_aliases: { "my-smart-model": "gpt-4" } },
        }),
      );

      renderWithProviders(<TeamInfoView {...defaultProps} />);

      await openSettingsEditor(user);

      expect(screen.getByTestId("alias-editor-initial")).toHaveTextContent(
        JSON.stringify({ "my-smart-model": "gpt-4" }),
      );
    });

    it("should pass model_aliases to teamUpdateCall when aliases are added", async () => {
      const user = userEvent.setup({ delay: null });
      vi.mocked(networking.teamInfoCall).mockResolvedValue(createMockTeamData({ models: ["gpt-4"] }));
      vi.mocked(networking.teamUpdateCall).mockResolvedValue({ data: {}, team_id: "123" } as any);

      renderWithProviders(<TeamInfoView {...defaultProps} />);

      await openSettingsEditor(user);

      await user.click(screen.getByRole("button", { name: "Set Alias" }));
      await user.click(screen.getByRole("button", { name: /save changes/i }));

      await waitFor(() => {
        expect(networking.teamUpdateCall).toHaveBeenCalledWith(
          "test-token",
          expect.objectContaining({
            team_id: "123",
            model_aliases: { "gpt-4o": "gpt-4" },
          }),
        );
      });
    });

    it("should send an empty model_aliases map to clear existing aliases", async () => {
      const user = userEvent.setup({ delay: null });
      vi.mocked(networking.teamInfoCall).mockResolvedValue(
        createMockTeamData({
          models: ["gpt-4"],
          litellm_model_table: { model_aliases: { "my-smart-model": "gpt-4" } },
        }),
      );
      vi.mocked(networking.teamUpdateCall).mockResolvedValue({ data: {}, team_id: "123" } as any);

      renderWithProviders(<TeamInfoView {...defaultProps} />);

      await openSettingsEditor(user);

      await user.click(screen.getByRole("button", { name: "Clear Aliases" }));
      await user.click(screen.getByRole("button", { name: /save changes/i }));

      await waitFor(() => {
        expect(networking.teamUpdateCall).toHaveBeenCalled();
      });

      const payload = vi.mocked(networking.teamUpdateCall).mock.calls[0][1] as Record<string, unknown>;
      expect(payload.model_aliases).toEqual({});
    });

    it("should not include model_aliases when the team has none and the editor is untouched", async () => {
      const user = userEvent.setup({ delay: null });
      vi.mocked(networking.teamInfoCall).mockResolvedValue(
        createMockTeamData({ models: ["gpt-4"], litellm_model_table: null }),
      );
      vi.mocked(networking.teamUpdateCall).mockResolvedValue({ data: {}, team_id: "123" } as any);

      renderWithProviders(<TeamInfoView {...defaultProps} />);

      await openSettingsEditor(user);

      await user.click(screen.getByRole("button", { name: /save changes/i }));

      await waitFor(() => {
        expect(networking.teamUpdateCall).toHaveBeenCalled();
      });

      const payload = vi.mocked(networking.teamUpdateCall).mock.calls[0][1] as Record<string, unknown>;
      expect(payload).not.toHaveProperty("model_aliases");
    });
  });

  describe("guardrails dropdown grouping", () => {
    const guardrail = (name: string, defaultOn: boolean) => ({
      guardrail_name: name,
      litellm_params: { default_on: defaultOn },
    });

    const openGuardrailsDropdown = async (user: ReturnType<typeof userEvent.setup>) => {
      renderWithProviders(<TeamInfoView {...defaultProps} />);

      await waitFor(() => {
        const teamNameElements = screen.queryAllByText("Test Team");
        expect(teamNameElements.length).toBeGreaterThan(0);
      });

      await user.click(screen.getByRole("tab", { name: "Settings" }));

      await waitFor(() => {
        expect(screen.getByRole("button", { name: /edit settings/i })).toBeInTheDocument();
      });

      await user.click(screen.getByRole("button", { name: /edit settings/i }));

      await waitFor(() => {
        expect(screen.getByLabelText(/^Guardrails/)).toBeInTheDocument();
      });

      await user.click(screen.getByLabelText(/^Guardrails/));

      const listbox = await screen.findByRole("listbox", {}, { timeout: 5000 });
      return listbox.closest('[data-slot="combobox-content"]') as HTMLElement;
    };

    beforeEach(() => {
      testQueryClient.clear();
      vi.mocked(networking.teamInfoCall).mockResolvedValue(createMockTeamData());
    });

    it("should not render the Global or Other group headers when no global guardrails exist", async () => {
      const user = userEvent.setup({ delay: null });
      vi.mocked(networking.getGuardrailsList).mockResolvedValue({
        guardrails: [guardrail("dwacxzcz", false), guardrail("dwadsa", false)],
      });

      const dropdown = await openGuardrailsDropdown(user);

      await waitFor(() => {
        expect(within(dropdown).getByTitle("dwacxzcz")).toBeInTheDocument();
      });
      expect(within(dropdown).getByTitle("dwadsa")).toBeInTheDocument();
      expect(within(dropdown).queryByText("Global")).not.toBeInTheDocument();
      expect(within(dropdown).queryByText("Other")).not.toBeInTheDocument();
    });

    it("should not render the Global or Other group headers when every guardrail is global", async () => {
      const user = userEvent.setup({ delay: null });
      vi.mocked(networking.getGuardrailsList).mockResolvedValue({
        guardrails: [guardrail("always-on", true)],
      });

      const dropdown = await openGuardrailsDropdown(user);

      await waitFor(() => {
        expect(within(dropdown).getByTitle("always-on")).toBeInTheDocument();
      });
      expect(within(dropdown).queryByText("Global")).not.toBeInTheDocument();
      expect(within(dropdown).queryByText("Other")).not.toBeInTheDocument();
    });

    it("should render both group headers when global and non-global guardrails exist", async () => {
      const user = userEvent.setup({ delay: null });
      vi.mocked(networking.getGuardrailsList).mockResolvedValue({
        guardrails: [guardrail("always-on", true), guardrail("opt-in", false)],
      });

      const dropdown = await openGuardrailsDropdown(user);

      await waitFor(() => {
        expect(within(dropdown).getByText("Global")).toBeInTheDocument();
      });
      expect(within(dropdown).getByText("Other")).toBeInTheDocument();
      expect(within(dropdown).getByTitle("always-on")).toBeInTheDocument();
      expect(within(dropdown).getByTitle("opt-in")).toBeInTheDocument();
    });
  });

  describe("allowed pass through routes", () => {
    beforeEach(() => {
      testQueryClient.clear();
      vi.mocked(networking.teamInfoCall).mockResolvedValue(createMockTeamData({ models: ["gpt-4"] }));
      vi.mocked(networking.teamUpdateCall).mockResolvedValue({ data: {}, team_id: "123" } as any);
      vi.mocked(networking.getPassThroughEndpointsCall).mockResolvedValue({
        endpoints: [{ path: "/bedrock-passthrough", methods: ["POST"] }],
      });
    });

    it("should show a route picked from the dropdown in the field and save it", async () => {
      const user = userEvent.setup({ delay: null });

      renderWithProviders(<TeamInfoView {...defaultProps} premiumUser={true} />);

      await waitFor(() => {
        expect(screen.queryAllByText("Test Team").length).toBeGreaterThan(0);
      });

      await user.click(screen.getByRole("tab", { name: "Settings" }));
      await user.click(await screen.findByRole("button", { name: /edit settings/i }));

      await user.click(await screen.findByRole("combobox", { name: "Select pass through routes" }));

      const option = await screen.findByText("POST /bedrock-passthrough");
      await user.click(option);

      await user.keyboard("{Escape}");

      await waitFor(() => {
        expect(screen.getByText("POST /bedrock-passthrough")).toBeInTheDocument();
      });
      await user.click(screen.getByRole("button", { name: /save changes/i }));

      await waitFor(() => {
        expect(networking.teamUpdateCall).toHaveBeenCalledWith(
          "test-token",
          expect.objectContaining({
            team_id: "123",
            metadata: expect.objectContaining({
              allowed_passthrough_routes: ["/bedrock-passthrough"],
            }),
          }),
        );
      });
    });
  });
});

describe("TeamInfoView - which team member fields reach the update payload depends on the open sections", () => {
  const props = {
    teamId: "123",
    onUpdate: vi.fn(),
    onClose: vi.fn(),
    accessToken: "test-token",
    is_team_admin: true,
    is_proxy_admin: true,
    userModels: ["gpt-4"],
    editTeam: false,
  };

  beforeEach(seedDefaultMocks);

  afterEach(() => {
    vi.clearAllMocks();
  });

  const openEditor = async (
    user: ReturnType<typeof userEvent.setup>,
    teamMemberBudgetTable: TeamData["team_info"]["team_member_budget_table"] = {
      max_budget: 42,
      budget_duration: "30d",
      tpm_limit: 11,
      rpm_limit: 22,
    },
  ) => {
    vi.mocked(networking.teamInfoCall).mockResolvedValue(
      createMockTeamData({
        team_member_budget_table: teamMemberBudgetTable,
        default_team_member_models: ["gpt-4"],
      }),
    );
    vi.mocked(networking.teamUpdateCall).mockResolvedValue({ data: {}, team_id: "123" } as any);

    renderWithProviders(<TeamInfoView {...props} />);
    await waitFor(() => expect(screen.queryAllByText("Test Team").length).toBeGreaterThan(0));
    await user.click(screen.getByRole("tab", { name: "Settings" }));
    await user.click(await screen.findByRole("button", { name: /edit settings/i }));
    await screen.findByLabelText("Team Name");
  };

  const save = async (user: ReturnType<typeof userEvent.setup>) => {
    await user.click(screen.getByRole("button", { name: /save changes/i }));
    await waitFor(() => expect(networking.teamUpdateCall).toHaveBeenCalled());
    return vi.mocked(networking.teamUpdateCall).mock.calls[0][1] as Record<string, unknown>;
  };

  it("omits every stored team member field when Team Member Settings is left closed", async () => {
    const user = userEvent.setup({ delay: null });
    await openEditor(user);

    const payload = await save(user);

    expect(payload.team_member_budget_duration).toBeUndefined();
    expect(payload).not.toHaveProperty("team_member_budget");
    expect(payload).not.toHaveProperty("team_member_tpm_limit");
    expect(payload).not.toHaveProperty("team_member_rpm_limit");
    expect(payload).not.toHaveProperty("default_team_member_models");

    const wireBody = JSON.parse(JSON.stringify(payload));
    expect(Object.keys(wireBody).filter((key) => key.startsWith("team_member"))).toEqual([]);
    expect(wireBody).not.toHaveProperty("default_team_member_models");
  });

  it("resends every stored team member field once Team Member Settings is opened", async () => {
    const user = userEvent.setup({ delay: null });
    await openEditor(user);

    await user.click(screen.getByText("Team Member Settings"));
    await screen.findByLabelText("Default Budget (USD)");
    const payload = await save(user);

    expect(payload.team_member_budget_duration).toBe("30d");
    expect(payload.team_member_budget).toBe(42);
    expect(payload.team_member_tpm_limit).toBe(11);
    expect(payload.team_member_rpm_limit).toBe(22);
    expect(payload.default_team_member_models).toEqual(["gpt-4"]);
  });

  it("sends a null team_member_budget_duration when Default Budget Duration is set to never reset", async () => {
    const user = userEvent.setup({ delay: null });
    await openEditor(user);

    await user.click(screen.getByText("Team Member Settings"));
    await screen.findByLabelText("Default Budget (USD)");
    await chooseSelectOption(user, screen.getByLabelText("Default Budget Duration"), "Never resets");

    const payload = await save(user);

    expect(payload.team_member_budget_duration).toBeNull();
    expect(payload.team_member_budget).toBe(42);
    expect(JSON.stringify(payload)).toContain('"team_member_budget_duration":null');
  });

  it("shows Never resets for a stored member budget whose duration is null", async () => {
    const user = userEvent.setup({ delay: null });
    await openEditor(user, { max_budget: 42, budget_duration: null, tpm_limit: null, rpm_limit: null });

    await user.click(screen.getByText("Team Member Settings"));

    expect(await screen.findByLabelText("Default Budget Duration")).toHaveTextContent("Never resets");
  });

  it("omits team_member_budget_duration when the dropdown is left untouched on a team with no member budget", async () => {
    const user = userEvent.setup({ delay: null });
    await openEditor(user, null);

    await user.click(screen.getByText("Team Member Settings"));
    const durationSelect = await screen.findByLabelText("Default Budget Duration");
    expect(durationSelect).toHaveTextContent("Inherit team reset period");
    expect(durationSelect).not.toHaveTextContent("Never resets");
    await user.type(screen.getByLabelText("Default Budget (USD)"), "100");

    const payload = await save(user);

    expect(payload.team_member_budget).toBe(100);
    expect(JSON.parse(JSON.stringify(payload))).not.toHaveProperty("team_member_budget_duration");
  });

  it("omits object_permission.search_tools while Search Tool Settings is closed", async () => {
    const user = userEvent.setup({ delay: null });
    await openEditor(user);

    const payload = await save(user);

    expect(payload.object_permission).not.toHaveProperty("search_tools");
  });

  it("includes object_permission.search_tools once Search Tool Settings is opened", async () => {
    const user = userEvent.setup({ delay: null });
    await openEditor(user);

    await user.click(screen.getByText("Search Tool Settings"));
    await screen.findByPlaceholderText("Select search tools (optional, empty = all allowed)");
    const payload = await save(user);

    expect(payload.object_permission).toHaveProperty("search_tools");
  });
});

describe("TeamInfoView - the exact bytes the update call sends", () => {
  const props = {
    teamId: "123",
    onUpdate: vi.fn(),
    onClose: vi.fn(),
    accessToken: "test-token",
    is_team_admin: true,
    is_proxy_admin: true,
    userModels: ["gpt-4"],
    editTeam: false,
  };

  beforeEach(seedDefaultMocks);

  afterEach(() => {
    vi.clearAllMocks();
  });

  const storedTeam = () =>
    createMockTeamData({
      models: ["gpt-4"],
      max_budget: 100,
      budget_duration: "1d",
      tpm_limit: 1000,
      rpm_limit: 1000,
      team_member_budget_table: { max_budget: 42, budget_duration: "30d", tpm_limit: 11, rpm_limit: 22 },
      default_team_member_models: ["gpt-4"],
      object_permission: { search_tools: ["tool-a"], vector_stores: ["vs-1"] },
    });

  const openEditor = async (user: ReturnType<typeof userEvent.setup>) => {
    vi.mocked(networking.teamInfoCall).mockResolvedValue(storedTeam());
    vi.mocked(networking.teamUpdateCall).mockResolvedValue({ data: {}, team_id: "123" } as any);

    renderWithProviders(<TeamInfoView {...props} />);
    await waitFor(() => expect(screen.queryAllByText("Test Team").length).toBeGreaterThan(0));
    await user.click(screen.getByRole("tab", { name: "Settings" }));
    await user.click(await screen.findByRole("button", { name: /edit settings/i }));
    await screen.findByLabelText("Team Name");
  };

  const save = async (user: ReturnType<typeof userEvent.setup>) => {
    await user.click(screen.getByRole("button", { name: /save changes/i }));
    await waitFor(() => expect(networking.teamUpdateCall).toHaveBeenCalled());
    return vi.mocked(networking.teamUpdateCall).mock.calls[0][1] as Record<string, unknown>;
  };

  const wireBody = (payload: Record<string, unknown>) => JSON.parse(JSON.stringify(payload)) as Record<string, unknown>;

  const alwaysSent = {
    team_id: "123",
    team_alias: "Test Team",
    models: ["gpt-4"],
    tpm_limit: 1000,
    rpm_limit: 1000,
    model_tpm_limit: {},
    model_rpm_limit: {},
    max_budget: 100,
    soft_budget: null,
    budget_duration: "1d",
    metadata: {
      allowed_passthrough_routes: [],
      guardrails: [],
      opted_out_global_guardrails: [],
      disable_global_guardrails: false,
      soft_budget_alerting_emails: [],
    },
    access_group_ids: [],
  };

  const mcpPermissions = {
    mcp_servers: [],
    mcp_access_groups: [],
    mcp_tool_permissions: {},
    mcp_toolsets: [],
    vector_stores: ["vs-1"],
  };

  it("leaves every team member key out of the request body for an untouched save with both sections closed", async () => {
    const user = userEvent.setup({ delay: null });
    await openEditor(user);

    const payload = await save(user);

    expect(payload).toStrictEqual({
      ...alwaysSent,
      team_member_budget_duration: undefined,
      object_permission: mcpPermissions,
    });
    expect(wireBody(payload)).toStrictEqual({
      ...alwaysSent,
      object_permission: mcpPermissions,
    });
  });

  it("resends every stored value once both sections are opened", async () => {
    const user = userEvent.setup({ delay: null });
    await openEditor(user);

    await user.click(screen.getByText("Team Member Settings"));
    await screen.findByLabelText("Default Budget (USD)");
    await user.click(screen.getByText("Search Tool Settings"));
    await screen.findByPlaceholderText("Select search tools (optional, empty = all allowed)");

    const payload = await save(user);

    const expected = {
      ...alwaysSent,
      team_member_budget_duration: "30d",
      team_member_budget: 42,
      team_member_tpm_limit: 11,
      team_member_rpm_limit: 22,
      default_team_member_models: ["gpt-4"],
      object_permission: { ...mcpPermissions, search_tools: ["tool-a"] },
    };
    expect(payload).toStrictEqual(expected);
    expect(wireBody(payload)).toStrictEqual(expected);
  });

  it("carries every typed value to the update payload at the type and shape antd sends today", async () => {
    const user = userEvent.setup({ delay: null });
    await openEditor(user);

    const alias = screen.getByLabelText("Team Name");
    await user.clear(alias);
    await user.type(alias, "Renamed Team");

    const softBudget = screen.getByLabelText("Soft Budget (USD)");
    await user.clear(softBudget);
    await user.type(softBudget, "9.5");

    const emails = screen.getByLabelText(/Soft Budget Alerting Emails/);
    await user.clear(emails);
    await user.type(emails, "a@test.com,  b@test.com ");

    const tpm = screen.getByLabelText("Tokens per minute Limit (TPM)");
    await user.clear(tpm);
    await user.type(tpm, "555");

    const payload = await save(user);

    expect(payload.team_alias).toBe("Renamed Team");
    expect(payload.soft_budget).toBe("9.5");
    expect(payload.tpm_limit).toBe("555");
    expect((payload.metadata as Record<string, unknown>).soft_budget_alerting_emails).toStrictEqual([
      "a@test.com",
      "b@test.com",
    ]);
  });

  it("builds model_tpm_limit and model_rpm_limit from the model-specific rate limit rows", async () => {
    const user = userEvent.setup({ delay: null });
    vi.mocked(networking.teamInfoCall).mockResolvedValue(
      createMockTeamData({
        models: ["gpt-4"],
        max_budget: 100,
        budget_duration: "1d",
        tpm_limit: 1000,
        rpm_limit: 1000,
        object_permission: { vector_stores: ["vs-1"] },
        metadata: { model_tpm_limit: { "gpt-4": 30 }, model_rpm_limit: { "gpt-4": 40 } },
      }),
    );
    vi.mocked(networking.teamUpdateCall).mockResolvedValue({ data: {}, team_id: "123" } as any);

    renderWithProviders(<TeamInfoView {...props} />);
    await waitFor(() => expect(screen.queryAllByText("Test Team").length).toBeGreaterThan(0));
    await user.click(screen.getByRole("tab", { name: "Settings" }));
    await user.click(await screen.findByRole("button", { name: /edit settings/i }));
    await screen.findByLabelText("Team Name");

    const payload = await save(user);

    expect(payload.model_tpm_limit).toStrictEqual({ "gpt-4": 30 });
    expect(payload.model_rpm_limit).toStrictEqual({ "gpt-4": 40 });
  });

  it("keeps a team member budget edited before the section is collapsed and resends it on reopen", async () => {
    const user = userEvent.setup({ delay: null });
    await openEditor(user);

    await user.click(screen.getByText("Team Member Settings"));
    const budgetInput = await screen.findByLabelText("Default Budget (USD)");
    await user.clear(budgetInput);
    await user.type(budgetInput, "77");

    await user.click(screen.getByText("Team Member Settings"));
    await waitFor(() => expect(screen.queryByLabelText("Default Budget (USD)")).not.toBeInTheDocument());

    await user.click(screen.getByText("Team Member Settings"));
    expect(await screen.findByLabelText("Default Budget (USD)")).toHaveValue(77);

    const payload = await save(user);
    expect(payload.team_member_budget).toBe(77);
  });

  it("sends no team member key at all when the section is collapsed again after an edit", async () => {
    const user = userEvent.setup({ delay: null });
    await openEditor(user);

    await user.click(screen.getByText("Team Member Settings"));
    const budgetInput = await screen.findByLabelText("Default Budget (USD)");
    await user.clear(budgetInput);
    await user.type(budgetInput, "77");

    await user.click(screen.getByText("Team Member Settings"));
    await waitFor(() => expect(screen.queryByLabelText("Default Budget (USD)")).not.toBeInTheDocument());

    const payload = await save(user);

    expect(Object.keys(wireBody(payload)).filter((key) => key.startsWith("team_member"))).toEqual([]);
    expect(wireBody(payload)).not.toHaveProperty("default_team_member_models");
  });

  it("puts the global guardrails back on the team when the kill switch is turned off again", async () => {
    const user = userEvent.setup({ delay: null });
    testQueryClient.clear();
    vi.mocked(networking.getGuardrailsList).mockResolvedValue({
      guardrails: [
        { guardrail_name: "always-on", litellm_params: { default_on: true } },
        { guardrail_name: "opt-in", litellm_params: { default_on: false } },
      ],
    });
    vi.mocked(networking.teamInfoCall).mockResolvedValue(
      createMockTeamData({
        models: ["gpt-4"],
        metadata: { guardrails: ["opt-in"], disable_global_guardrails: true },
      }),
    );
    vi.mocked(networking.teamUpdateCall).mockResolvedValue({ data: {}, team_id: "123" } as any);

    renderWithProviders(<TeamInfoView {...props} premiumUser={true} />);
    await waitFor(() => expect(screen.queryAllByText("Test Team").length).toBeGreaterThan(0));
    await user.click(screen.getByRole("tab", { name: "Settings" }));
    await user.click(await screen.findByRole("button", { name: /edit settings/i }));
    await screen.findByLabelText("Team Name");

    expect(screen.queryAllByLabelText("always-on")).toHaveLength(0);

    await user.click(screen.getByRole("switch", { name: /Disable all global guardrails/ }));

    expect(await screen.findAllByLabelText("always-on")).toHaveLength(1);

    const payload = await save(user);

    expect(payload.metadata).toStrictEqual(
      expect.objectContaining({
        guardrails: ["opt-in"],
        opted_out_global_guardrails: [],
        disable_global_guardrails: false,
      }),
    );
  });

  it("sends a typed model rate limit as a number", async () => {
    const user = userEvent.setup({ delay: null });
    vi.mocked(networking.teamInfoCall).mockResolvedValue(
      createMockTeamData({ models: ["gpt-4"], metadata: { model_tpm_limit: { "gpt-4": 30 } } }),
    );
    vi.mocked(networking.teamUpdateCall).mockResolvedValue({ data: {}, team_id: "123" } as any);

    renderWithProviders(<TeamInfoView {...props} />);
    await waitFor(() => expect(screen.queryAllByText("Test Team").length).toBeGreaterThan(0));
    await user.click(screen.getByRole("tab", { name: "Settings" }));
    await user.click(await screen.findByRole("button", { name: /edit settings/i }));
    await screen.findByLabelText("Team Name");

    const rpmInput = await screen.findByPlaceholderText("RPM Limit");
    await user.clear(rpmInput);
    await user.type(rpmInput, "45");

    const payload = await save(user);

    expect(payload.model_rpm_limit).toStrictEqual({ "gpt-4": 45 });
    expect(payload.model_tpm_limit).toStrictEqual({ "gpt-4": 30 });
  });

  it("leaves stored policies out of the update body for a caller without the viewPolicies capability", async () => {
    const user = userEvent.setup({ delay: null });
    can.mockReturnValue(false);
    vi.mocked(networking.teamInfoCall).mockResolvedValue(createMockTeamData({ models: ["gpt-4"], policies: ["pci"] }));
    vi.mocked(networking.teamUpdateCall).mockResolvedValue({ data: {}, team_id: "123" } as any);

    renderWithProviders(<TeamInfoView {...props} />);
    await waitFor(() => expect(screen.queryAllByText("Test Team").length).toBeGreaterThan(0));
    await user.click(screen.getByRole("tab", { name: "Settings" }));
    await user.click(await screen.findByRole("button", { name: /edit settings/i }));
    await screen.findByLabelText("Team Name");

    const payload = await save(user);

    expect(payload).toHaveProperty("team_alias");
    expect(wireBody(payload)).not.toHaveProperty("policies");
  });

  it("blocks the save on an empty team name and names the rule", async () => {
    const user = userEvent.setup({ delay: null });
    await openEditor(user);

    await user.clear(screen.getByLabelText("Team Name"));
    await user.click(screen.getByRole("button", { name: /save changes/i }));

    expect(await screen.findByText("Please input a team name")).toBeInTheDocument();
    expect(networking.teamUpdateCall).not.toHaveBeenCalled();
  });
});
