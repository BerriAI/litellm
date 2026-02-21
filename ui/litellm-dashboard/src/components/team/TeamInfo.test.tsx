import * as networking from "@/components/networking";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "../../../tests/test-utils";
import TeamInfoView from "./TeamInfo";

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
}));

vi.mock("@/components/utils/dataUtils", () => ({
  copyToClipboard: vi.fn().mockResolvedValue(true),
  formatNumberWithCommas: vi.fn((value: number) => value.toLocaleString()),
}));

vi.mock("@/app/(dashboard)/hooks/models/useModels", () => ({
  useAllProxyModels: vi.fn(),
}));

vi.mock("@/app/(dashboard)/hooks/teams/useTeams", () => ({
  useTeam: vi.fn(),
}));

vi.mock("@/app/(dashboard)/hooks/organizations/useOrganizations", () => ({
  useOrganization: vi.fn(),
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
    ) : null
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
    ) : null
  ),
}));

vi.mock("@/components/common_components/DeleteResourceModal", () => ({
  default: vi.fn(({ isOpen, onCancel, onOk }) =>
    isOpen ? (
      <div>
        <button onClick={onCancel}>Cancel</button>
        <button onClick={onOk}>Confirm Delete</button>
      </div>
    ) : null
  ),
}));

vi.mock("@/components/team/member_permissions", () => ({
  default: vi.fn(() => <div>Member Permissions</div>),
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

import { useAllProxyModels } from "@/app/(dashboard)/hooks/models/useModels";
import { useOrganization } from "@/app/(dashboard)/hooks/organizations/useOrganizations";
import { useTeam } from "@/app/(dashboard)/hooks/teams/useTeams";
import { useCurrentUser } from "@/app/(dashboard)/hooks/users/useCurrentUser";

const mockUseAllProxyModels = vi.mocked(useAllProxyModels);
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

  beforeEach(() => {
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

    vi.mocked(networking.getGuardrailsList).mockResolvedValue({ guardrails: [] });
    vi.mocked(networking.getPoliciesList).mockResolvedValue({ policies: [] });
    vi.mocked(networking.fetchMCPAccessGroups).mockResolvedValue([]);
    vi.mocked(networking.getTeamPermissionsCall).mockResolvedValue({
      all_available_permissions: [],
      team_member_permissions: [],
    });
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("should render", async () => {
    vi.mocked(networking.teamInfoCall).mockResolvedValue(createMockTeamData());

    renderWithProviders(<TeamInfoView {...defaultProps} />);

    await waitFor(() => {
      const teamNameElements = screen.queryAllByText("Test Team");
      expect(teamNameElements.length).toBeGreaterThan(0);
    });
  });

  it("should display loading state while fetching team data", () => {
    vi.mocked(networking.teamInfoCall).mockImplementation(() => new Promise(() => { }));

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
      })
    );

    renderWithProviders(<TeamInfoView {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText("Budget Status")).toBeInTheDocument();
    });
  });

  it("should display guardrails in overview when present", async () => {
    vi.mocked(networking.teamInfoCall).mockResolvedValue(
      createMockTeamData({
        guardrails: ["guardrail1", "guardrail2"],
      })
    );

    renderWithProviders(<TeamInfoView {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText("Guardrails")).toBeInTheDocument();
    });
  });

  it("should display policies in overview when present", async () => {
    vi.mocked(networking.teamInfoCall).mockResolvedValue(
      createMockTeamData({
        policies: ["policy1"],
      })
    );
    vi.mocked(networking.getPolicyInfoWithGuardrails).mockResolvedValue({
      resolved_guardrails: ["guardrail1"],
    });

    renderWithProviders(<TeamInfoView {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText("Policies")).toBeInTheDocument();
    });
  });

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

  it("should navigate to settings tab when clicked", async () => {
    const user = userEvent.setup();
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

  it("should open edit mode when edit button is clicked", async () => {
    const user = userEvent.setup();
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
    const user = userEvent.setup();
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

  it("should call onClose when back button is clicked", async () => {
    const user = userEvent.setup();
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
    const user = userEvent.setup();
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

  it("should disable secret manager settings for non-premium users", async () => {
    const user = userEvent.setup();
    vi.mocked(networking.teamInfoCall).mockResolvedValue(
      createMockTeamData({
        metadata: {
          secret_manager_settings: { provider: "aws", secret_id: "abc" },
        },
      })
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
      '{"namespace": "admin", "mount": "secret", "path_prefix": "litellm"}'
    );
    expect(secretField).toBeDisabled();
  });

  it("should allow premium users to edit secret manager settings", async () => {
    const user = userEvent.setup();
    vi.mocked(networking.teamInfoCall).mockResolvedValue(
      createMockTeamData({
        metadata: {
          secret_manager_settings: { provider: "aws", secret_id: "abc" },
        },
      })
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
      '{"namespace": "admin", "mount": "secret", "path_prefix": "litellm"}'
    );
    expect(secretField).not.toBeDisabled();
  });

  it("should add team member when form is submitted", async () => {
    const user = userEvent.setup();
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

  it("should display team member budget information when present", async () => {
    vi.mocked(networking.teamInfoCall).mockResolvedValue(
      createMockTeamData({
        team_member_budget_table: {
          max_budget: 500,
          budget_duration: "30d",
          tpm_limit: 5000,
          rpm_limit: 50,
        },
      })
    );

    renderWithProviders(<TeamInfoView {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText("Budget Status")).toBeInTheDocument();
    });
  });

  it("should display virtual keys information", async () => {
    vi.mocked(networking.teamInfoCall).mockResolvedValue({
      ...createMockTeamData(),
      keys: [
        { user_id: "user1", token: "key1" },
        { token: "key2" },
      ],
    });

    renderWithProviders(<TeamInfoView {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText("Virtual Keys")).toBeInTheDocument();
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
      })
    );

    renderWithProviders(<TeamInfoView {...defaultProps} />);

    await waitFor(() => {
      const teamNameElements = screen.queryAllByText("Test Team");
      expect(teamNameElements.length).toBeGreaterThan(0);
    });
  });

  it("should display soft budget in settings view when present", async () => {
    const user = userEvent.setup();
    vi.mocked(networking.teamInfoCall).mockResolvedValue(
      createMockTeamData({
        soft_budget: 500.75,
        max_budget: 1000,
      })
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
      <TeamInfoView {...defaultProps} editTeam={true} is_team_admin={false} is_proxy_admin={false} />
    );

    await waitFor(() => {
      const teamNameElements = screen.queryAllByText("Test Team");
      expect(teamNameElements.length).toBeGreaterThan(0);
    });

    expect(screen.getByText("Budget Status")).toBeInTheDocument();
  });

  it("should display soft budget alerting emails in settings view when present", async () => {
    const user = userEvent.setup();
    vi.mocked(networking.teamInfoCall).mockResolvedValue(
      createMockTeamData({
        metadata: {
          soft_budget_alerting_emails: ["alert1@test.com", "alert2@test.com"],
        },
      })
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
    const user = userEvent.setup();
    const accessGroupIds = ["ag-1", "ag-2"];
    vi.mocked(networking.teamInfoCall).mockResolvedValue(
      createMockTeamData({
        access_group_ids: accessGroupIds,
        models: ["gpt-4"],
      })
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
        })
      );
    });
  });
});
