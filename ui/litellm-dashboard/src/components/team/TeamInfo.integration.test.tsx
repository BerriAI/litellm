import { useTeamMetadataSchema } from "@/app/(dashboard)/hooks/teams/useTeamMetadataSchema";
import * as networking from "@/components/networking";
import { fireEvent, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "../../../tests/test-utils";
import { toast } from "@/lib/toast";
import type { Team } from "../key_team_helpers/key_list";
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

vi.mock("next/navigation", () => ({ useRouter: () => ({ push: vi.fn() }) }));

vi.mock("@/components/networking", () => ({
  serverRootPath: "",
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
  getPassThroughEndpointsCall: vi.fn().mockResolvedValue({ endpoints: [] }),
  fetchMCPServers: vi.fn().mockResolvedValue([]),
  fetchMCPToolsets: vi.fn().mockResolvedValue([]),
  listMCPTools: vi.fn().mockResolvedValue({ tools: [] }),
  vectorStoreListCall: vi.fn().mockResolvedValue({ data: [] }),
  getAgentsList: vi.fn().mockResolvedValue({ agents: [] }),
  getClaudeCodePluginsList: vi.fn().mockResolvedValue({ plugins: [], count: 0 }),
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

vi.mock("@/app/(dashboard)/hooks/uiSettings/useUISettings", () => ({
  useUISettings: vi.fn(),
}));

vi.mock("@/app/(dashboard)/hooks/models/useModels", () => ({
  useAllProxyModels: vi.fn(),
  useModelAccessGroupNames: vi.fn(() => new Set<string>()),
}));

vi.mock("@/app/(dashboard)/hooks/teams/useTeams", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/app/(dashboard)/hooks/teams/useTeams")>()),
  useTeam: vi.fn(),
}));

vi.mock("@/app/(dashboard)/hooks/organizations/useOrganizations", () => ({
  organizationKeys: { all: ["organizations"] },
  useOrganization: vi.fn(),
  useOrganizations: vi.fn().mockReturnValue({ data: [], isLoading: false }),
}));

vi.mock("@/app/(dashboard)/hooks/users/useCurrentUser", () => ({
  useCurrentUser: vi.fn(),
}));

vi.mock("@/app/(dashboard)/hooks/mcpServers/useMCPServers", () => ({
  useMCPServers: vi.fn(),
}));

vi.mock("@/app/(dashboard)/hooks/mcpServers/useMCPToolsets", () => ({
  useMCPToolsets: vi.fn(),
}));

vi.mock("@/components/mcp_server_management/MCPServerSelector", () => ({
  default: ({
    value,
    onChange,
  }: {
    value?: { servers: string[]; accessGroups: string[]; toolsets?: string[] };
    onChange: (next: { servers: string[]; accessGroups: string[]; toolsets: string[] }) => void;
  }) => (
    <>
      <button
        type="button"
        onClick={() =>
          onChange({ servers: [], accessGroups: value?.accessGroups ?? [], toolsets: value?.toolsets ?? [] })
        }
      >
        deselect all mcp servers
      </button>
      <button type="button" onClick={() => onChange({ servers: value?.servers ?? [], accessGroups: [], toolsets: [] })}>
        remove all access groups
      </button>
    </>
  ),
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
      <button type="button" onClick={() => onAliasUpdate({ "gpt-4o": "gpt-4" })}>
        Set Alias
      </button>
      <button type="button" onClick={() => onAliasUpdate({})}>
        Clear Aliases
      </button>
    </div>
  )),
}));

vi.mock("@/app/(dashboard)/hooks/accessGroups/useAccessGroups", () => ({
  useAccessGroups: vi.fn().mockReturnValue({
    data: [
      { access_group_id: "ag-1", access_group_name: "Group 1", access_mcp_server_ids: [] },
      { access_group_id: "ag-2", access_group_name: "Group 2", access_mcp_server_ids: [] },
    ],
    isLoading: false,
    isError: false,
  }),
}));

vi.mock("@/components/common_components/AccessGroupSelector", () => ({
  default: ({ value, onChange }: { value?: string[]; onChange?: (next: string[]) => void }) => (
    <button type="button" onClick={() => onChange?.((value ?? []).slice(1))}>
      remove first unified access group
    </button>
  ),
}));

vi.mock("@/app/(dashboard)/hooks/keys/useKeys", () => {
  const useKeysResult = {
    data: { keys: [], total_count: 0, current_page: 1, total_pages: 1 },
    isPending: false,
    isFetching: false,
    refetch: vi.fn(),
  };
  return { useKeys: vi.fn().mockReturnValue(useKeysResult) };
});

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
import { useMCPServers } from "@/app/(dashboard)/hooks/mcpServers/useMCPServers";
import { useMCPToolsets } from "@/app/(dashboard)/hooks/mcpServers/useMCPToolsets";
import { useAccessGroups } from "@/app/(dashboard)/hooks/accessGroups/useAccessGroups";
import { useUISettings } from "@/app/(dashboard)/hooks/uiSettings/useUISettings";

const mockUseAllProxyModels = vi.mocked(useAllProxyModels);
const mockUseKeys = vi.mocked(useKeys);
const mockUseTeam = vi.mocked(useTeam);
const mockUseOrganization = vi.mocked(useOrganization);
const mockUseCurrentUser = vi.mocked(useCurrentUser);
const mockUseMCPServers = vi.mocked(useMCPServers);
const mockUseMCPToolsets = vi.mocked(useMCPToolsets);
const mockUseAccessGroups = vi.mocked(useAccessGroups);
const mockUseUISettings = vi.mocked(useUISettings);

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
  } as unknown as ReturnType<typeof useAllProxyModels>);
  mockUseTeam.mockReturnValue({
    data: undefined,
    isLoading: false,
  } as unknown as ReturnType<typeof useTeam>);
  mockUseOrganization.mockReturnValue({
    data: undefined,
    isLoading: false,
  } as unknown as ReturnType<typeof useOrganization>);
  mockUseCurrentUser.mockReturnValue({
    data: { models: [] },
    isLoading: false,
  } as unknown as ReturnType<typeof useCurrentUser>);
  mockUseMCPServers.mockReturnValue({ data: [], isLoading: false, isError: false } as unknown as ReturnType<
    typeof useMCPServers
  >);
  mockUseMCPToolsets.mockReturnValue({ data: [], isLoading: false, isError: false } as unknown as ReturnType<
    typeof useMCPToolsets
  >);
  mockUseAccessGroups.mockReturnValue({
    data: [
      { access_group_id: "ag-1", access_group_name: "Group 1", access_mcp_server_ids: [] },
      { access_group_id: "ag-2", access_group_name: "Group 2", access_mcp_server_ids: [] },
    ],
    isLoading: false,
    isError: false,
  } as unknown as ReturnType<typeof useAccessGroups>);
  mockUseUISettings.mockReturnValue({
    data: { values: {} },
    isLoading: false,
  } as unknown as ReturnType<typeof useUISettings>);
  mockUseKeys.mockReturnValue({
    data: { keys: [], total_count: 0, current_page: 1, total_pages: 1 },
    isPending: false,
    isFetching: false,
    refetch: vi.fn(),
  } as unknown as ReturnType<typeof useKeys>);
  vi.mocked(useTeamMetadataSchema).mockReturnValue({
    data: [],
    isLoading: false,
  } as unknown as ReturnType<typeof useTeamMetadataSchema>);

  can.mockReturnValue(true);
  vi.mocked(networking.getGuardrailsList).mockResolvedValue({ guardrails: [] });
  vi.mocked(networking.getPoliciesList).mockResolvedValue({ policies: [] });
  vi.mocked(networking.fetchMCPAccessGroups).mockResolvedValue([]);
  vi.mocked(networking.getTeamPermissionsCall).mockResolvedValue({
    all_available_permissions: [],
    team_member_permissions: [],
  });
};
describe("TeamInfoView - member budget reset prompt", () => {
  const props = {
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

  const customBudgetMembership = (
    userId: string,
    maxBudget: number | null = 50,
    budgetDuration: string | null = null,
  ): TeamData["team_memberships"][number] => ({
    user_id: userId,
    team_id: "123",
    budget_id: `budget-${userId}`,
    budget_source: "custom",
    spend: 0,
    total_spend: 0,
    litellm_budget_table: {
      budget_id: `budget-${userId}`,
      soft_budget: null,
      max_budget: maxBudget,
      max_parallel_requests: null,
      tpm_limit: null,
      rpm_limit: null,
      model_max_budget: null,
      budget_duration: budgetDuration,
      budget_reset_at: null,
    },
  });

  const savedTeam: Team = {
    team_id: "123",
    team_alias: "Test Team",
    models: [],
    max_budget: null,
    budget_duration: null,
    tpm_limit: null,
    rpm_limit: null,
    organization_id: "org-1",
    created_at: "2024-01-01T00:00:00Z",
    keys: [],
    members_with_roles: [],
    spend: 0,
  };

  const submitNewDefault = async (user: ReturnType<typeof userEvent.setup>, input: HTMLElement, value: string) => {
    fireEvent.change(input, { target: { value } });
    await user.click(screen.getByRole("button", { name: /save changes/i }));
  };

  const memberships = (...rows: TeamData["team_memberships"]) => rows;

  const openEditor = async (user: ReturnType<typeof userEvent.setup>, rows: TeamData["team_memberships"]) => {
    const data = {
      ...createMockTeamData({
        team_member_budget_table: { max_budget: 10, budget_duration: null, tpm_limit: null, rpm_limit: null },
      }),
      team_memberships: rows,
    } as TeamData;
    vi.mocked(networking.teamInfoCall).mockResolvedValue(data);
    vi.mocked(networking.teamUpdateCall).mockResolvedValue({ data: savedTeam, team_id: "123" });

    const view = renderWithProviders(<TeamInfoView {...props} />);
    await waitFor(() => expect(screen.queryAllByText("Test Team").length).toBeGreaterThan(0));
    await user.click(screen.getByRole("tab", { name: "Settings" }));
    await user.click(await screen.findByRole("button", { name: /edit settings/i }));
    await user.click(screen.getByText("Team Member Settings"));
    return { input: await screen.findByLabelText("Default Budget (USD)"), view };
  };

  const savedBody = () => vi.mocked(networking.teamUpdateCall).mock.calls[0][1];
  const promptTitle = "Apply the $20.00 default to custom budgets?";

  beforeEach(() => {
    seedDefaultMocks();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("holds the save behind a prompt that counts members on each side of the new default", async () => {
    const user = userEvent.setup({ delay: null });
    const { input } = await openEditor(
      user,
      memberships(
        customBudgetMembership("user-low", 5),
        customBudgetMembership("user-high", 50),
        customBudgetMembership("user-weekly", 5, "7d"),
      ),
    );

    await submitNewDefault(user, input, "20");

    expect(await screen.findByText(promptTitle)).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: "Raise 1 member below $20.00" })).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: "Lower 1 member above $20.00" })).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: "Raise and lower all 2 members" })).toBeInTheDocument();
    expect(screen.getByText("1 member resets on a different schedule and won't change")).toBeInTheDocument();
    expect(networking.teamUpdateCall).not.toHaveBeenCalled();
  });

  it("keeps every custom budget when saved with the default choice", async () => {
    const user = userEvent.setup({ delay: null });
    const { input } = await openEditor(user, memberships(customBudgetMembership("user-high")));

    await submitNewDefault(user, input, "20");
    await user.click(await screen.findByRole("button", { name: "Save" }));

    await waitFor(() => expect(networking.teamUpdateCall).toHaveBeenCalledTimes(1));
    expect(savedBody()).toMatchObject({ team_member_budget: 20, team_member_budget_backfill: "none" });
    await waitFor(() => expect(toast.success).toHaveBeenCalledWith("Team settings updated successfully"));
    expect(screen.queryByText(promptTitle)).not.toBeInTheDocument();
  });

  it("sends the chosen direction in the same save and reports what the server moved", async () => {
    const user = userEvent.setup({ delay: null });
    const { input } = await openEditor(user, memberships(customBudgetMembership("user-high")));
    vi.mocked(networking.teamUpdateCall).mockResolvedValue({
      data: savedTeam,
      team_id: "123",
      team_member_budget_backfill_result: { moved_to_default: 1, skipped_other_reset_period: 0 },
    });

    await submitNewDefault(user, input, "20");
    await user.click(await screen.findByRole("radio", { name: "Lower 1 member above $20.00" }));
    await user.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() => expect(networking.teamUpdateCall).toHaveBeenCalledTimes(1));
    expect(savedBody()).toMatchObject({ team_member_budget: 20, team_member_budget_backfill: "lower_to_default" });
    await waitFor(() => expect(toast.success).toHaveBeenCalledWith("Moved 1 member to the $20.00 default"));
  });

  it("returns to the prompt without its own toast when the save fails", async () => {
    const user = userEvent.setup({ delay: null });
    const { input } = await openEditor(user, memberships(customBudgetMembership("user-high")));
    vi.mocked(networking.teamUpdateCall).mockRejectedValueOnce(new Error("save failed"));

    await submitNewDefault(user, input, "20");
    await user.click(await screen.findByRole("button", { name: "Save" }));

    await waitFor(() => expect(networking.teamUpdateCall).toHaveBeenCalledTimes(1));
    expect(await screen.findByRole("button", { name: "Save" })).toBeEnabled();
    expect(toast.success).not.toHaveBeenCalled();
  });

  it("aborts the save entirely on cancel", async () => {
    const user = userEvent.setup({ delay: null });
    const { input } = await openEditor(user, memberships(customBudgetMembership("user-high")));

    await submitNewDefault(user, input, "20");
    await user.click(await screen.findByRole("button", { name: "Cancel" }));

    await waitFor(() => expect(screen.queryByText(promptTitle)).not.toBeInTheDocument());
    expect(networking.teamUpdateCall).not.toHaveBeenCalled();
  });

  it.each([
    ["no member has a custom budget", memberships(), "20"],
    ["a custom member's cap is null", memberships(customBudgetMembership("user-limits-only", null)), "20"],
    ["the only custom member resets on another schedule", memberships(customBudgetMembership("u", 5, "7d")), "20"],
    ["the only custom member already matches", memberships(customBudgetMembership("user-equal", 20)), "20"],
    ["the default is resubmitted unchanged", memberships(customBudgetMembership("user-high")), "10"],
  ])("saves directly when %s", async (_case, rows, value) => {
    const user = userEvent.setup({ delay: null });
    const { input } = await openEditor(user, rows);

    await submitNewDefault(user, input, value);

    await waitFor(() => expect(networking.teamUpdateCall).toHaveBeenCalledTimes(1));
    expect(savedBody().team_member_budget).toBe(Number(value));
    expect(savedBody()).not.toHaveProperty("team_member_budget_backfill");
    expect(screen.queryByText(/default to custom budgets/)).not.toBeInTheDocument();
  });

  it("dismisses a pending prompt when the selected team changes", async () => {
    const user = userEvent.setup({ delay: null });
    const { input, view } = await openEditor(user, memberships(customBudgetMembership("user-high")));

    await submitNewDefault(user, input, "20");
    expect(await screen.findByText(promptTitle)).toBeInTheDocument();

    view.rerender(<TeamInfoView {...props} teamId="456" />);

    await waitFor(() => expect(screen.queryByText(promptTitle)).not.toBeInTheDocument());
    expect(networking.teamUpdateCall).not.toHaveBeenCalled();
  });

  it("omits the default member model budget when it has not changed", async () => {
    const user = userEvent.setup({ delay: null });
    const storedBudget = { "gpt-4": { budget_limit: 3, time_period: "30d" } };
    const data = createMockTeamData({
      models: ["gpt-4"],
      team_member_budget_table: {
        max_budget: 10,
        budget_duration: null,
        tpm_limit: null,
        rpm_limit: null,
        model_max_budget: storedBudget,
      },
    });
    vi.mocked(networking.teamInfoCall).mockResolvedValue(data);
    vi.mocked(networking.teamUpdateCall).mockResolvedValue({ data: savedTeam, team_id: "123" });

    renderWithProviders(<TeamInfoView {...props} premiumUser />);
    await waitFor(() => expect(screen.queryAllByText("Test Team").length).toBeGreaterThan(0));
    await user.click(screen.getByRole("tab", { name: "Settings" }));
    await user.click(await screen.findByRole("button", { name: /edit settings/i }));
    await user.click(screen.getByText("Team Member Settings"));
    await screen.findByPlaceholderText("Max spend ($)");
    await user.click(screen.getByRole("button", { name: /save changes/i }));

    await waitFor(() => expect(networking.teamUpdateCall).toHaveBeenCalled());
    expect(vi.mocked(networking.teamUpdateCall).mock.calls[0][1]).not.toHaveProperty("team_member_model_max_budget");
  });

  it("sends the default member model budget after an edit", async () => {
    const user = userEvent.setup({ delay: null });
    const storedBudget = { "gpt-4": { budget_limit: 3, time_period: "30d" } };
    const data = createMockTeamData({
      models: ["gpt-4"],
      team_member_budget_table: {
        max_budget: 10,
        budget_duration: null,
        tpm_limit: null,
        rpm_limit: null,
        model_max_budget: storedBudget,
      },
    });
    vi.mocked(networking.teamInfoCall).mockResolvedValue(data);
    vi.mocked(networking.teamUpdateCall).mockResolvedValue({ data: savedTeam, team_id: "123" });

    renderWithProviders(<TeamInfoView {...props} premiumUser />);
    await waitFor(() => expect(screen.queryAllByText("Test Team").length).toBeGreaterThan(0));
    await user.click(screen.getByRole("tab", { name: "Settings" }));
    await user.click(await screen.findByRole("button", { name: /edit settings/i }));
    await user.click(screen.getByText("Team Member Settings"));
    fireEvent.change(await screen.findByPlaceholderText("Max spend ($)"), { target: { value: "7" } });
    await user.click(screen.getByRole("button", { name: /save changes/i }));

    await waitFor(() => expect(networking.teamUpdateCall).toHaveBeenCalled());
    expect(vi.mocked(networking.teamUpdateCall).mock.calls[0][1].team_member_model_max_budget).toStrictEqual({
      "gpt-4": { budget_limit: 7, time_period: "30d" },
    });
  });

  it("offers access-group models in the default member budget picker", async () => {
    const user = userEvent.setup({ delay: null });
    vi.mocked(networking.teamInfoCall).mockResolvedValue(
      createMockTeamData({
        models: ["gpt-4o"],
        access_group_models: ["claude-haiku-4-5"],
      }) as TeamData,
    );

    renderWithProviders(<TeamInfoView {...props} userModels={["gpt-4o", "claude-haiku-4-5"]} premiumUser />);
    await waitFor(() => expect(screen.queryAllByText("Test Team").length).toBeGreaterThan(0));
    await user.click(screen.getByRole("tab", { name: "Settings" }));
    await user.click(await screen.findByRole("button", { name: /edit settings/i }));
    await user.click(screen.getByText("Team Member Settings"));
    await user.click(screen.getAllByRole("button", { name: /add model budget/i })[0]);
    await user.click(screen.getByPlaceholderText("Select model"));

    expect(await screen.findByRole("option", { name: "claude-haiku-4-5" })).toBeInTheDocument();
  });

  it("offers user models in the default member budget picker when the team model list is empty", async () => {
    const user = userEvent.setup({ delay: null });
    vi.mocked(networking.teamInfoCall).mockResolvedValue(createMockTeamData({ models: [] }) as TeamData);

    renderWithProviders(<TeamInfoView {...props} userModels={["gpt-4o", "claude-haiku-4-5"]} premiumUser />);
    await waitFor(() => expect(screen.queryAllByText("Test Team").length).toBeGreaterThan(0));
    await user.click(screen.getByRole("tab", { name: "Settings" }));
    await user.click(await screen.findByRole("button", { name: /edit settings/i }));
    await user.click(screen.getByText("Team Member Settings"));
    await user.click(screen.getAllByRole("button", { name: /add model budget/i })[0]);
    await user.click(screen.getByPlaceholderText("Select model"));

    expect(await screen.findByRole("option", { name: "gpt-4o" })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "claude-haiku-4-5" })).toBeInTheDocument();
  });
});

describe("TeamInfoView - model rate limit picker", () => {
  const props = {
    teamId: "123",
    onUpdate: vi.fn(),
    onClose: vi.fn(),
    accessToken: "test-token",
    is_team_admin: true,
    is_proxy_admin: true,
    editTeam: false,
  };

  beforeEach(() => {
    seedDefaultMocks();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("offers access-group models, not no-default-models, in the model rate limit picker", async () => {
    const user = userEvent.setup({ delay: null });
    vi.mocked(networking.teamInfoCall).mockResolvedValue(
      createMockTeamData({
        models: ["no-default-models"],
        access_group_models: ["claude-haiku-4-5"],
      }) as TeamData,
    );

    renderWithProviders(<TeamInfoView {...props} userModels={["gpt-4o", "claude-haiku-4-5"]} premiumUser />);
    await waitFor(() => expect(screen.queryAllByText("Test Team").length).toBeGreaterThan(0));
    await user.click(screen.getByRole("tab", { name: "Settings" }));
    await user.click(await screen.findByRole("button", { name: /edit settings/i }));
    await user.click(screen.getByRole("button", { name: /add model limit/i }));
    await user.click(screen.getByPlaceholderText("Select model"));

    expect(await screen.findByRole("option", { name: "claude-haiku-4-5" })).toBeInTheDocument();
    expect(screen.queryByRole("option", { name: "no-default-models" })).not.toBeInTheDocument();
    expect(screen.queryByRole("option", { name: "gpt-4o" })).not.toBeInTheDocument();
  });
});
