import { useTeamMetadataSchema } from "@/app/(dashboard)/hooks/teams/useTeamMetadataSchema";
import * as networking from "@/components/networking";
import { fireEvent, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "../../../tests/test-utils";
import { toast } from "@/lib/toast";
import type { Team } from "../key_team_helpers/key_list";
import { MAX_BULK_TEAM_MEMBER_BUDGET_UPDATES } from "./memberBudgetReset";
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

const { bulkUpdatePOST } = vi.hoisted(() => ({ bulkUpdatePOST: vi.fn() }));
vi.mock("@/lib/http/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/http/api")>();
  return { ...actual, fetchClient: { ...actual.fetchClient, POST: bulkUpdatePOST } };
});

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
}));

vi.mock("@/app/(dashboard)/hooks/teams/useTeams", () => ({
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

describe("TeamInfoView - pooled budget", () => {
  beforeEach(seedDefaultMocks);
  afterEach(() => vi.clearAllMocks());

  const openEditor = async (maxBudget: number | null) => {
    const overrides = {
      max_budget: maxBudget,
      spend: 12,
      budget_duration: "30d",
      team_member_budget_table: { max_budget: 10, budget_duration: "7d", tpm_limit: null, rpm_limit: null },
    };
    const data = createMockTeamData(overrides);
    vi.mocked(networking.teamInfoCall).mockResolvedValue(data);
    renderWithProviders(
      <TeamInfoView
        teamId="123"
        accessToken="test-token"
        is_team_admin
        is_proxy_admin
        userModels={[]}
        editTeam={false}
        onUpdate={vi.fn()}
        onClose={vi.fn()}
      />,
    );
    fireEvent.click(await screen.findByRole("tab", { name: "Settings" }));
    fireEvent.click(await screen.findByRole("button", { name: /edit settings/i }));
    return await screen.findByRole("switch", { name: "Pooled budget" });
  };

  const save = async () => {
    fireEvent.click(screen.getByRole("button", { name: /save changes/i }));
    await waitFor(() => expect(networking.teamUpdateCall).toHaveBeenCalledTimes(1));
    return vi.mocked(networking.teamUpdateCall).mock.calls[0][1] as Record<string, unknown>;
  };

  it.each([0, 100])("keeps an existing pooled budget of %s enabled on an unrelated save", async (amount) => {
    expect(await openEditor(amount)).toBeChecked();
    expect(screen.getByLabelText("Pool amount (USD)")).toHaveValue(amount);
    fireEvent.change(screen.getByLabelText("Team Name"), { target: { value: "Renamed Team" } });

    expect(await save()).toMatchObject({ team_alias: "Renamed Team", max_budget: amount, budget_duration: "30d" });
  });

  it("removes the shared cap explicitly while retaining member limits and reset periods", async () => {
    fireEvent.click(await openEditor(100));
    fireEvent.click(screen.getByText("Team Member Settings"));
    expect(await screen.findByLabelText("Default Budget (USD)")).toHaveValue(10);
    expect(screen.queryByLabelText("Pool amount (USD)")).not.toBeInTheDocument();

    const payload = await save();
    const expected = {
      max_budget: null,
      budget_duration: "30d",
      team_member_budget: 10,
      team_member_budget_duration: "7d",
    };
    expect(payload).toMatchObject(expected);
    expect(payload).not.toHaveProperty("spend");
    expect(payload).not.toHaveProperty("budget_reset_at");
  });

  it("requires an amount when opting an uncapped team into a pooled budget", async () => {
    fireEvent.click(await openEditor(null));
    fireEvent.click(screen.getByRole("button", { name: /save changes/i }));
    expect(await screen.findByText("Enter a pooled budget amount")).toBeInTheDocument();
    expect(networking.teamUpdateCall).not.toHaveBeenCalled();

    fireEvent.change(screen.getByLabelText("Pool amount (USD)"), { target: { value: "25" } });
    expect(await save()).toMatchObject({ max_budget: 25 });
  });
});

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
      budget_duration: null,
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

  const openEditorWithCustomMembers = async (
    user: ReturnType<typeof userEvent.setup>,
    userIds: string[] = ["user-custom"],
    maxBudget: number | null = 50,
  ) => {
    const data = {
      ...createMockTeamData({
        team_member_budget_table: { max_budget: 10, budget_duration: null, tpm_limit: null, rpm_limit: null },
      }),
      team_memberships: userIds.map((id) => customBudgetMembership(id, maxBudget)),
    } as TeamData;
    vi.mocked(networking.teamInfoCall).mockResolvedValue(data);
    vi.mocked(networking.teamUpdateCall).mockResolvedValue({ data: savedTeam, team_id: "123" });

    renderWithProviders(<TeamInfoView {...props} />);
    await waitFor(() => expect(screen.queryAllByText("Test Team").length).toBeGreaterThan(0));
    await user.click(screen.getByRole("tab", { name: "Settings" }));
    await user.click(await screen.findByRole("button", { name: /edit settings/i }));
    await user.click(screen.getByText("Team Member Settings"));
    return await screen.findByLabelText("Default Budget (USD)");
  };

  const submitNewDefault = async (user: ReturnType<typeof userEvent.setup>, input: HTMLElement, value: string) => {
    fireEvent.change(input, { target: { value } });
    await user.click(screen.getByRole("button", { name: /save changes/i }));
  };

  beforeEach(() => {
    seedDefaultMocks();
    bulkUpdatePOST.mockResolvedValue({ data: { data: [] } });
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("holds the save behind the prompt when the default changes while a member has a custom budget", async () => {
    const user = userEvent.setup({ delay: null });
    const input = await openEditorWithCustomMembers(user);

    await submitNewDefault(user, input, "20");

    expect(await screen.findByText("Reset member budgets?")).toBeInTheDocument();
    expect(screen.getByText(/1 member has a custom budget/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Reset to $20.00" })).toBeInTheDocument();
    expect(networking.teamUpdateCall).not.toHaveBeenCalled();
    expect(bulkUpdatePOST).not.toHaveBeenCalled();
  });

  it("saves the new default and leaves custom budgets alone on keep", async () => {
    const user = userEvent.setup({ delay: null });
    const input = await openEditorWithCustomMembers(user);

    await submitNewDefault(user, input, "20");
    await user.click(await screen.findByRole("button", { name: "Keep custom budget" }));

    await waitFor(() => expect(networking.teamUpdateCall).toHaveBeenCalled());
    expect(vi.mocked(networking.teamUpdateCall).mock.calls[0][1].team_member_budget).toBe(20);
    expect(bulkUpdatePOST).not.toHaveBeenCalled();
    await waitFor(() => expect(screen.queryByText("Reset member budgets?")).not.toBeInTheDocument());
  });

  it("saves the new default then clears each custom budget through the bulk endpoint on reset all", async () => {
    const user = userEvent.setup({ delay: null });
    const input = await openEditorWithCustomMembers(user, ["user-a", "user-b"]);
    bulkUpdatePOST.mockResolvedValue({
      data: {
        data: [
          { success: true, user_id: "user-a" },
          { success: true, user_id: "user-b" },
        ],
      },
    });

    await submitNewDefault(user, input, "20");
    await user.click(await screen.findByRole("button", { name: "Reset all to $20.00" }));

    await waitFor(() => expect(bulkUpdatePOST).toHaveBeenCalledTimes(1));
    expect(networking.teamUpdateCall).toHaveBeenCalled();
    expect(vi.mocked(networking.teamUpdateCall).mock.calls[0][1].team_member_budget).toBe(20);
    expect(bulkUpdatePOST).toHaveBeenCalledWith("/management/v1/teams/{team_id}/members/bulk_update", {
      params: { path: { team_id: "123" } },
      body: {
        members: [
          { user_id: "user-a", max_budget_in_team: null },
          { user_id: "user-b", max_budget_in_team: null },
        ],
      },
    });
    await waitFor(() => expect(toast.success).toHaveBeenCalledWith("Reset 2 member budgets to the team default"));
  });

  it("keeps the prompt mounted while the member-budget reset is in flight", async () => {
    const user = userEvent.setup({ delay: null });
    const input = await openEditorWithCustomMembers(user);
    const bulk = Promise.withResolvers<{ data: { data: { success: boolean; user_id: string }[] } }>();
    vi.mocked(networking.teamInfoCall).mockImplementationOnce(() => new Promise(() => {}));
    bulkUpdatePOST.mockImplementationOnce(() => bulk.promise);

    await submitNewDefault(user, input, "20");
    await user.click(await screen.findByRole("button", { name: "Reset to $20.00" }));

    await waitFor(() => expect(bulkUpdatePOST).toHaveBeenCalled());
    expect(screen.getByText("Reset member budgets?")).toBeInTheDocument();
    expect(screen.queryByText("Loading...")).not.toBeInTheDocument();

    bulk.resolve({ data: { data: [{ success: true, user_id: "user-custom" }] } });
    await waitFor(() => expect(screen.queryByText("Reset member budgets?")).not.toBeInTheDocument());
  });

  it("offers a retry after a failed reset and never re-saves the team", async () => {
    const user = userEvent.setup({ delay: null });
    const input = await openEditorWithCustomMembers(user);
    bulkUpdatePOST
      .mockRejectedValueOnce(new Error("bulk update down"))
      .mockResolvedValueOnce({ data: { data: [{ success: true, user_id: "user-custom" }] } });

    await submitNewDefault(user, input, "20");
    await user.click(await screen.findByRole("button", { name: "Reset to $20.00" }));

    await waitFor(() =>
      expect(toast.fromError).toHaveBeenCalledWith("Team updated, but member budgets could not be reset"),
    );
    expect(screen.getByText("Reset member budgets?")).toBeInTheDocument();
    expect(screen.queryByText("Loading...")).not.toBeInTheDocument();
    await waitFor(() => expect(vi.mocked(networking.teamInfoCall).mock.calls.length).toBeGreaterThan(1));

    await user.click(screen.getByRole("button", { name: "Retry reset" }));

    await waitFor(() => expect(bulkUpdatePOST).toHaveBeenCalledTimes(2));
    expect(vi.mocked(networking.teamUpdateCall).mock.calls).toHaveLength(1);
    await waitFor(() => expect(toast.success).toHaveBeenCalledWith("Reset 1 member budget to the team default"));
    await waitFor(() => expect(screen.queryByText("Reset member budgets?")).not.toBeInTheDocument());
  });

  it("returns to the prompt without its own toast when the team save fails", async () => {
    const user = userEvent.setup({ delay: null });
    const input = await openEditorWithCustomMembers(user);
    vi.mocked(networking.teamUpdateCall).mockRejectedValueOnce(new Error("save failed"));

    await submitNewDefault(user, input, "20");
    await user.click(await screen.findByRole("button", { name: "Reset to $20.00" }));

    expect(await screen.findByRole("button", { name: "Reset to $20.00" })).toBeEnabled();
    expect(bulkUpdatePOST).not.toHaveBeenCalled();
    expect(toast.fromError).not.toHaveBeenCalled();
    expect(toast.error).not.toHaveBeenCalled();
    expect(screen.queryByText("Loading...")).not.toBeInTheDocument();
  });

  it("reports how many budgets were already reset when a later batch fails", async () => {
    const user = userEvent.setup({ delay: null });
    const userIds = Array.from({ length: MAX_BULK_TEAM_MEMBER_BUDGET_UPDATES + 1 }, (_, i) => `user-${i}`);
    const input = await openEditorWithCustomMembers(user, userIds);
    bulkUpdatePOST
      .mockResolvedValueOnce({
        data: {
          data: userIds.slice(0, MAX_BULK_TEAM_MEMBER_BUDGET_UPDATES).map((user_id) => ({ success: true, user_id })),
        },
      })
      .mockRejectedValueOnce(new Error("second batch down"))
      .mockResolvedValueOnce({
        data: { data: [{ success: true, user_id: `user-${MAX_BULK_TEAM_MEMBER_BUDGET_UPDATES}` }] },
      });

    await submitNewDefault(user, input, "20");
    await user.click(await screen.findByRole("button", { name: "Reset all to $20.00" }));

    await waitFor(() =>
      expect(toast.error).toHaveBeenCalledWith(
        `Reset ${MAX_BULK_TEAM_MEMBER_BUDGET_UPDATES} of ${MAX_BULK_TEAM_MEMBER_BUDGET_UPDATES + 1} member budgets; the rest could not be reset`,
      ),
    );
    expect(bulkUpdatePOST).toHaveBeenCalledTimes(2);

    await user.click(screen.getByRole("button", { name: "Retry reset" }));

    await waitFor(() => expect(bulkUpdatePOST).toHaveBeenCalledTimes(3));
    expect(vi.mocked(bulkUpdatePOST).mock.calls[2][1]).toEqual({
      params: { path: { team_id: "123" } },
      body: {
        members: [{ user_id: `user-${MAX_BULK_TEAM_MEMBER_BUDGET_UPDATES}`, max_budget_in_team: null }],
      },
    });
    expect(vi.mocked(networking.teamUpdateCall).mock.calls).toHaveLength(1);
    await waitFor(() =>
      expect(toast.success).toHaveBeenCalledWith(
        `Reset ${MAX_BULK_TEAM_MEMBER_BUDGET_UPDATES + 1} member budgets to the team default`,
      ),
    );
  });

  it("surfaces a failure toast when some members cannot be reset", async () => {
    const user = userEvent.setup({ delay: null });
    const input = await openEditorWithCustomMembers(user, ["user-a", "user-b"]);
    bulkUpdatePOST.mockResolvedValue({
      data: {
        data: [
          { success: true, user_id: "user-a" },
          { success: false, user_id: "user-b", error: "no such member" },
        ],
      },
    });

    await submitNewDefault(user, input, "20");
    await user.click(await screen.findByRole("button", { name: "Reset all to $20.00" }));

    await waitFor(() =>
      expect(toast.error).toHaveBeenCalledWith("Team updated, but 1 member budget could not be reset"),
    );
  });

  it("aborts the save entirely on cancel", async () => {
    const user = userEvent.setup({ delay: null });
    const input = await openEditorWithCustomMembers(user);

    await submitNewDefault(user, input, "20");
    await user.click(await screen.findByRole("button", { name: "Cancel" }));

    await waitFor(() => expect(screen.queryByText("Reset member budgets?")).not.toBeInTheDocument());
    expect(networking.teamUpdateCall).not.toHaveBeenCalled();
    expect(bulkUpdatePOST).not.toHaveBeenCalled();
  });

  it("saves directly when no member carries a custom budget", async () => {
    const user = userEvent.setup({ delay: null });
    const input = await openEditorWithCustomMembers(user, []);

    await submitNewDefault(user, input, "20");

    await waitFor(() => expect(networking.teamUpdateCall).toHaveBeenCalled());
    expect(vi.mocked(networking.teamUpdateCall).mock.calls[0][1].team_member_budget).toBe(20);
    expect(screen.queryByText("Reset member budgets?")).not.toBeInTheDocument();
    expect(bulkUpdatePOST).not.toHaveBeenCalled();
  });

  it("saves directly when the default is resubmitted unchanged", async () => {
    const user = userEvent.setup({ delay: null });
    await openEditorWithCustomMembers(user);

    await user.click(screen.getByRole("button", { name: /save changes/i }));

    await waitFor(() => expect(networking.teamUpdateCall).toHaveBeenCalled());
    expect(vi.mocked(networking.teamUpdateCall).mock.calls[0][1].team_member_budget).toBe(10);
    expect(screen.queryByText("Reset member budgets?")).not.toBeInTheDocument();
    expect(bulkUpdatePOST).not.toHaveBeenCalled();
  });

  it("dismisses a pending prompt when the selected team changes", async () => {
    const user = userEvent.setup({ delay: null });
    const data = {
      ...createMockTeamData({
        team_member_budget_table: { max_budget: 10, budget_duration: null, tpm_limit: null, rpm_limit: null },
      }),
      team_memberships: [customBudgetMembership("user-custom")],
    } as TeamData;
    vi.mocked(networking.teamInfoCall).mockResolvedValue(data);
    vi.mocked(networking.teamUpdateCall).mockResolvedValue({ data: savedTeam, team_id: "123" });

    const { rerender } = renderWithProviders(<TeamInfoView {...props} />);
    await waitFor(() => expect(screen.queryAllByText("Test Team").length).toBeGreaterThan(0));
    await user.click(screen.getByRole("tab", { name: "Settings" }));
    await user.click(await screen.findByRole("button", { name: /edit settings/i }));
    await user.click(screen.getByText("Team Member Settings"));
    const input = await screen.findByLabelText("Default Budget (USD)");

    await submitNewDefault(user, input, "20");
    expect(await screen.findByText("Reset member budgets?")).toBeInTheDocument();

    rerender(<TeamInfoView {...props} teamId="456" />);

    await waitFor(() => expect(screen.queryByText("Reset member budgets?")).not.toBeInTheDocument());
    expect(bulkUpdatePOST).not.toHaveBeenCalled();
  });

  it("does not reopen the dialog or refetch the old team when the team changes mid-reset", async () => {
    const user = userEvent.setup({ delay: null });
    const data = {
      ...createMockTeamData({
        team_member_budget_table: { max_budget: 10, budget_duration: null, tpm_limit: null, rpm_limit: null },
      }),
      team_memberships: [customBudgetMembership("user-custom")],
    } as TeamData;
    vi.mocked(networking.teamInfoCall).mockResolvedValue(data);
    vi.mocked(networking.teamUpdateCall).mockResolvedValue({ data: savedTeam, team_id: "123" });
    const bulkDone = Promise.withResolvers<{ data: { data: { success: boolean; user_id: string }[] } }>();
    bulkUpdatePOST.mockReturnValue(bulkDone.promise);

    const { rerender } = renderWithProviders(<TeamInfoView {...props} />);
    await waitFor(() => expect(screen.queryAllByText("Test Team").length).toBeGreaterThan(0));
    await user.click(screen.getByRole("tab", { name: "Settings" }));
    await user.click(await screen.findByRole("button", { name: /edit settings/i }));
    await user.click(screen.getByText("Team Member Settings"));
    const input = await screen.findByLabelText("Default Budget (USD)");

    await submitNewDefault(user, input, "20");
    await user.click(await screen.findByRole("button", { name: "Reset to $20.00" }));
    await waitFor(() => expect(bulkUpdatePOST).toHaveBeenCalled());

    const infoCallsBeforeSwitch = vi.mocked(networking.teamInfoCall).mock.calls.length;
    rerender(<TeamInfoView {...props} teamId="456" />);
    bulkDone.resolve({ data: { data: [{ success: true, user_id: "user-custom" }] } });

    await waitFor(() => expect(screen.queryByText("Reset member budgets?")).not.toBeInTheDocument());
    await waitFor(() =>
      expect(
        vi
          .mocked(networking.teamInfoCall)
          .mock.calls.slice(infoCallsBeforeSwitch)
          .some((call) => call[1] === "456"),
      ).toBe(true),
    );
    const infoCallsAfterSwitch = vi.mocked(networking.teamInfoCall).mock.calls.slice(infoCallsBeforeSwitch);
    expect(infoCallsAfterSwitch.every((call) => call[1] === "456")).toBe(true);
    expect(screen.queryByRole("button", { name: "Retry reset" })).not.toBeInTheDocument();
  });

  it("saves directly when a custom member's cap is null and already inherits the default", async () => {
    const user = userEvent.setup({ delay: null });
    const input = await openEditorWithCustomMembers(user, ["user-limits-only"], null);

    await submitNewDefault(user, input, "20");

    await waitFor(() => expect(networking.teamUpdateCall).toHaveBeenCalled());
    expect(vi.mocked(networking.teamUpdateCall).mock.calls[0][1].team_member_budget).toBe(20);
    expect(screen.queryByText("Reset member budgets?")).not.toBeInTheDocument();
    expect(bulkUpdatePOST).not.toHaveBeenCalled();
  });
});
