import { useTeamMetadataSchema } from "@/app/(dashboard)/hooks/teams/useTeamMetadataSchema";
import * as networking from "@/components/networking";
import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "../../../tests/test-utils";
import { toast } from "@/lib/toast";
import type { Team } from "../key_team_helpers/key_list";
import { MAX_BULK_TEAM_MEMBER_BUDGET_UPDATES } from "./memberBudgetReset";
import TeamInfoView, { type TeamData } from "./TeamInfo";
import LanguageSelector from "@/components/LanguageSelector";
import { LanguageProvider } from "@/contexts/LanguageContext";
import { Toaster } from "@/components/ui/sonner";
import { ApiError } from "@/lib/http/client";
import type { ComponentProps } from "react";

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
  default: vi.fn(
    ({ setIsAddMemberModalVisible, setSelectedEditMember, setIsEditMemberModalVisible, handleMemberDelete }) => (
      <div>
        <button onClick={() => setIsAddMemberModalVisible(true)}>Add Member</button>
        <button
          onClick={() => {
            setSelectedEditMember({ user_email: "edit@test.com", user_id: "edit-user", role: "user" });
            setIsEditMemberModalVisible(true);
          }}
        >
          Edit Member
        </button>
        <button
          onClick={() => handleMemberDelete({ user_email: "delete@test.com", user_id: "delete-user", role: "user" })}
        >
          Delete Member
        </button>
      </div>
    ),
  ),
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

  const expectBudgetNotice = async (kind: "success" | "error" | "fromError", message: string) => {
    await waitFor(() => expect(toast[kind]).toHaveBeenCalled());
    const content =
      kind === "fromError"
        ? vi.mocked(toast.fromError).mock.calls.at(-1)?.[1]?.description
        : vi.mocked(toast[kind]).mock.calls.at(-1)?.[0];
    // 保留团队保存载荷断言，通知改按真实显示文本验证，不绑定参数实现。
    renderWithProviders(<>{content}</>);
    expect(screen.getByText(message)).toBeVisible();
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
    await expectBudgetNotice("success", "Reset 2 member budgets to the team default");
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

    await expectBudgetNotice("fromError", "Team updated, but member budgets could not be reset");
    expect(screen.getByText("Reset member budgets?")).toBeInTheDocument();
    expect(screen.queryByText("Loading...")).not.toBeInTheDocument();
    await waitFor(() => expect(vi.mocked(networking.teamInfoCall).mock.calls.length).toBeGreaterThan(1));

    await user.click(screen.getByRole("button", { name: "Retry reset" }));

    await waitFor(() => expect(bulkUpdatePOST).toHaveBeenCalledTimes(2));
    expect(vi.mocked(networking.teamUpdateCall).mock.calls).toHaveLength(1);
    await expectBudgetNotice("success", "Reset 1 member budget to the team default");
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

    await expectBudgetNotice(
      "error",
      `Reset ${MAX_BULK_TEAM_MEMBER_BUDGET_UPDATES} of ${MAX_BULK_TEAM_MEMBER_BUDGET_UPDATES + 1} member budgets; the rest could not be reset`,
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
    await expectBudgetNotice(
      "success",
      `Reset ${MAX_BULK_TEAM_MEMBER_BUDGET_UPDATES + 1} member budgets to the team default`,
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

    await expectBudgetNotice("error", "Team updated, but 1 member budget could not be reset");
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

describe("TeamInfoView - 校验语言与草稿", () => {
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

  beforeEach(() => {
    localStorage.clear();
    authState.userRole = "Admin";
    seedDefaultMocks();
    vi.mocked(networking.teamInfoCall).mockResolvedValue(createMockTeamData());
    vi.mocked(networking.teamUpdateCall).mockResolvedValue({ team_id: "123" });
  });

  afterEach(() => {
    authState.userRole = "Admin";
    localStorage.clear();
    vi.clearAllMocks();
  });

  const openLanguageEditor = async (user: ReturnType<typeof userEvent.setup>, defaultLanguage = "en") => {
    renderWithProviders(
      <LanguageProvider defaultLanguage={defaultLanguage} userId="team-validation" storageScope="team-validation">
        <LanguageSelector />
        <TeamInfoView {...props} />
      </LanguageProvider>,
    );
    await user.click(await screen.findByRole("tab", { name: defaultLanguage === "en" ? "Settings" : "设置" }));
    await user.click(screen.getByRole("button", { name: defaultLanguage === "en" ? "Edit Settings" : "编辑设置" }));
  };

  const expectErrorLanguageSwitch = async (
    user: ReturnType<typeof userEvent.setup>,
    english: string,
    chinese: string,
    count = 1,
  ) => {
    expect(await screen.findAllByText(english)).toHaveLength(count);
    const infoCalls = vi.mocked(networking.teamInfoCall).mock.calls.length;
    await user.selectOptions(screen.getByLabelText("Interface language"), "zh-CN");
    expect(await screen.findAllByText(chinese)).toHaveLength(count);
    expect(screen.queryByText(english)).not.toBeInTheDocument();
    await user.selectOptions(screen.getByLabelText("界面语言"), "en");
    expect(await screen.findAllByText(english)).toHaveLength(count);
    expect(screen.queryByText(chinese)).not.toBeInTheDocument();
    expect(networking.teamInfoCall).toHaveBeenCalledTimes(infoCalls);
    expect(networking.teamUpdateCall).not.toHaveBeenCalled();
  };

  it("已有名称错误随语言双向切换，草稿保留且只在保存时提交原值", async () => {
    const user = userEvent.setup({ delay: null });
    await openLanguageEditor(user);
    fireEvent.change(screen.getByLabelText("Team Name"), { target: { value: "" } });
    fireEvent.change(screen.getByLabelText("Max Budget (USD)"), { target: { value: "23" } });
    await user.click(screen.getByRole("button", { name: "Save Changes" }));
    await expectErrorLanguageSwitch(user, "Please input a team name", "请输入团队名称");
    expect(screen.getByLabelText("Team Name")).toHaveValue("");
    expect(screen.getByLabelText("Max Budget (USD)")).toHaveValue(23);
    fireEvent.change(screen.getByLabelText("Team Name"), { target: { value: "Home" } });
    fireEvent.change(screen.getByLabelText("Estimated Output Tokens"), { target: { value: "1024" } });
    fireEvent.change(screen.getByLabelText("Estimated Output Tokens Per Model"), {
      target: { value: '{"Home":2048}' },
    });
    await user.selectOptions(screen.getByLabelText("Interface language"), "zh-CN");
    await user.click(screen.getByRole("button", { name: "保存更改" }));
    await waitFor(() => expect(networking.teamUpdateCall).toHaveBeenCalledTimes(1));
    const expectedUpdate = {
      team_id: "123",
      team_alias: "Home",
      max_budget: 23,
      metadata: {
        default_estimated_output_tokens: 1024,
        default_estimated_output_tokens_per_model: { Home: 2048 },
      },
    };
    expect(vi.mocked(networking.teamUpdateCall).mock.calls[0][1]).toMatchObject(expectedUpdate);
  });

  it("中文校验后切回英文保留名称草稿，不重新查询或提交", async () => {
    const user = userEvent.setup({ delay: null });
    await openLanguageEditor(user, "zh-CN");
    fireEvent.change(screen.getByLabelText("团队名称"), { target: { value: "" } });
    await user.click(screen.getByRole("button", { name: "保存更改" }));
    expect(await screen.findByText("请输入团队名称")).toBeVisible();
    const infoCalls = vi.mocked(networking.teamInfoCall).mock.calls.length;
    await user.selectOptions(screen.getByLabelText("界面语言"), "en");
    expect(await screen.findByText("Please input a team name")).toBeVisible();
    expect(screen.queryByText("请输入团队名称")).not.toBeInTheDocument();
    expect(screen.getByLabelText("Team Name")).toHaveValue("");
    expect(networking.teamInfoCall).toHaveBeenCalledTimes(infoCalls);
    expect(networking.teamUpdateCall).not.toHaveBeenCalled();
  });

  it.each([
    [false, "Enter a whole number from 1 to 100", "请输入 1 到 100 之间的整数", 1],
    [true, "Duplicate threshold", "阈值重复", 2],
  ] as const)("预算告警错误随语言切换，重复=%s", async (duplicate, english, chinese, count) => {
    const user = userEvent.setup({ delay: null });
    await openLanguageEditor(user);
    await user.click(screen.getByText("Team Member Settings"));
    await user.click(screen.getByRole("button", { name: "Add Budget Alert Threshold" }));
    if (duplicate) {
      await user.click(screen.getByRole("button", { name: "Add Budget Alert Threshold" }));
      screen
        .getAllByPlaceholderText("% of budget")
        .forEach((input) => fireEvent.change(input, { target: { value: "50" } }));
    }
    await user.click(screen.getByRole("button", { name: "Save Changes" }));
    await expectErrorLanguageSwitch(user, english, chinese, count);
    expect(screen.getAllByPlaceholderText("% of budget")).toHaveLength(count);
    screen.getAllByPlaceholderText("% of budget").forEach((input) => expect(input).toHaveValue(duplicate ? 50 : null));
  });

  it.each([
    ["missing", "Missing model", "请选择模型", 1],
    ["limits", "Set at least one of TPM or RPM", "至少设置 TPM 或 RPM 之一", 1],
    ["duplicate", "Duplicate model", "模型重复", 2],
  ] as const)("模型限额错误随语言切换，类型=%s", async (kind, english, chinese, count) => {
    const user = userEvent.setup({ delay: null });
    await openLanguageEditor(user);
    for (let index = 0; index < count; index++) {
      await user.click(screen.getByRole("button", { name: "Add Model Limit" }));
      if (kind !== "missing") {
        await user.click(screen.getAllByPlaceholderText("Select model")[index]);
        await user.click(screen.getByRole("option", { name: "gpt-4" }));
      }
      if (kind === "duplicate") {
        fireEvent.change(screen.getAllByPlaceholderText("TPM Limit")[index], { target: { value: "25" } });
      }
    }
    await user.click(screen.getByRole("button", { name: "Save Changes" }));
    await expectErrorLanguageSwitch(user, english, chinese, count);
    if (kind !== "missing") {
      screen.getAllByPlaceholderText("Select model").forEach((input) => expect(input).toHaveValue("gpt-4"));
    }
    if (kind === "duplicate") {
      screen.getAllByPlaceholderText("TPM Limit").forEach((input) => expect(input).toHaveValue(25));
    }
  });

  it.each([
    ["Estimated Output Tokens", "-1", "Enter a positive integer", "请输入正整数"],
    [
      "Estimated Output Tokens Per Model",
      '{"Home":0}',
      'Enter a JSON object of positive integers, e.g. {"gpt-4": 4096}',
      '请输入由正整数组成的 JSON 对象，例如 {"gpt-4": 4096}',
    ],
  ])("预计输出 Token 错误随语言切换，字段=%s", async (label, value, english, chinese) => {
    const user = userEvent.setup({ delay: null });
    await openLanguageEditor(user);
    fireEvent.change(screen.getByLabelText(label), { target: { value } });
    // 直接提交覆盖 Zod 层错误，避免数值输入的原生 min 约束先拦截事件。
    fireEvent.submit((screen.getByLabelText(label) as HTMLInputElement | HTMLTextAreaElement).form!);
    await expectErrorLanguageSwitch(user, english, chinese);
    expect(screen.getByLabelText(label)).toHaveValue(label === "Estimated Output Tokens" ? Number(value) : value);
  });

  it.each([
    [
      "Estimated Output Tokens",
      "请求未指定 max_tokens 时，为 TPM 限速预留的预计输出 Token 数。此设置覆盖该团队的内置估算。",
      "Admin",
    ],
    [
      "Estimated Output Tokens Per Model",
      "请求未指定 max_tokens 时，按模型为 TPM 限速预留的预计输出 Token 数。此设置优先于团队级估算。",
      "Admin",
    ],
    [
      "Estimated Output Tokens",
      "仅代理管理员可修改。此设置指定请求未提供 max_tokens 时限速器预留的输出 Token 数，这些 Token 会计入团队和组织的 TPM 窗口。",
      "Internal User",
    ],
  ])("团队预计输出 Token 提示使用中文，字段=%s，提示=%s，角色=%s", async (label, chinese, role) => {
    const user = userEvent.setup({ delay: null });
    authState.userRole = role;
    await openLanguageEditor(user);
    // 上游提示触发器为无角色 SVG，使用共享组件公开的 data-slot 定位。
    const trigger = within(screen.getByText(label, { selector: "label" })).getByText("", {
      selector: '[data-slot="tooltip-trigger"]',
    });
    await user.selectOptions(screen.getByLabelText("Interface language"), "zh-CN");
    await user.hover(trigger);
    expect(await screen.findByText(chinese)).toBeVisible();
    expect(networking.teamUpdateCall).not.toHaveBeenCalled();
    if (role === "Internal User") {
      expect(screen.getByLabelText("预计输出 Token 数")).toBeDisabled();
      expect(screen.getByLabelText("各模型预计输出 Token 数")).toBeDisabled();
    }
  });
});

describe("TeamInfoView - 异步通知语言", () => {
  let realToast: typeof toast;
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

  beforeEach(async () => {
    localStorage.clear();
    authState.userRole = "Admin";
    seedDefaultMocks();
    vi.mocked(networking.teamInfoCall).mockResolvedValue(createMockTeamData());
    vi.mocked(networking.teamUpdateCall).mockResolvedValue({ team_id: "123" });
    realToast = (await vi.importActual<typeof import("@/lib/toast")>("@/lib/toast")).toast;
    // 保留原套件的调用替身，本组固定通知交给真实 Sonner 渲染并验证语言。
    vi.mocked(toast.success).mockImplementation(realToast.success);
    vi.mocked(toast.error).mockImplementation(realToast.error);
    vi.mocked(toast.fromError).mockImplementation(realToast.fromError);
    vi.mocked(toast.dismiss).mockImplementation(realToast.dismiss);
  });

  afterEach(() => {
    realToast.dismiss();
    vi.mocked(toast.success).mockReset();
    vi.mocked(toast.error).mockReset();
    vi.mocked(toast.fromError).mockReset();
    vi.mocked(toast.dismiss).mockReset();
    authState.userRole = "Admin";
    localStorage.clear();
    vi.clearAllMocks();
  });

  const renderNotifications = (extraProps: Partial<ComponentProps<typeof TeamInfoView>> = {}) =>
    renderWithProviders(
      <LanguageProvider defaultLanguage="en" userId="team-notification" storageScope="team-notification">
        <LanguageSelector />
        <TeamInfoView {...props} {...extraProps} />
        <Toaster expand duration={Infinity} />
      </LanguageProvider>,
    );

  const switchNotificationLanguage = (language: "en" | "zh-CN") =>
    fireEvent.change(screen.getByRole("combobox", { name: /Interface language|界面语言/ }), {
      target: { value: language },
    });

  const openNotificationSettings = async (user: ReturnType<typeof userEvent.setup>) => {
    await user.click(await screen.findByRole("tab", { name: "Settings" }));
    await user.click(screen.getByRole("button", { name: "Edit Settings" }));
  };

  it.each([false, true])("加载失败的通知随语言切换，不重复请求，等待中切换=%s", async (pending) => {
    const loading = Promise.withResolvers<TeamData>();
    vi.mocked(networking.teamInfoCall).mockReturnValueOnce(loading.promise);
    renderNotifications();
    await waitFor(() => expect(networking.teamInfoCall).toHaveBeenCalledTimes(1));
    if (pending) switchNotificationLanguage("zh-CN");
    await act(async () => loading.reject(new Error("private transport detail")));
    if (!pending) {
      expect(await screen.findByText("Failed to load team information")).toBeVisible();
      switchNotificationLanguage("zh-CN");
    }
    expect(await screen.findByText("加载团队信息失败，请重试")).toBeVisible();
    expect(screen.queryByText("private transport detail")).not.toBeInTheDocument();
    switchNotificationLanguage("en");
    expect(await screen.findByText("Failed to load team information")).toBeVisible();
    expect(networking.teamInfoCall).toHaveBeenCalledTimes(1);
  });

  it.each(["unrestricted", "team_admin"] as const)("保存成功通知在请求前后均能切换语言，权限=%s", async (kind) => {
    const user = userEvent.setup({ delay: null });
    const saving = Promise.withResolvers<Awaited<ReturnType<typeof networking.teamUpdateCall>>>();
    if (kind === "team_admin") {
      authState.userRole = "Internal User";
      vi.mocked(networking.teamInfoCall).mockResolvedValue(
        createMockTeamData({
          tpm_limit: 100,
          caller_edit_access: { kind: "team_admin", editable_fields: ["tpm_limit"] },
        }),
      );
    }
    vi.mocked(networking.teamUpdateCall).mockReturnValueOnce(saving.promise);
    renderNotifications(kind === "team_admin" ? { is_proxy_admin: false } : {});
    await openNotificationSettings(user);
    const label = kind === "team_admin" ? "Tokens per minute Limit (TPM)" : "Team Name";
    fireEvent.change(screen.getByLabelText(label), { target: { value: kind === "team_admin" ? "250" : "Home" } });
    await user.click(screen.getByRole("button", { name: "Save Changes" }));
    await waitFor(() => expect(networking.teamUpdateCall).toHaveBeenCalledTimes(1));
    switchNotificationLanguage("zh-CN");
    await act(async () => saving.resolve({ team_id: "123" }));
    expect(await screen.findByText("团队设置已更新")).toBeVisible();
    switchNotificationLanguage("en");
    expect(await screen.findByText("Team settings updated successfully")).toBeVisible();
    expect(networking.teamUpdateCall).toHaveBeenCalledTimes(1);
    const payload = vi.mocked(networking.teamUpdateCall).mock.calls[0][1];
    if (kind === "team_admin") expect(payload).toStrictEqual({ team_id: "123", tpm_limit: 250 });
    else expect(payload).toMatchObject({ team_alias: "Home" });
  });

  it("禁止团队管理员编辑的标题和说明均可切换，不放开权限", async () => {
    const user = userEvent.setup({ delay: null });
    vi.mocked(networking.teamInfoCall).mockResolvedValue(
      createMockTeamData({ caller_edit_access: { kind: "team_admin_disabled" } }),
    );
    renderNotifications({ is_proxy_admin: false });
    await openNotificationSettings(user);
    expect(await screen.findByText("Team admins cannot edit team settings on this proxy")).toBeVisible();
    switchNotificationLanguage("zh-CN");
    expect(await screen.findByText("此代理不允许团队管理员编辑团队设置")).toBeVisible();
    expect(screen.getByText("请联系代理管理员，在“设置 > UI > 团队管理员可编辑字段”中启用可编辑项。")).toBeVisible();
    expect(screen.queryByLabelText("团队名称")).not.toBeInTheDocument();
    expect(networking.teamUpdateCall).not.toHaveBeenCalled();
  });

  const memberCalls = {
    add: networking.teamMemberAddCall,
    update: networking.teamMemberUpdateCall,
    delete: networking.teamMemberDeleteCall,
  };
  const memberButtons = { add: "Add Member", update: "Edit Member", delete: "Delete Member" };

  const beginMemberOperation = async (user: ReturnType<typeof userEvent.setup>, kind: "add" | "update" | "delete") => {
    await user.click(await screen.findByRole("tab", { name: "Members" }));
    await user.click(
      screen.getByRole("button", {
        name: memberButtons[kind],
      }),
    );
    await user.click(screen.getByRole("button", { name: kind === "delete" ? "Confirm Delete" : "Submit" }));
  };

  it.each([
    ["add", "Team member added successfully", "团队成员已添加"],
    ["update", "Team member updated successfully", "团队成员已更新"],
    ["delete", "Team member removed successfully", "团队成员已移除"],
  ] as const)("成员操作成功通知在请求前后均能切换语言，操作=%s", async (kind, english, chinese) => {
    const user = userEvent.setup({ delay: null });
    const operation = Promise.withResolvers<void>();
    const call = memberCalls[kind];
    vi.mocked(call).mockReturnValueOnce(operation.promise);
    renderNotifications();
    await beginMemberOperation(user, kind);
    await waitFor(() => expect(call).toHaveBeenCalledTimes(1));
    switchNotificationLanguage("zh-CN");
    await act(async () => operation.resolve());
    expect(await screen.findByText(chinese)).toBeVisible();
    switchNotificationLanguage("en");
    expect(await screen.findByText(english)).toBeVisible();
    expect(call).toHaveBeenCalledTimes(1);
  });

  it.each([
    ["add", "Failed to add team member", "添加团队成员失败，请重试"],
    ["update", "Failed to update team member", "更新团队成员失败，请重试"],
    ["delete", "Failed to remove team member", "移除团队成员失败，请重试"],
  ] as const)("成员操作的本地失败说明随语言切换，操作=%s", async (kind, english, chinese) => {
    const user = userEvent.setup({ delay: null });
    const call = memberCalls[kind];
    vi.mocked(call).mockRejectedValueOnce({ status_code: 403 });
    renderNotifications();
    await beginMemberOperation(user, kind);
    expect(await screen.findByText(english)).toBeVisible();
    switchNotificationLanguage("zh-CN");
    expect(await screen.findByText(chinese)).toBeVisible();
    expect(call).toHaveBeenCalledTimes(1);
  });

  it.each(["add", "update"] as const)("服务端业务文案保持原文并保留 403 分类，操作=%s", async (kind) => {
    const user = userEvent.setup({ delay: null });
    const call = kind === "add" ? networking.teamMemberAddCall : networking.teamMemberUpdateCall;
    vi.mocked(call).mockRejectedValueOnce(new ApiError("Home", 403, { error: { message: "Home" } }));
    renderNotifications();
    await beginMemberOperation(user, kind);
    expect(await screen.findByText("Home")).toBeVisible();
    switchNotificationLanguage("zh-CN");
    expect(await screen.findByText("无访问权限")).toBeVisible();
    expect(screen.getByText("Home")).toBeVisible();
    expect(call).toHaveBeenCalledTimes(1);
  });

  it.each(["add", "update"] as const)("不误译与本地提示同名的服务端消息，操作=%s", async (kind) => {
    const user = userEvent.setup({ delay: null });
    const call = kind === "add" ? networking.teamMemberAddCall : networking.teamMemberUpdateCall;
    const message = kind === "add" ? "Failed to add team member" : "Failed to update team member";
    vi.mocked(call).mockRejectedValueOnce(new ApiError(message, 403, { error: { message } }));
    renderNotifications();
    await beginMemberOperation(user, kind);
    expect(await screen.findByText(message)).toBeVisible();
    switchNotificationLanguage("zh-CN");
    expect(await screen.findByText("无访问权限")).toBeVisible();
    expect(screen.getByText(message)).toBeVisible();
    expect(call).toHaveBeenCalledTimes(1);
  });

  it.each([
    ["servers", "the MCP server list could not be loaded", "MCP 服务列表加载失败"],
    ["toolsets", "the MCP toolset list could not be loaded", "MCP 工具集列表加载失败"],
    ["groups", "the access group list could not be loaded", "访问组列表加载失败"],
    ["loading", "the MCP server inventory is still loading", "MCP 服务清单仍在加载"],
  ] as const)("MCP 权限保存失败提示随语言切换，保持草稿和保存门禁，原因=%s", async (kind, reason, chinese) => {
    const user = userEvent.setup({ delay: null });
    const inventory = { data: [], isLoading: kind === "loading", isError: kind === "servers" };
    mockUseMCPServers.mockReturnValue(inventory as unknown as ReturnType<typeof useMCPServers>);
    mockUseMCPToolsets.mockReturnValue({
      data: [],
      isLoading: false,
      isError: kind === "toolsets",
    } as unknown as ReturnType<typeof useMCPToolsets>);
    mockUseAccessGroups.mockReturnValue({
      data: [],
      isLoading: false,
      isError: kind === "groups",
    } as unknown as ReturnType<typeof useAccessGroups>);
    vi.mocked(networking.teamInfoCall).mockResolvedValue(
      createMockTeamData({
        object_permission: {
          mcp_servers: ["server-1"],
          mcp_tool_permissions: { "server-1": ["query"] },
        },
      }),
    );
    renderNotifications();
    await openNotificationSettings(user);
    fireEvent.change(screen.getByLabelText("Team Name"), { target: { value: "Home" } });
    await user.click(screen.getByRole("button", { name: "Save Changes" }));
    const english = `Cannot save MCP tool permissions because ${reason}. Retry once the page has finished loading`;
    expect(await screen.findByText(english)).toBeVisible();
    switchNotificationLanguage("zh-CN");
    expect(await screen.findByText(`无法保存 MCP 工具权限：${chinese}。请等待页面加载完成后重试`)).toBeVisible();
    expect(screen.getByLabelText("团队名称")).toHaveValue("Home");
    switchNotificationLanguage("en");
    expect(await screen.findByText(english)).toBeVisible();
    expect(networking.teamUpdateCall).not.toHaveBeenCalled();
    expect(networking.teamInfoCall).toHaveBeenCalledTimes(1);
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
