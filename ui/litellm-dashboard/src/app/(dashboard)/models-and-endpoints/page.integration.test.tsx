/* @vitest-environment jsdom */
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import * as networking from "@/components/networking";
import { renderWithProviders } from "../../../../tests/test-utils";
import ModelsAndEndpointsPage from "./page";

vi.mock("./panels/AllModelsPanel", () => ({ default: () => <div data-testid="panel-all-models" /> }));
vi.mock("./panels/AddModelPanel", () => ({ default: () => <div data-testid="panel-add" /> }));
vi.mock("./panels/AutoRoutersTabPanel", () => ({ default: () => <div data-testid="panel-auto-routers" /> }));
vi.mock("./panels/LlmCredentialsPanel", () => ({ default: () => <div data-testid="panel-credentials" /> }));
vi.mock("./panels/PassThroughPanel", () => ({ default: () => <div data-testid="panel-pass-through" /> }));
vi.mock("./panels/HealthStatusPanel", () => ({ default: () => <div data-testid="panel-health" /> }));
vi.mock("./panels/ModelRetrySettingsPanel", () => ({ default: () => <div data-testid="panel-retry" /> }));
vi.mock("./panels/ModelGroupAliasPanel", () => ({ default: () => <div data-testid="panel-alias" /> }));
vi.mock("./panels/PriceDataPanel", () => ({ default: () => <div data-testid="panel-price" /> }));
vi.mock("./panels/AccessGroupBudgetsPanel", () => ({ default: () => <div data-testid="panel-budgets" /> }));
vi.mock("@/components/molecules/cost_optimization_feedback_banner", () => ({ default: () => null }));
vi.mock("@/components/model_info_view", () => ({
  default: ({ modelId }: { modelId: string }) => <div data-testid="model-info">model:{modelId}</div>,
}));
vi.mock("./useModelDashboardData", () => ({
  useModelDashboardData: () => ({ availableModelAccessGroups: [], allModelsOnProxy: [], availableModelGroups: [] }),
}));

const authState = vi.hoisted(() => ({ userRole: "Admin", isViewOnly: false }));
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => ({
    token: "123",
    accessToken: "123",
    userId: "user-1",
    userEmail: "admin@example.com",
    userRole: authState.userRole,
    premiumUser: true,
    isViewOnly: authState.isViewOnly,
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
  useUISettings: vi.fn(() => ({ data: { values: {} }, isLoading: false })),
}));

vi.mock("@/app/(dashboard)/hooks/models/useModels", () => ({
  useAllProxyModels: vi.fn(() => ({ data: { data: [] }, isLoading: false })),
}));

vi.mock("@/app/(dashboard)/hooks/teams/useTeams", () => ({
  useTeams: vi.fn(() => ({ data: [], isLoading: false })),
  useTeam: vi.fn(() => ({ data: undefined, isLoading: false })),
}));

vi.mock("@/app/(dashboard)/hooks/organizations/useOrganizations", () => ({
  organizationKeys: { all: ["organizations"] },
  useOrganization: vi.fn(() => ({ data: undefined, isLoading: false })),
  useOrganizations: vi.fn().mockReturnValue({ data: [], isLoading: false }),
}));

vi.mock("@/app/(dashboard)/hooks/users/useCurrentUser", () => ({
  useCurrentUser: vi.fn(() => ({ data: { models: [] }, isLoading: false })),
}));

vi.mock("@/app/(dashboard)/hooks/mcpServers/useMCPServers", () => ({
  useMCPServers: vi.fn(() => ({ data: [], isLoading: false, isError: false })),
}));

vi.mock("@/app/(dashboard)/hooks/mcpServers/useMCPToolsets", () => ({
  useMCPToolsets: vi.fn(() => ({ data: [], isLoading: false, isError: false })),
}));

vi.mock("@/components/mcp_server_management/MCPServerSelector", () => ({
  default: () => <div>mcp server selector</div>,
}));

vi.mock("@/components/team/TeamMemberTab", () => ({
  default: vi.fn(() => <div>member tab</div>),
}));

vi.mock("@/components/common_components/user_search_modal", () => ({
  default: vi.fn(() => null),
}));

vi.mock("@/components/team/EditMembership", () => ({
  default: vi.fn(() => null),
}));

vi.mock("@/components/common_components/DeleteResourceModal", () => ({
  default: vi.fn(() => null),
}));

vi.mock("@/components/team/member_permissions", () => ({
  default: vi.fn(() => <div>Member Permissions</div>),
}));

vi.mock("@/components/common_components/ModelAliasManager", () => ({
  default: vi.fn(() => <div>alias manager</div>),
}));

vi.mock("@/app/(dashboard)/hooks/accessGroups/useAccessGroups", () => ({
  useAccessGroups: vi.fn().mockReturnValue({ data: [], isLoading: false, isError: false }),
}));

vi.mock("@/components/common_components/AccessGroupSelector", () => ({
  default: () => <div>access group selector</div>,
}));

vi.mock("@/app/(dashboard)/hooks/keys/useKeys", () => {
  const keysResult = {
    data: { keys: [], total_count: 0, current_page: 1, total_pages: 1 },
    isPending: false,
    isFetching: false,
    refetch: vi.fn(),
  };
  return { useKeys: vi.fn(() => keysResult) };
});

vi.mock("@/components/key_team_helpers/filter_helpers", () => ({
  fetchTeamFilterOptions: vi.fn().mockResolvedValue({ keyAliases: [], organizationIds: [], userIds: [] }),
  fetchAllKeyAliases: vi.fn().mockResolvedValue([]),
  fetchAllOrganizations: vi.fn().mockResolvedValue([]),
}));

const createMockTeamData = (overrides = {}) => ({
  team_id: "team-a1b2",
  team_info: {
    team_alias: "Test Team",
    team_id: "team-a1b2",
    organization_id: null,
    admins: ["admin@test.com"],
    members: ["user1@test.com"],
    members_with_roles: [
      { user_id: "user1@test.com", user_email: "user1@test.com", role: "member", spend: 0, budget_id: "budget1" },
    ],
    metadata: { disable_global_guardrails: true },
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

describe("ModelsAndEndpointsPage ?team drill-in", () => {
  beforeEach(() => {
    authState.userRole = "Admin";
    authState.isViewOnly = false;
    can.mockReturnValue(true);
    vi.mocked(networking.getGuardrailsList).mockResolvedValue({ guardrails: [] });
    vi.mocked(networking.getPoliciesList).mockResolvedValue({ policies: [] });
    vi.mocked(networking.fetchMCPAccessGroups).mockResolvedValue([]);
    vi.mocked(networking.getTeamPermissionsCall).mockResolvedValue({
      all_available_permissions: [],
      team_member_permissions: [],
    } as never);
    vi.mocked(networking.teamInfoCall).mockResolvedValue(createMockTeamData() as never);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (global as any).ResizeObserver = class {
      observe() {}
      unobserve() {}
      disconnect() {}
    };
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("shows the Disable all global guardrails switch to a proxy admin session", async () => {
    const user = userEvent.setup({ delay: null });
    renderWithProviders(<ModelsAndEndpointsPage />, { searchParams: { team: "team-a1b2" } });

    await user.click(await screen.findByRole("tab", { name: "Settings" }));
    await user.click(await screen.findByRole("button", { name: /edit settings/i }));
    await screen.findByLabelText(/Team Name/);

    expect(screen.getByText("Disable all global guardrails")).toBeInTheDocument();
  });

  it("keeps the switch hidden from an internal user session on the same team", async () => {
    authState.userRole = "Internal User";
    renderWithProviders(<ModelsAndEndpointsPage />, { searchParams: { team: "team-a1b2" } });

    await screen.findByText("Test Team");

    expect(screen.queryByRole("button", { name: /edit settings/i })).not.toBeInTheDocument();
    expect(screen.queryByText("Disable all global guardrails")).not.toBeInTheDocument();
  });
});
