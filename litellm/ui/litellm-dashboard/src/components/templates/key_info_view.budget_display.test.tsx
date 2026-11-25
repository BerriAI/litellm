import { renderWithProviders } from "../../../tests/test-utils";
import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { KeyResponse, Team } from "../key_team_helpers/key_list";
import KeyInfoView from "./key_info_view";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import useTeams from "@/app/(dashboard)/hooks/useTeams";
import { useOrganizations } from "@/app/(dashboard)/hooks/organizations/useOrganizations";
import { useApplyUserBudgetToTeamKeys } from "@/app/(dashboard)/hooks/uiSettings/useApplyUserBudgetToTeamKeys";
import type { Organization } from "../networking";

vi.mock("next/navigation", () => ({
  usePathname: () => "/ui/",
  useRouter: () => ({ push: vi.fn() }),
}));

vi.mock("./key_edit_view", () => ({
  KeyEditView: () => <div data-testid="key-edit-view-stub" />,
}));

vi.mock("@/app/(dashboard)/hooks/useTeams", () => ({ default: vi.fn() }));
vi.mock("@/app/(dashboard)/hooks/organizations/useOrganizations", () => ({ useOrganizations: vi.fn() }));
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({ default: vi.fn() }));
vi.mock("@/app/(dashboard)/hooks/projects/useProjects", () => ({
  useProjects: vi.fn().mockReturnValue({ data: [], isLoading: false }),
}));
vi.mock("@/app/(dashboard)/hooks/uiSettings/useApplyUserBudgetToTeamKeys", () => ({
  useApplyUserBudgetToTeamKeys: vi.fn(() => false),
}));
vi.mock("@/app/(dashboard)/hooks/keys/useResetKeySpend", () => ({
  useResetKeySpend: vi.fn(() => ({ mutate: vi.fn(), isPending: false })),
}));
vi.mock("../networking", () => ({
  serverRootPath: "",
  getProxyBaseUrl: vi.fn(() => ""),
  getUiSettings: vi.fn().mockResolvedValue({ values: {} }),
  keyDeleteCall: vi.fn().mockResolvedValue({}),
  keyUpdateCall: vi.fn().mockResolvedValue({}),
  getPolicyInfoWithGuardrails: vi.fn().mockResolvedValue({ resolved_guardrails: [] }),
}));

const MOCK_KEY_DATA = {
  token: "test-token-123",
  token_id: "test-token-123",
  key_name: "sk-...abcd",
  key_alias: "lit-2845-budget-display",
  spend: 0.0001,
  max_budget: null as number | null,
  expires: "null",
  models: [],
  aliases: {},
  config: {},
  user_id: "default_user_id",
  team_id: null,
  max_parallel_requests: null,
  metadata: {},
  tpm_limit: null,
  rpm_limit: null,
  budget_duration: null,
  budget_reset_at: null,
  allowed_cache_controls: [],
  permissions: {},
  model_spend: {},
  model_max_budget: {},
  soft_budget_cooldown: false,
  blocked: false,
  litellm_budget_table: {},
  organization_id: null,
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-01T00:00:00Z",
  team_spend: 0,
  team_alias: "",
  team_tpm_limit: null,
  team_rpm_limit: null,
  team_max_budget: null,
  team_models: [],
  team_blocked: false,
  soft_budget: null,
  team_model_aliases: {},
  team_member_spend: 0,
  team_metadata: {},
  end_user_id: null,
  end_user_tpm_limit: null,
  end_user_rpm_limit: null,
  end_user_max_budget: null,
  last_refreshed_at: 0,
  api_key: "sk-...abcd",
  user_role: "user",
  rpm_limit_per_model: {},
  tpm_limit_per_model: {},
  user_tpm_limit: null,
  user_rpm_limit: null,
  user_email: "test@example.com",
  object_permission: {
    object_permission_id: "perm-1",
    mcp_servers: [],
    mcp_access_groups: [],
    mcp_tool_permissions: {},
    vector_stores: [],
  },
  auto_rotate: false,
  rotation_interval: undefined,
  last_rotation_at: undefined,
  key_rotation_at: undefined,
} as unknown as KeyResponse;

const baseAuthorized = {
  accessToken: "test-token",
  userId: "test-user",
  userRole: "admin",
  premiumUser: true,
  token: "test-token",
  userEmail: null,
  disabledPersonalKeyCreation: null,
  showSSOBanner: false,
  isLoading: false,
  isAuthorized: true,
};

const TEST_BUDGET_TEAM_FIELDS = {
  team_id: "team-123",
  team_alias: "Test Budget",
  max_budget: 1200,
  budget_duration: "30d",
};

const TEAM_KEY_WITH_BUDGETED_OWNER = {
  ...MOCK_KEY_DATA,
  max_budget: null,
  team_id: "team-123",
  user: {
    user_id: "user-1",
    user_email: "owner@example.com",
    user_alias: "Budget Owner",
    max_budget: 1500,
    budget_duration: "1mo",
  },
} as unknown as KeyResponse;

const makeTeam = (overrides: Partial<Team>): Team => ({
  team_id: "team-default",
  team_alias: "Default Team",
  models: [],
  max_budget: null,
  budget_duration: null,
  tpm_limit: null,
  rpm_limit: null,
  organization_id: "",
  created_at: "2026-01-01T00:00:00Z",
  keys: [],
  members_with_roles: [],
  spend: 0,
  ...overrides,
});

const makeOrganization = (overrides: Partial<Organization>): Organization =>
  ({
    organization_id: "org-1",
    organization_alias: "Acme Org",
    budget_id: "budget-1",
    metadata: {},
    models: [],
    spend: 0,
    model_spend: {},
    created_at: "2026-01-01T00:00:00Z",
    created_by: "admin",
    updated_at: "2026-01-01T00:00:00Z",
    updated_by: "admin",
    litellm_budget_table: { max_budget: null, budget_duration: null },
    teams: null,
    users: null,
    members: null,
    ...overrides,
  }) as Organization;

const mockOrganizations = (organizations: Organization[]) =>
  vi.mocked(useOrganizations).mockReturnValue({ data: organizations } as ReturnType<typeof useOrganizations>);

const appliedBudgetRows = () => within(screen.getByTestId("applied-budgets")).queryAllByRole("listitem");

const canStillSpend = () => screen.getByTestId("key-can-still-spend");

describe("KeyInfoView overview budget display (LIT-2845)", () => {
  beforeEach(() => {
    vi.mocked(useTeams).mockReturnValue({ teams: [], setTeams: vi.fn() });
    vi.mocked(useAuthorized).mockReturnValue(baseAuthorized);
    mockOrganizations([]);
    vi.mocked(useApplyUserBudgetToTeamKeys).mockReturnValue(false);
  });

  it("renders a sub-dollar max_budget ($0.10) with 2-decimal precision in the overview Spend card", async () => {
    renderWithProviders(
      <KeyInfoView
        keyData={{ ...MOCK_KEY_DATA, max_budget: 0.1 }}
        onClose={() => {}}
        keyId={"test-key-id"}
        onKeyDataUpdate={() => {}}
        teams={[]}
      />,
    );

    await screen.findByTestId("applied-budgets");
    expect(appliedBudgetRows()[0]).toHaveTextContent("$0.0001 of $0.10");
    expect(canStillSpend()).toHaveTextContent("$0.0999");
    expect(canStillSpend()).toHaveTextContent("$0.0001 of $0.10 used");
  });

  it("still renders whole-dollar max_budget with 2-decimal precision", async () => {
    renderWithProviders(
      <KeyInfoView
        keyData={{ ...MOCK_KEY_DATA, max_budget: 100 }}
        onClose={() => {}}
        keyId={"test-key-id"}
        onKeyDataUpdate={() => {}}
        teams={[]}
      />,
    );
    expect(await screen.findByText("of $100.00")).toBeInTheDocument();
  });

  it("says nothing stops spend when max_budget is null and nothing else applies", async () => {
    renderWithProviders(
      <KeyInfoView
        keyData={{ ...MOCK_KEY_DATA, max_budget: null } as unknown as KeyResponse}
        onClose={() => {}}
        keyId={"test-key-id"}
        onKeyDataUpdate={() => {}}
        teams={[]}
      />,
    );
    expect(await screen.findByTestId("key-can-still-spend")).toHaveTextContent(
      "No budget limits this key, so nothing stops its spend.",
    );
    expect(screen.getByTestId("key-period-spend")).toHaveTextContent("$0.0001");
  });

  it("never pairs key spend with the team budget: lists the team budget under Budgets that apply", async () => {
    vi.mocked(useTeams).mockReturnValue({
      teams: [makeTeam({ ...TEST_BUDGET_TEAM_FIELDS, spend: 300 })],
      setTeams: vi.fn(),
    });
    renderWithProviders(
      <KeyInfoView
        keyData={{ ...MOCK_KEY_DATA, max_budget: null, team_id: "team-123" } as unknown as KeyResponse}
        onClose={() => {}}
        keyId={"test-key-id"}
        onKeyDataUpdate={() => {}}
        teams={[]}
      />,
    );
    await screen.findByTestId("applied-budgets");
    expect(screen.getByTestId("key-period-spend")).toHaveTextContent("$0.0001");
    expect(canStillSpend()).toHaveTextContent("$900.00");
    const rows = appliedBudgetRows();
    expect(rows).toHaveLength(1);
    expect(rows[0]).toHaveTextContent("Test Budget");
    expect(rows[0]).toHaveTextContent("$300.00 of $1,200.00");
    expect(rows[0]).toHaveTextContent("Resets every 30d");
    expect(within(rows[0]).getByRole("meter")).toHaveAttribute("aria-valuetext", "25% used");
  });

  it("lists the organization budget when the team's org has one", async () => {
    vi.mocked(useTeams).mockReturnValue({
      teams: [makeTeam({ team_id: "team-456", team_alias: "Org Team", organization_id: "org-1" })],
      setTeams: vi.fn(),
    });
    mockOrganizations([
      makeOrganization({ spend: 4900, litellm_budget_table: { max_budget: 5000, budget_duration: null } }),
    ]);
    renderWithProviders(
      <KeyInfoView
        keyData={{ ...MOCK_KEY_DATA, max_budget: null, team_id: "team-456" } as unknown as KeyResponse}
        onClose={() => {}}
        keyId={"test-key-id"}
        onKeyDataUpdate={() => {}}
        teams={[]}
      />,
    );
    await screen.findByTestId("applied-budgets");
    const rows = appliedBudgetRows();
    expect(rows).toHaveLength(1);
    expect(rows[0]).toHaveTextContent("Acme Org");
    expect(rows[0]).toHaveTextContent("$4,900.00 of $5,000.00");
    expect(rows[0]).toHaveTextContent("Does not reset");
  });

  it("lists the owner's user budget for a personal key with no budget of its own", async () => {
    renderWithProviders(
      <KeyInfoView
        keyData={
          {
            ...MOCK_KEY_DATA,
            max_budget: null,
            team_id: null,
            user: {
              user_id: "user-1",
              user_email: "owner@example.com",
              user_alias: "Budget Owner",
              max_budget: 1500,
              budget_duration: "1mo",
            },
          } as unknown as KeyResponse
        }
        onClose={() => {}}
        keyId={"test-key-id"}
        onKeyDataUpdate={() => {}}
        teams={[]}
      />,
    );
    await screen.findByTestId("applied-budgets");
    const rows = appliedBudgetRows();
    expect(rows).toHaveLength(1);
    expect(rows[0]).toHaveTextContent("Budget Owner");
    expect(rows[0]).toHaveTextContent("$1,500.00 limit");
    expect(rows[0]).toHaveTextContent("Resets every 1mo");
  });

  it("omits the owner's user budget for a team key when apply_user_budget_to_team_keys is off", async () => {
    vi.mocked(useTeams).mockReturnValue({
      teams: [makeTeam(TEST_BUDGET_TEAM_FIELDS)],
      setTeams: vi.fn(),
    });
    renderWithProviders(
      <KeyInfoView
        keyData={TEAM_KEY_WITH_BUDGETED_OWNER}
        onClose={() => {}}
        keyId={"test-key-id"}
        onKeyDataUpdate={() => {}}
        teams={[]}
      />,
    );
    await screen.findByTestId("applied-budgets");
    const card = screen.getByTestId("applied-budgets");
    expect(card).toHaveTextContent("Test Budget");
    expect(card).not.toHaveTextContent("Budget Owner");
  });

  it("lists the owner's user budget for a team key when apply_user_budget_to_team_keys is on", async () => {
    vi.mocked(useApplyUserBudgetToTeamKeys).mockReturnValue(true);
    vi.mocked(useTeams).mockReturnValue({
      teams: [makeTeam(TEST_BUDGET_TEAM_FIELDS)],
      setTeams: vi.fn(),
    });
    renderWithProviders(
      <KeyInfoView
        keyData={TEAM_KEY_WITH_BUDGETED_OWNER}
        onClose={() => {}}
        keyId={"test-key-id"}
        onKeyDataUpdate={() => {}}
        teams={[]}
      />,
    );
    await screen.findByTestId("applied-budgets");
    const rows = appliedBudgetRows();
    expect(rows).toHaveLength(2);
    expect(rows[0]).toHaveTextContent("Test Budget");
    expect(rows[1]).toHaveTextContent("Budget Owner");
  });

  it("says no budget limits the key when neither key, team, nor org has a budget", async () => {
    vi.mocked(useTeams).mockReturnValue({
      teams: [makeTeam({ team_id: "team-789", team_alias: "Free Team" })],
      setTeams: vi.fn(),
    });
    renderWithProviders(
      <KeyInfoView
        keyData={{ ...MOCK_KEY_DATA, max_budget: null, team_id: "team-789" } as unknown as KeyResponse}
        onClose={() => {}}
        keyId={"test-key-id"}
        onKeyDataUpdate={() => {}}
        teams={[]}
      />,
    );
    expect(await screen.findByTestId("key-spend-card")).toHaveTextContent(
      "No budget limits this key. Spend is tracked but never blocked.",
    );
    expect(screen.queryByTestId("applied-budgets")).not.toBeInTheDocument();
    expect(canStillSpend()).toHaveTextContent("nothing stops its spend");
  });

  it("keeps the team budget visible next to the key's own budget, since both are enforced", async () => {
    const nearlySpentTeam = { team_id: "team-123", team_alias: "Test Budget", max_budget: 1200, spend: 1100 };
    vi.mocked(useTeams).mockReturnValue({ teams: [makeTeam(nearlySpentTeam)], setTeams: vi.fn() });
    renderWithProviders(
      <KeyInfoView
        keyData={{ ...MOCK_KEY_DATA, max_budget: 25, spend: 5, team_id: "team-123" } as unknown as KeyResponse}
        onClose={() => {}}
        keyId={"test-key-id"}
        onKeyDataUpdate={() => {}}
        teams={[]}
      />,
    );
    await screen.findByTestId("applied-budgets");
    const rows = appliedBudgetRows();
    expect(rows).toHaveLength(2);
    expect(rows[0]).toHaveTextContent("lit-2845-budget-display");
    expect(rows[0]).toHaveTextContent("$5.00 of $25.00");
    expect(rows[1]).toHaveTextContent("Test Budget");
    expect(rows[1]).toHaveTextContent("$1,100.00 of $1,200.00");
    expect(canStillSpend()).toHaveTextContent("$20.00");
    expect(canStillSpend()).toHaveTextContent("before the key budget blocks this key");
  });
});

describe("KeyInfoView budget reset visibility", () => {
  beforeEach(() => {
    vi.mocked(useTeams).mockReturnValue({ teams: [], setTeams: vi.fn() });
    vi.mocked(useAuthorized).mockReturnValue(baseAuthorized);
    mockOrganizations([]);
  });

  const KEY_WITH_RESET = {
    ...MOCK_KEY_DATA,
    max_budget: 0.1,
    budget_duration: "1d",
    budget_reset_at: "2026-07-22T12:00:00+00:00",
  } as unknown as KeyResponse;

  it("shows the next budget reset in the overview Spend card when budget_reset_at is set", async () => {
    renderWithProviders(
      <KeyInfoView
        keyData={KEY_WITH_RESET}
        onClose={() => {}}
        keyId={"test-key-id"}
        onKeyDataUpdate={() => {}}
        teams={[]}
      />,
    );
    expect(await screen.findByTestId("key-spend-reset")).toHaveTextContent("Resets Jul 22, 2026 at 12:00 PM UTC");
  });

  it("omits the reset line from the overview Spend card when budget_reset_at is null", async () => {
    renderWithProviders(
      <KeyInfoView
        keyData={{ ...MOCK_KEY_DATA, max_budget: 0.1 } as unknown as KeyResponse}
        onClose={() => {}}
        keyId={"test-key-id"}
        onKeyDataUpdate={() => {}}
        teams={[]}
      />,
    );
    expect(await screen.findByTestId("key-spend-reset")).toHaveTextContent("Never resets.");
    expect(screen.queryByText(/^Resets /)).not.toBeInTheDocument();
  });

  it("shows the duration and next reset in the Settings tab", async () => {
    renderWithProviders(
      <KeyInfoView
        keyData={KEY_WITH_RESET}
        onClose={() => {}}
        keyId={"test-key-id"}
        onKeyDataUpdate={() => {}}
        teams={[]}
      />,
    );
    await waitFor(() => {
      expect(screen.getByRole("tab", { name: "Settings" })).toBeInTheDocument();
    });
    fireEvent.click(screen.getByRole("tab", { name: "Settings" }));
    await waitFor(() => {
      expect(screen.getByText("Budget Reset")).toBeInTheDocument();
    });
    expect(screen.getByText(/Every 1d, next Jul 22, 2026/)).toBeInTheDocument();
  });

  it("shows 'Never' in the Settings tab when no reset is scheduled", async () => {
    renderWithProviders(
      <KeyInfoView
        keyData={MOCK_KEY_DATA}
        onClose={() => {}}
        keyId={"test-key-id"}
        onKeyDataUpdate={() => {}}
        teams={[]}
      />,
    );
    await waitFor(() => {
      expect(screen.getByRole("tab", { name: "Settings" })).toBeInTheDocument();
    });
    fireEvent.click(screen.getByRole("tab", { name: "Settings" }));
    await waitFor(() => {
      expect(screen.getByText("Budget Reset")).toBeInTheDocument();
    });
    expect(screen.getByTestId("budget-reset-value")).toHaveTextContent("Never");
  });
});
