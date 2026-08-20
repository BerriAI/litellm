import { renderWithProviders } from "../../../tests/test-utils";
import { fireEvent, screen, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { KeyBudgetEntry } from "@/app/(dashboard)/hooks/keys/useKeyBudgets";
import type { KeyResponse } from "../key_team_helpers/key_list";
import KeyInfoView from "./key_info_view";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import useTeams from "@/app/(dashboard)/hooks/useTeams";

// The Budgets tab is the whole point of the endpoint: the key, its team and its user were all
// unlimited and the request still 429'd, because the gate was the team-member budget. These tests
// pin that a user can pick that row out of the table without reading auth source.

const apiMocks = vi.hoisted(() => ({ useQuery: vi.fn() }));

vi.mock("@/lib/http/api", () => ({
  $api: { useQuery: apiMocks.useQuery },
  fetchClient: { GET: vi.fn(), POST: vi.fn() },
}));

vi.mock("next/navigation", () => ({ useRouter: () => ({ push: vi.fn() }) }));

vi.mock("./key_edit_view", () => ({
  KeyEditView: () => <div data-testid="key-edit-view-stub" />,
}));

vi.mock("@/app/(dashboard)/hooks/useTeams", () => ({ default: vi.fn() }));
vi.mock("@/app/(dashboard)/hooks/organizations/useOrganizations", () => ({
  useOrganizations: vi.fn().mockReturnValue({ data: [] }),
}));
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({ default: vi.fn() }));
vi.mock("@/app/(dashboard)/hooks/projects/useProjects", () => ({
  useProjects: vi.fn().mockReturnValue({ data: [], isLoading: false }),
}));
vi.mock("@/app/(dashboard)/hooks/keys/useResetKeySpend", () => ({
  useResetKeySpend: vi.fn(() => ({ mutate: vi.fn(), isPending: false })),
}));
vi.mock("../networking", () => ({
  serverRootPath: "",
  keyDeleteCall: vi.fn().mockResolvedValue({}),
  keyUpdateCall: vi.fn().mockResolvedValue({}),
  getPolicyInfoWithGuardrails: vi.fn().mockResolvedValue({ resolved_guardrails: [] }),
}));

const MOCK_KEY_DATA = {
  token: "test-token-123",
  token_id: "test-token-123",
  key_name: "sk-...abcd",
  key_alias: "ci-runner",
  spend: 1000.2,
  max_budget: null,
  expires: "null",
  models: [],
  aliases: {},
  config: {},
  user_id: "default_user_id",
  team_id: "team-123",
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
  team_max_budget: null,
  team_models: [],
  team_blocked: false,
  soft_budget: null,
  team_model_aliases: {},
  team_member_spend: 0,
  team_metadata: {},
  end_user_id: null,
  last_refreshed_at: 0,
  api_key: "sk-...abcd",
  user_role: "user",
  rpm_limit_per_model: {},
  tpm_limit_per_model: {},
  user_email: "alice@example.com",
  object_permission: {
    object_permission_id: "perm-1",
    mcp_servers: [],
    mcp_access_groups: [],
    mcp_tool_permissions: {},
    vector_stores: [],
  },
  auto_rotate: false,
} as unknown as KeyResponse;

const baseAuthorized = {
  accessToken: "test-token",
  userId: "test-user",
  userRole: "admin",
  userRoleLabel: "Admin",
  isViewOnly: false,
  premiumUser: true,
  token: "test-token",
  userEmail: null,
  disabledPersonalKeyCreation: null,
  showSSOBanner: false,
  isLoading: false,
  isAuthorized: true,
};

const UNCONFIGURED_BUDGET = {
  scope: "key",
  entity_type: "key",
  entity_id: null,
  entity_label: null,
  enforcement: "hard",
  max_budget: null,
  spend: 0,
  remaining: null,
  comparison: ">=",
  budget_duration: null,
  budget_reset_at: null,
  window_start: null,
  source: "key.max_budget",
  status: "unlimited",
  note: null,
} as KeyBudgetEntry;

const KEY_UNLIMITED: KeyBudgetEntry = {
  ...UNCONFIGURED_BUDGET,
  entity_id: "test-token-123",
  entity_label: "ci-runner",
  spend: 1000.2,
};

const USER_WITHIN_BUDGET: KeyBudgetEntry = {
  ...UNCONFIGURED_BUDGET,
  scope: "user",
  entity_type: "user",
  entity_id: "default_user_id",
  entity_label: "alice@example.com",
  max_budget: 250,
  spend: 10,
  remaining: 240,
  source: "user.max_budget",
  status: "ok",
};

const TEAM_SOFT_OVER: KeyBudgetEntry = {
  ...UNCONFIGURED_BUDGET,
  scope: "team",
  entity_type: "team",
  entity_id: "team-123",
  entity_label: "Platform",
  enforcement: "soft",
  max_budget: 500,
  spend: 900,
  remaining: -400,
  source: "budget_table:b-soft",
  status: "exceeded",
  note: "alert only, never blocks; compared against recorded spend rather than the live counter",
};

const TEAM_MEMBER_BLOCKING: KeyBudgetEntry = {
  ...UNCONFIGURED_BUDGET,
  scope: "team_member",
  entity_type: "team_member",
  entity_id: "default_user_id:team-123",
  entity_label: "alice @ Platform",
  max_budget: 1000,
  spend: 1000.2,
  remaining: -0.2,
  budget_duration: "30d",
  budget_reset_at: "2026-09-01T12:00:00+00:00",
  source: "team.metadata.team_member_budget_id",
  status: "exceeded",
};

const ORG_UNCONFIGURED: KeyBudgetEntry = {
  ...UNCONFIGURED_BUDGET,
  scope: "organization",
  entity_type: "organization",
  entity_id: "org-1",
  entity_label: "Acme Org",
  spend: 12.5,
  source: "organization.budget_id",
};

// Longest note the endpoint can emit: 276 characters over three clauses, reachable only on an
// end_user row where reservation tightened the operator and a custom auth callable is configured.
// The reservation clause lands only on team, tag and end_user, so this is a real ceiling.
const WORST_CASE_NOTE =
  "the reservation layer blocks this scope as soon as spend reaches the limit, before the read-time " +
  "check would trip; only enforced on LLM routes that name this end user; a custom auth callable can " +
  "set a request-scoped end user cap that overrides this one and is not visible here";

const ALL_BUDGETS = [KEY_UNLIMITED, USER_WITHIN_BUDGET, TEAM_SOFT_OVER, TEAM_MEMBER_BLOCKING, ORG_UNCONFIGURED];

const mockBudgets = (budgets: KeyBudgetEntry[]) => {
  const loaded = { data: { key: "test-token-123", budgets }, isLoading: false, isError: false, error: null };
  apiMocks.useQuery.mockReturnValue(loaded);
};

const renderKeyInfo = () =>
  renderWithProviders(
    <KeyInfoView
      keyData={MOCK_KEY_DATA}
      onClose={() => {}}
      keyId="test-key-id"
      onKeyDataUpdate={() => {}}
      teams={[]}
    />,
  );

const openBudgetsTab = async () => {
  fireEvent.click(await screen.findByRole("tab", { name: "Budgets" }));
  return within(await screen.findByTestId("key-budgets-panel"));
};

const renderAndOpenBudgetsTab = async () => {
  renderKeyInfo();
  return openBudgetsTab();
};

const rowFor = (panel: ReturnType<typeof within>, entityLabel: string): HTMLElement => {
  const row = panel.getByText(entityLabel).closest("tr");
  if (row === null) throw new Error(`no table row rendered for ${entityLabel}`);
  return row;
};

describe("KeyInfoView Budgets tab", () => {
  beforeEach(() => {
    apiMocks.useQuery.mockReset();
    mockBudgets(ALL_BUDGETS);
    vi.mocked(useTeams).mockReturnValue({ teams: [], setTeams: vi.fn() });
    vi.mocked(useAuthorized).mockReturnValue(baseAuthorized);
  });

  it("does not fetch budgets until the tab is opened, then asks for this key's id", async () => {
    renderKeyInfo();
    expect(await screen.findByRole("tab", { name: "Budgets" })).toBeInTheDocument();
    expect(apiMocks.useQuery).not.toHaveBeenCalled();

    await openBudgetsTab();

    expect(apiMocks.useQuery).toHaveBeenCalledWith(
      "get",
      "/key/{key_id}/budgets",
      { params: { path: { key_id: "test-token-123" } } },
      { enabled: true },
    );
  });

  it("identifies the team-member budget as the single one blocking the key", async () => {
    const panel = await renderAndOpenBudgetsTab();

    const blocking = panel.getAllByTestId("key-budget-blocking");
    expect(blocking).toHaveLength(1);
    expect(blocking[0]).toHaveTextContent("Exceeded");

    const blockedRow = rowFor(panel, "alice @ Platform");
    expect(within(blockedRow).getByText("Team member")).toBeInTheDocument();
    expect(within(blockedRow).getByText("Blocks requests")).toBeInTheDocument();
    expect(blockedRow).toHaveTextContent("$1,000.2000 of $1,000.00");
  });

  it("shows an over-budget soft limit as alert-only, never as a blocker", async () => {
    const panel = await renderAndOpenBudgetsTab();

    const softRow = rowFor(panel, "Platform");
    expect(within(softRow).getByText("Alert only")).toBeInTheDocument();
    expect(within(softRow).getByText("Exceeded (alert only)")).toBeInTheDocument();
    expect(within(softRow).queryByTestId("key-budget-blocking")).not.toBeInTheDocument();
    expect(within(softRow).queryByText("Blocks requests")).not.toBeInTheDocument();
    expect(softRow).toHaveTextContent(
      "alert only, never blocks; compared against recorded spend rather than the live counter",
    );
  });

  it("renders the longest note the endpoint can emit without dropping any of it", async () => {
    const endUser: KeyBudgetEntry = {
      ...UNCONFIGURED_BUDGET,
      scope: "end_user",
      entity_type: "end_user",
      entity_id: "customer-42",
      entity_label: "customer-42",
      max_budget: 25,
      spend: 25,
      remaining: 0,
      source: "budget_table:b-end-user",
      status: "exceeded",
      note: WORST_CASE_NOTE,
    };
    mockBudgets([endUser]);
    const panel = await renderAndOpenBudgetsTab();

    const note = panel.getByText(WORST_CASE_NOTE);
    expect(note).toBeInTheDocument();
    // jsdom has no layout, so nothing here can prove the text is visually unclipped. Asserting the
    // absence of the clipping utilities is the only mechanical guard against re-truncating the note.
    expect(note).not.toHaveClass("truncate");
    expect(note.className).not.toMatch(/line-clamp|overflow-hidden|whitespace-nowrap/);
    expect(note).not.toHaveAttribute("title");
  });

  it("explains why two rows on identical numbers get opposite statuses", async () => {
    const inclusive: KeyBudgetEntry = {
      ...UNCONFIGURED_BUDGET,
      scope: "team_member",
      entity_type: "team_member",
      entity_label: "alice @ Platform",
      comparison: ">=",
      max_budget: 300,
      spend: 300,
      remaining: 0,
      status: "exceeded",
    };
    const exclusive: KeyBudgetEntry = {
      ...UNCONFIGURED_BUDGET,
      scope: "project",
      entity_type: "project",
      entity_label: "Platform",
      comparison: ">",
      max_budget: 300,
      spend: 300,
      remaining: 0,
      status: "ok",
    };
    mockBudgets([inclusive, exclusive]);
    const panel = await renderAndOpenBudgetsTab();

    const blockedRow = rowFor(panel, "alice @ Platform");
    const allowedRow = rowFor(panel, "Platform");

    expect(blockedRow).toHaveTextContent("Blocks at ≥ $300.00");
    expect(allowedRow).toHaveTextContent("Blocks at > $300.00");

    expect(within(blockedRow).getByTestId("key-budget-blocking")).toBeInTheDocument();
    expect(within(allowedRow).queryByTestId("key-budget-blocking")).not.toBeInTheDocument();
    expect(within(allowedRow).getByText("Within budget")).toBeInTheDocument();

    expect(blockedRow).toHaveTextContent("$300.0000 of $300.00");
    expect(allowedRow).toHaveTextContent("$300.0000 of $300.00");
  });

  it("states no threshold for a scope with nothing configured", async () => {
    const panel = await renderAndOpenBudgetsTab();

    expect(rowFor(panel, "Acme Org")).not.toHaveTextContent("Blocks at");
  });

  it("renders a scope with nothing configured as Unlimited rather than $0", async () => {
    const panel = await renderAndOpenBudgetsTab();

    const orgRow = rowFor(panel, "Acme Org");
    expect(orgRow).toHaveTextContent("· Unlimited");
    expect(within(orgRow).getAllByText("Unlimited")).toHaveLength(2);
    expect(orgRow).not.toHaveTextContent("$0.00");
    expect(within(orgRow).queryByRole("meter")).not.toBeInTheDocument();
  });

  it("puts the blocking budget above the alert-only, healthy and unlimited ones", async () => {
    const panel = await renderAndOpenBudgetsTab();

    const [, ...dataRows] = panel.getAllByRole("row");
    expect(dataRows[0]).toHaveTextContent("alice @ Platform");
    expect(dataRows[1]).toHaveTextContent("Exceeded (alert only)");
    expect(dataRows[2]).toHaveTextContent("Within budget");
    expect(dataRows.slice(3).every((row) => row.textContent?.includes("Unlimited"))).toBe(true);
  });

  it("shows when the blocking budget next resets", async () => {
    const panel = await renderAndOpenBudgetsTab();

    const blockedRow = rowFor(panel, "alice @ Platform");
    expect(blockedRow).toHaveTextContent("Sep 1, 2026");
    expect(blockedRow).toHaveTextContent("Every 30d");
  });

  it("says a scope never resets when no reset is scheduled", async () => {
    const panel = await renderAndOpenBudgetsTab();

    expect(rowFor(panel, "alice@example.com")).toHaveTextContent("Never");
  });

  it("surfaces a failed lookup instead of an empty budget table", async () => {
    const failed = { data: undefined, isLoading: false, isError: true, error: new Error("Admin-only endpoint") };
    apiMocks.useQuery.mockReturnValue(failed);
    const panel = await renderAndOpenBudgetsTab();

    expect(panel.getByText("Could not load budgets")).toBeInTheDocument();
    expect(panel.getByText("Admin-only endpoint")).toBeInTheDocument();
    expect(panel.queryByRole("table")).not.toBeInTheDocument();
  });

  it("tells the user nothing applies when the server returns no budgets", async () => {
    mockBudgets([]);
    const panel = await renderAndOpenBudgetsTab();

    expect(panel.getByText("No budgets apply to this key.")).toBeInTheDocument();
    expect(panel.queryByTestId("key-budget-blocking")).not.toBeInTheDocument();
  });
});
