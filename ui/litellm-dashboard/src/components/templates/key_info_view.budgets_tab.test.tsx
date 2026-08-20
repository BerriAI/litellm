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
  spend_state: "live",
  notes: [],
} as KeyBudgetEntry;

// `code` and `severity` are the contract; `text` is explicitly free to be reworded, so every note
// text below is synthetic. Copying the resolver's prose only buys tests that go stale silently.
const noteText = (code: string): string => `synthetic ${code} caveat`;

const ALERT_ONLY_NOTE = {
  code: "alert_only",
  severity: "info",
  text: noteText("alert_only"),
} as const;

const PROJECT_DEAD_NOTE = {
  code: "project_spend_not_tracked",
  severity: "warning",
  text: noteText("project_spend_not_tracked"),
} as const;

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
  notes: [ALERT_ONLY_NOTE],
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

// The widest row the endpoint can emit is an end_user budget the reservation layer tightened, on a
// proxy with a custom auth callable: three notes rather than one sentence, in the server's order.
const WORST_CASE_NOTES = [
  { code: "reservation_blocks_at_limit", severity: "info", text: noteText("reservation_blocks_at_limit") },
  { code: "end_user_route_only", severity: "warning", text: noteText("end_user_route_only") },
  {
    code: "custom_auth_may_override_end_user_cap",
    severity: "warning",
    text: noteText("custom_auth_may_override_end_user_cap"),
  },
] as const;

// Deliberately far past anything the endpoint emits, so no rewording on the server can move this
// guard. Only its length and the absence of clipping matter, never its wording.
const OVERLONG_NOTE_TEXT = `${"a caveat clause that keeps going ".repeat(16)}end`;

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
    expect(softRow).toHaveTextContent(ALERT_ONLY_NOTE.text);
  });

  it("does not render a spend the server could not read as a healthy $0.00", async () => {
    const unreadable: KeyBudgetEntry = {
      ...UNCONFIGURED_BUDGET,
      scope: "tag",
      entity_type: "tag",
      entity_id: "prod",
      entity_label: "prod",
      max_budget: 1000,
      spend: null,
      spend_state: "unavailable",
      remaining: null,
      source: "budget_table:b-tag",
      status: "ok",
    };
    mockBudgets([unreadable]);
    const panel = await renderAndOpenBudgetsTab();

    const row = rowFor(panel, "prod");
    expect(row).toHaveTextContent("Unknown of $1,000.00");
    expect(row).not.toHaveTextContent("$0.00");
    // The meter is the part that lies loudest: drawn at 0% it reads as untouched headroom.
    expect(within(row).queryByRole("meter")).not.toBeInTheDocument();
    expect(row).toHaveTextContent("Blocks at ≥ $1,000.00");
  });

  it("shows a genuine no-counter-yet zero as $0.00 with its meter, not as unknown", async () => {
    const cold: KeyBudgetEntry = {
      ...UNCONFIGURED_BUDGET,
      scope: "key_model",
      entity_type: "key",
      entity_label: "claude-opus-5",
      max_budget: 40,
      spend: null,
      spend_state: "no_counter",
      remaining: 40,
      source: "key.model_max_budget",
      status: "ok",
    };
    mockBudgets([cold]);
    const panel = await renderAndOpenBudgetsTab();

    const row = rowFor(panel, "claude-opus-5");
    expect(row).toHaveTextContent("$0.00 of $40.00");
    expect(row).not.toHaveTextContent("Unknown");
    expect(within(row).getByRole("meter")).toBeInTheDocument();
  });

  it("treats a spend state this build predates as unreadable rather than as a confident number", async () => {
    const future: KeyBudgetEntry = {
      ...UNCONFIGURED_BUDGET,
      scope: "team",
      entity_type: "team",
      entity_label: "Platform",
      max_budget: 500,
      spend: 20,
      remaining: 480,
      source: "team.max_budget",
      status: "ok",
      spend_state: "reconciling" as KeyBudgetEntry["spend_state"],
    };
    mockBudgets([future]);
    const panel = await renderAndOpenBudgetsTab();

    const row = rowFor(panel, "Platform");
    expect(row).toHaveTextContent("Unknown of $500.00");
    expect(within(row).queryByRole("meter")).not.toBeInTheDocument();
  });

  it("renders every note on the widest row as its own line, none of them dropped", async () => {
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
      notes: [...WORST_CASE_NOTES],
    };
    mockBudgets([endUser]);
    const panel = await renderAndOpenBudgetsTab();

    const rendered = WORST_CASE_NOTES.map((note) => panel.getByText(note.text));
    expect(rendered).toHaveLength(3);
    // Separate elements, not one joined blob, so each caveat can carry its own severity.
    expect(new Set(rendered).size).toBe(3);
  });

  it("renders a note far longer than any the endpoint emits without dropping any of it", async () => {
    const reserved: KeyBudgetEntry = {
      ...UNCONFIGURED_BUDGET,
      scope: "team",
      entity_type: "team",
      entity_label: "Platform",
      max_budget: 300,
      spend: 120,
      remaining: 180,
      source: "team.max_budget",
      status: "ok",
      notes: [{ code: "rolling_window", severity: "info", text: OVERLONG_NOTE_TEXT }],
    };
    mockBudgets([reserved]);
    const panel = await renderAndOpenBudgetsTab();

    expect(OVERLONG_NOTE_TEXT.length).toBeGreaterThan(500);
    const note = panel.getByText(OVERLONG_NOTE_TEXT);

    // jsdom has no layout, so nothing here can prove the text is visually unclipped. Asserting the
    // absence of the clipping utilities is the only mechanical guard against re-truncating a note.
    expect(note).not.toHaveClass("truncate");
    expect(note.className).not.toMatch(/line-clamp|overflow-hidden|whitespace-nowrap/);
    expect(note).not.toHaveAttribute("title");
  });

  it("keeps two per-model rows on one cap apart by the request model each measures", async () => {
    const direct: KeyBudgetEntry = {
      ...UNCONFIGURED_BUDGET,
      scope: "key_model",
      entity_type: "key",
      entity_id: "claude-opus-5",
      entity_label: "claude-opus-5",
      max_budget: 40,
      spend: 5,
      remaining: 35,
      comparison: ">",
      source: "key.model_max_budget[claude-opus-5]",
      status: "ok",
    };
    const routed: KeyBudgetEntry = {
      ...direct,
      entity_id: "bedrock/claude-opus-5",
      spend: 38,
      remaining: 2,
    };
    mockBudgets([direct, routed]);
    const panel = await renderAndOpenBudgetsTab();

    const routedRow = rowFor(panel, "bedrock/claude-opus-5");
    expect(routedRow).toHaveTextContent("$38.0000 of $40.00");
    // The cap is repeated on both rows, so it cannot be what tells them apart.
    expect(panel.getAllByText("claude-opus-5")).toHaveLength(2);
    expect(routedRow).toHaveTextContent("claude-opus-5");

    const [, ...dataRows] = panel.getAllByRole("row");
    expect(dataRows).toHaveLength(2);
    expect(dataRows.filter((row) => row.textContent?.includes("bedrock/claude-opus-5"))).toHaveLength(1);
  });

  it("floats the one exceeded per-model row above its healthy siblings on the same cap", async () => {
    const cap = {
      ...UNCONFIGURED_BUDGET,
      scope: "key_model",
      entity_type: "key",
      entity_label: "claude-opus-5",
      max_budget: 40,
      comparison: ">",
      source: "key.model_max_budget[claude-opus-5]",
    } as const;
    const healthy: KeyBudgetEntry = { ...cap, entity_id: "claude-opus-5", spend: 1, remaining: 39, status: "ok" };
    const alsoHealthy: KeyBudgetEntry = {
      ...cap,
      entity_id: "vertex_ai/claude-opus-5",
      spend: 2,
      remaining: 38,
      status: "ok",
    };
    const over: KeyBudgetEntry = {
      ...cap,
      entity_id: "bedrock/claude-opus-5",
      spend: 41,
      remaining: -1,
      status: "exceeded",
    };
    mockBudgets([healthy, alsoHealthy, over]);
    const panel = await renderAndOpenBudgetsTab();

    const [, ...dataRows] = panel.getAllByRole("row");
    expect(dataRows[0]).toHaveTextContent("bedrock/claude-opus-5");
    expect(within(dataRows[0]).getByTestId("key-budget-blocking")).toBeInTheDocument();
    // Siblings keep the server's order behind it rather than being reshuffled among themselves.
    expect(dataRows[1]).toHaveTextContent("claude-opus-5");
    expect(dataRows[2]).toHaveTextContent("vertex_ai/claude-opus-5");
    expect(panel.getAllByTestId("key-budget-blocking")).toHaveLength(1);
  });

  it("renders notes in the order the server sent them, most specific to these numbers first", async () => {
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
      notes: [...WORST_CASE_NOTES],
    };
    mockBudgets([endUser]);
    const panel = await renderAndOpenBudgetsTab();

    const texts = WORST_CASE_NOTES.map((note) => note.text);
    const rendered = texts.map((text) => panel.getByText(text));
    const positions = rendered.map((node) => Array.from(node.parentElement?.children ?? []).indexOf(node));
    expect(positions).toStrictEqual([...positions].sort((a, b) => a - b));
    // Server order is meaningful: the reservation note explains the comparison the row renders.
    expect(positions[0]).toBeLessThan(positions[2]);
  });

  it("says a throttled budget slows requests rather than claiming it blocks them", async () => {
    const throttled: KeyBudgetEntry = {
      ...UNCONFIGURED_BUDGET,
      entity_id: "ci-runner",
      entity_label: "ci-runner",
      max_budget: 100,
      spend: 140,
      remaining: -40,
      source: "key.max_budget",
      status: "exceeded",
      notes: [
        {
          code: "throttled_instead_of_blocked",
          severity: "warning",
          text: noteText("throttled_instead_of_blocked"),
        },
      ],
    };
    mockBudgets([throttled]);
    const panel = await renderAndOpenBudgetsTab();

    const row = rowFor(panel, "ci-runner");
    expect(within(row).getByText("Throttles requests")).toBeInTheDocument();
    expect(within(row).queryByText("Blocks requests")).not.toBeInTheDocument();
    expect(row).toHaveTextContent("Throttles at ≥ $100.00");
    expect(within(row).getByText("Exceeded (throttling)")).toBeInTheDocument();
    expect(within(row).queryByTestId("key-budget-blocking")).not.toBeInTheDocument();
  });

  it("marks a budget that structurally cannot trip and sinks it below every live row", async () => {
    const dead: KeyBudgetEntry = {
      ...UNCONFIGURED_BUDGET,
      scope: "project",
      entity_type: "project",
      entity_label: "checkout",
      max_budget: 300,
      spend: 0,
      remaining: 300,
      source: "project.budget_id",
      status: "ok",
      notes: [PROJECT_DEAD_NOTE],
    };
    mockBudgets([dead, TEAM_MEMBER_BLOCKING, USER_WITHIN_BUDGET, KEY_UNLIMITED]);
    const panel = await renderAndOpenBudgetsTab();

    const deadRow = rowFor(panel, "checkout");
    expect(within(deadRow).getByText("Cannot trip")).toBeInTheDocument();
    expect(within(deadRow).queryByText("Within budget")).not.toBeInTheDocument();
    expect(deadRow).toHaveTextContent(PROJECT_DEAD_NOTE.text);

    const [, ...dataRows] = panel.getAllByRole("row");
    expect(dataRows[dataRows.length - 1]).toHaveTextContent("checkout");
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
