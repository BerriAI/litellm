import {
  act,
  fireEvent,
  renderWithProviders as render,
  screen,
  testQueryClient,
  waitFor,
} from "../../../../../../tests/test-utils";
import userEvent, { PointerEventsCheckLevel } from "@testing-library/user-event";
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { Profiler } from "react";
import UserInfoView from "./user_info_view";
import type { DailyData, SpendMetrics } from "@/components/UsagePage/types";
import type { AutoRouterBenchmarksResponse } from "@/app/(dashboard)/cost-optimization/_components/autoRouterBenchmarks";

const mockTeamMemberAddCall = vi.fn();
const mockTeamMemberDeleteCall = vi.fn();
const mockTeamListCall = vi.fn();
const mockUserGetInfoV2 = vi.fn();
const mockTeamInfoCall = vi.fn();
const mockUserUpdateUserCall = vi.fn();
const mockFetchMCPServers = vi.fn();
const mockListMCPTools = vi.fn();
const mockUserDailyActivityCall = vi.fn();
const mockUserDailyActivityAggregatedCall = vi.fn();

const MCP_SERVER = { server_id: "srv-1", server_name: "GitHub MCP", alias: "GitHub MCP" };

const MOCK_USER_DATA = {
  user_id: "user-123",
  user_email: "test@example.com",
  user_alias: "Test Alias",
  user_role: "admin",
  spend: 0,
  max_budget: 100,
  models: [],
  budget_duration: "30d",
  budget_reset_at: null,
  metadata: {},
  created_at: "2025-01-01T00:00:00.000Z",
  updated_at: "2025-01-02T00:00:00.000Z",
  sso_user_id: null,
  teams: ["team-1", "team-2"],
  object_permission: {
    mcp_servers: ["srv-1"],
    mcp_access_groups: ["dev-group"],
    mcp_tool_permissions: { "srv-1": ["list_issues"] },
  },
};

const MOCK_USER_DATA_NO_TEAMS = {
  ...MOCK_USER_DATA,
  teams: [],
};

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn() }),
  usePathname: () => "/users",
  useSearchParams: () => new URLSearchParams(window.location.search),
}));

vi.mock("@/components/networking", async (importOriginal) => {
  const original = await importOriginal<typeof import("@/components/networking")>();
  return {
    formatDate: original.formatDate,
    serverRootPath: "/",
    userGetInfoV2: (...args: unknown[]) => mockUserGetInfoV2(...args),
    userDailyActivityCall: (...args: unknown[]) => mockUserDailyActivityCall(...args),
    userDailyActivityAggregatedCall: (...args: unknown[]) => mockUserDailyActivityAggregatedCall(...args),
    userDeleteCall: vi.fn(),
    userUpdateUserCall: (...args: unknown[]) => mockUserUpdateUserCall(...args),
    modelAvailableCall: vi.fn().mockResolvedValue({ data: [] }),
    modelInfoCall: vi.fn().mockResolvedValue({ data: [], total_pages: 1 }),
    invitationCreateCall: vi.fn(),
    teamInfoCall: (...args: unknown[]) => mockTeamInfoCall(...args),
    teamListCall: (...args: unknown[]) => mockTeamListCall(...args),
    teamMemberAddCall: (...args: unknown[]) => mockTeamMemberAddCall(...args),
    teamMemberDeleteCall: (...args: unknown[]) => mockTeamMemberDeleteCall(...args),
    getProxyBaseUrl: () => "https://litellm.test",
    fetchMCPServers: (...args: unknown[]) => mockFetchMCPServers(...args),
    fetchMCPToolsets: vi.fn().mockResolvedValue([]),
    listMCPTools: (...args: unknown[]) => mockListMCPTools(...args),
  };
});

vi.mock("@/app/(dashboard)/hooks/mcpServers/useMCPServers", () => ({
  useMCPServers: () => ({ data: [MCP_SERVER], isLoading: false }),
}));
vi.mock("@/app/(dashboard)/hooks/mcpServers/useMCPAccessGroups", () => ({
  useMCPAccessGroups: () => ({ data: ["dev-group"], isLoading: false }),
}));
vi.mock("@/app/(dashboard)/hooks/mcpServers/useMCPToolsets", () => ({
  useMCPToolsets: () => ({ data: [], isLoading: false }),
}));
describe("UserInfoView add-to-team form", () => {
  const defaultProps = {
    userId: "user-123",
    onClose: vi.fn(),
    accessToken: "test-token",
    userRole: "proxy_admin" as string | null,
    possibleUIRoles: null,
  };

  beforeEach(() => {
    vi.clearAllMocks();
    mockUserGetInfoV2.mockResolvedValue(MOCK_USER_DATA);
    mockTeamInfoCall.mockImplementation((_token: string, teamId: string) => {
      const teamMap: Record<string, any> = {
        "team-1": { team_id: "team-1", team_info: { team_alias: "Alpha Team" } },
        "team-2": { team_id: "team-2", team_info: { team_alias: "Beta Team" } },
        "team-3": { team_id: "team-3", team_info: { team_alias: "Gamma Team" } },
      };
      return Promise.resolve(teamMap[teamId] || { team_id: teamId, team_info: { team_alias: null } });
    });
    mockTeamListCall.mockResolvedValue([
      { team_id: "team-1", team_alias: "Alpha Team" },
      { team_id: "team-2", team_alias: "Beta Team" },
      { team_id: "team-3", team_alias: "Gamma Team" },
    ]);
    mockTeamMemberAddCall.mockResolvedValue({});
    mockFetchMCPServers.mockResolvedValue([MCP_SERVER]);
    mockListMCPTools.mockResolvedValue({ tools: [] });
  });

  const setup = () => userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });

  const openAddTeam = async (user: ReturnType<typeof userEvent.setup>, anchorTeam = "Alpha Team") => {
    render(<UserInfoView {...defaultProps} />);
    await screen.findByText(anchorTeam);
    await user.click(screen.getByText("Add Team"));
    await screen.findByText("Add User to Team");
  };

  const teamField = () => screen.getAllByRole("combobox")[0];
  const roleField = () => screen.getAllByRole("combobox")[1];
  const submitButton = () => screen.getByRole("button", { name: /Add to Team|Adding\.\.\./i });

  const chooseTeam = async (user: ReturnType<typeof userEvent.setup>, alias: string) => {
    await user.click(teamField());
    await user.click(await screen.findByTitle(alias));
  };

  // handleUserUpdate refreshes the local copy field by field rather than refetching,
  // so a field it forgets reads back stale the next time the form is opened and the
  // operator sees the save they just made apparently undone.
  describe("per-model budgets survive a save", () => {
    // Edit Settings lives on the details tab and is gated on write access.
    const budgetProps = {
      ...defaultProps,
      userRole: "Admin",
      initialTab: 1,
    };

    const openEditor = async (user: ReturnType<typeof userEvent.setup>) => {
      await user.click(await screen.findByRole("button", { name: /edit settings/i }));
      return screen.findByPlaceholderText("Max spend ($)");
    };

    beforeEach(() => {
      mockUserGetInfoV2.mockResolvedValue({
        ...MOCK_USER_DATA,
        model_max_budget: { "gpt-4": { budget_limit: 5, time_period: "30d" } },
      });
      mockUserUpdateUserCall.mockResolvedValue({});
    });

    it("shows the saved cap, not the pre-save one, when the form is reopened", async () => {
      const user = setup();
      render(<UserInfoView {...budgetProps} />);

      fireEvent.change(await openEditor(user), { target: { value: "42" } });
      await user.click(screen.getByRole("button", { name: /save changes/i }));

      await waitFor(() => {
        expect(mockUserUpdateUserCall).toHaveBeenCalled();
      });
      expect(mockUserUpdateUserCall.mock.calls[0][1].model_max_budget).toEqual({
        "gpt-4": { budget_limit: 42, time_period: "30d" },
      });

      expect(await openEditor(user)).toHaveValue(42);
    });

    it("should keep Unlimited selected after saving and reopening the user", async () => {
      const user = setup();
      render(<UserInfoView {...budgetProps} />);

      await openEditor(user);
      await user.click(screen.getByRole("checkbox", { name: "Unlimited Budget" }));
      await user.click(screen.getByRole("button", { name: /save changes/i }));

      await waitFor(() => expect(mockUserUpdateUserCall).toHaveBeenCalled());
      expect(mockUserUpdateUserCall.mock.calls[0][1]).toMatchObject({ max_budget: null });
      await openEditor(user);
      expect(screen.getByRole("checkbox", { name: "Unlimited Budget" })).toBeChecked();
    });

    it("should keep a cleared reset period after saving and reopening the user", async () => {
      const user = setup();
      render(<UserInfoView {...budgetProps} />);

      await openEditor(user);
      await user.click(screen.getByRole("combobox", { name: "Reset Budget" }));
      await user.click(await screen.findByRole("option", { name: "n/a" }));
      await user.click(screen.getByRole("button", { name: /save changes/i }));

      await waitFor(() => expect(mockUserUpdateUserCall).toHaveBeenCalled());
      expect(mockUserUpdateUserCall.mock.calls[0][1]).toMatchObject({ budget_duration: null });
      await openEditor(user);
      expect(screen.getByRole("combobox", { name: "Reset Budget" })).toHaveTextContent("n/a");
    });
  });

  it("offers only the teams the user is not already a member of", async () => {
    const user = setup();
    await openAddTeam(user);

    await user.click(teamField());

    expect(await screen.findByTitle("Gamma Team")).toBeInTheDocument();
    expect(screen.queryByTitle("Alpha Team")).not.toBeInTheDocument();
    expect(screen.queryByTitle("Beta Team")).not.toBeInTheDocument();
  });

  it("shows the default role name on the trigger rather than a blank or raw value", async () => {
    const user = userEvent.setup();
    await openAddTeam(user);

    expect(screen.getAllByRole("combobox")[1]).toHaveTextContent("user");
  });

  it("adds the user to the chosen team with the default role", async () => {
    const user = setup();
    await openAddTeam(user);

    await chooseTeam(user, "Gamma Team");
    await user.click(submitButton());

    await waitFor(() => expect(mockTeamMemberAddCall).toHaveBeenCalledTimes(1));
    expect(mockTeamMemberAddCall).toHaveBeenCalledWith("test-token", "team-3", {
      role: "user",
      user_id: "user-123",
    });
  });

  it("sends the admin role when it is chosen", async () => {
    const user = setup();
    await openAddTeam(user);

    await chooseTeam(user, "Gamma Team");
    await user.click(roleField());
    await user.click(await screen.findByText("admin", { selector: "span.font-medium" }));
    await user.click(submitButton());

    await waitFor(() => expect(mockTeamMemberAddCall).toHaveBeenCalledTimes(1));
    expect(mockTeamMemberAddCall).toHaveBeenCalledWith("test-token", "team-3", {
      role: "admin",
      user_id: "user-123",
    });
  });

  it("keeps the submit disabled until a team is chosen", async () => {
    const user = setup();
    await openAddTeam(user);

    expect(submitButton()).toBeDisabled();

    await chooseTeam(user, "Gamma Team");

    expect(submitButton()).toBeEnabled();
  });

  it("sends nothing while no team is chosen", async () => {
    const user = setup();
    await openAddTeam(user);

    await user.click(submitButton());

    expect(mockTeamMemberAddCall).not.toHaveBeenCalled();
  });

  it("narrows the team list to the aliases matching what was typed", async () => {
    mockUserGetInfoV2.mockResolvedValue(MOCK_USER_DATA_NO_TEAMS);
    const user = setup();
    render(<UserInfoView {...defaultProps} />);
    await screen.findByText("Add Team");
    await user.click(screen.getByText("Add Team"));
    await screen.findByText("Add User to Team");

    await user.click(teamField());
    await screen.findByTitle("Alpha Team");

    await user.keyboard("Gam");

    expect(await screen.findByTitle("Gamma Team")).toBeInTheDocument();
    await waitFor(() => expect(screen.queryByTitle("Alpha Team")).not.toBeInTheDocument());
  });

  it("keeps the dialog open when the member could not be added", async () => {
    mockTeamMemberAddCall.mockRejectedValue(new Error("nope"));
    const user = setup();
    await openAddTeam(user);

    await chooseTeam(user, "Gamma Team");
    await user.click(submitButton());

    await waitFor(() => expect(mockTeamMemberAddCall).toHaveBeenCalledTimes(1));
    expect(screen.getByText("Add User to Team")).toBeInTheDocument();
  });
});

const savingsDay = (date: string, metrics: Partial<SpendMetrics>): DailyData => ({
  date,
  metrics: {
    spend: 0,
    prompt_tokens: 0,
    completion_tokens: 0,
    total_tokens: 0,
    api_requests: 1,
    successful_requests: 1,
    failed_requests: 0,
    cache_read_input_tokens: 0,
    cache_creation_input_tokens: 0,
    ...metrics,
  },
  breakdown: { models: {}, model_groups: {}, mcp_servers: {}, providers: {}, api_keys: {}, entities: {} },
});

const savingsResponse = (results: DailyData[]) => ({
  results,
  metadata: { total_pages: 1, has_more: false, page: 1 },
});

const routerUsageResponse = (saved: number): AutoRouterBenchmarksResponse => ({
  start_date: "2026-09-01",
  end_date: "2026-09-19",
  routers_in_scope: 0,
  groups: [],
  totals: {
    sessions: 2,
    turns: 2,
    avg_turns_per_session: 1,
    avg_session_seconds: 0,
    avg_tokens_per_session: 100,
    spend: null,
    llm_spend: null,
    cost_coverage: "unavailable",
    cost_requests: null,
    savings_estimated_turns: 2,
    savings_estimated_actual_spend: 10,
    classifier_cost: null,
    saved_spend: saved,
    baseline_spend: null,
    saved_pct: null,
    saved_per_session: 3,
    cache: {
      coverage_pct: 100,
      hit_rate_pct: 0,
      same_model: { turns: 0, hits: 0, hit_rate_pct: 0 },
      first_visit: { turns: 2, hits: 0, hit_rate_pct: 0 },
      return_to_tier: { turns: 0, hits: 0, hit_rate_pct: 0 },
      unordered_turns: 0,
      return_misses_expired: 0,
      return_misses_within_ttl: 0,
      return_misses_unknown: 0,
      ttl_5m_turns: 0,
      ttl_1h_turns: 0,
    },
  },
});

describe("UserInfoView auto-router usage", () => {
  const props = {
    userId: "user-123",
    onClose: vi.fn(),
    accessToken: "admin-token",
    userRole: "proxy_admin",
    possibleUIRoles: null,
  };
  const mockFetch = vi.fn<typeof fetch>();

  beforeEach(() => {
    testQueryClient.clear();
    vi.clearAllMocks();
    mockUserGetInfoV2.mockImplementation((_token: string, userId: string) =>
      Promise.resolve({ ...MOCK_USER_DATA_NO_TEAMS, user_id: userId }),
    );
    mockFetch.mockReset().mockResolvedValue(Response.json(routerUsageResponse(42)));
    vi.stubGlobal("fetch", mockFetch);
  });

  afterEach(() => {
    testQueryClient.clear();
    vi.unstubAllGlobals();
  });

  it.each(["proxy_admin", "proxy_admin_viewer"])(
    "loads selected-user usage lazily for %s without a key filter",
    async (userRole) => {
      const user = userEvent.setup();
      render(<UserInfoView {...props} userRole={userRole} />);
      const tab = await screen.findByRole("tab", { name: "Auto-router usage" });
      expect(mockFetch).not.toHaveBeenCalled();
      await user.click(tab);

      expect(await screen.findByText("$42.00")).toBeInTheDocument();
      const request = mockFetch.mock.calls[0][0] as Request;
      const params = new URL(request.url).searchParams;
      expect(params.get("user_id")).toBe("user-123");
      expect(params.has("api_key")).toBe(false);
      expect(
        screen.getByText("Usage for this user across API keys and JWT-authenticated requests."),
      ).toBeInTheDocument();
      expect(screen.getByText("$3.00")).toBeInTheDocument();
      expect(screen.getByText("Estimated baseline spend")).toBeInTheDocument();
      expect(screen.getAllByRole("definition").map((node) => node.textContent)).toEqual([
        "Unavailable",
        "Unavailable",
        "Unavailable",
        "Unavailable",
      ]);
    },
  );

  it("switches query scope without displaying the previous user's usage", async () => {
    const nextUser = Promise.withResolvers<Response>();
    mockFetch.mockResolvedValueOnce(Response.json(routerUsageResponse(42))).mockReturnValue(nextUser.promise);
    const user = userEvent.setup();
    const { rerender } = render(<UserInfoView {...props} />);
    await user.click(await screen.findByRole("tab", { name: "Auto-router usage" }));
    expect(await screen.findByText("$42.00")).toBeInTheDocument();

    rerender(<UserInfoView {...props} userId="user-456" />);
    expect(screen.getByText("Loading auto-router usage...")).toBeInTheDocument();
    expect(screen.queryByText("$42.00")).not.toBeInTheDocument();
    await act(async () => nextUser.resolve(Response.json(routerUsageResponse(-7))));
    expect(await screen.findByText("-$7.00")).toBeInTheDocument();
    expect(
      mockFetch.mock.calls.map(([request]) => new URL((request as Request).url).searchParams.get("user_id")),
    ).toEqual(["user-123", "user-456"]);
  });

  it.each(["internal_user", "org_admin", null])("keeps the admin-only tab unavailable to %s", async (userRole) => {
    render(<UserInfoView {...props} userRole={userRole} />);
    await screen.findByRole("tab", { name: "Overview" });
    expect(screen.queryByRole("tab", { name: "Auto-router usage" })).not.toBeInTheDocument();
    expect(mockFetch).not.toHaveBeenCalled();
  });

  it("never turns an absent user ID into a deployment-wide request", async () => {
    const user = userEvent.setup();
    render(<UserInfoView {...props} userId="" />);
    await user.click(await screen.findByRole("tab", { name: "Auto-router usage" }));
    expect(screen.getByRole("alert")).toHaveTextContent("this user has no ID");
    expect(mockFetch).not.toHaveBeenCalled();
  });
});

describe("UserInfoView savings", () => {
  const props = {
    userId: "user-123",
    onClose: vi.fn(),
    accessToken: "admin-token",
    userRole: "proxy_admin",
    possibleUIRoles: null,
  };

  beforeEach(() => {
    vi.clearAllMocks();
    mockUserGetInfoV2.mockImplementation((_token: string, userId: string) =>
      Promise.resolve({ ...MOCK_USER_DATA_NO_TEAMS, user_id: userId }),
    );
    mockUserDailyActivityAggregatedCall.mockReset().mockResolvedValue(savingsResponse([]));
    mockUserDailyActivityCall.mockReset().mockResolvedValue(savingsResponse([]));
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it.each(["internal_user", "org_admin", "team_admin"])(
    "only offers self savings to %s and stops querying after switching to another user",
    async (userRole) => {
      const user = userEvent.setup();
      const { rerender } = render(<UserInfoView {...props} userId="user-1" userRole={userRole} />);
      await user.click(await screen.findByRole("tab", { name: "Savings" }));
      expect(await screen.findByText("No usage recorded for this user in this range.")).toBeInTheDocument();
      expect(mockUserDailyActivityAggregatedCall.mock.calls[0][3]).toBe("user-1");

      mockUserDailyActivityAggregatedCall.mockClear();
      mockUserDailyActivityCall.mockClear();
      rerender(<UserInfoView {...props} userId="another-user" userRole={userRole} />);
      await screen.findAllByText("another-user");
      expect(screen.getByRole("tab", { name: "Overview" })).toHaveAttribute("aria-selected", "true");
      expect(screen.queryByRole("tab", { name: "Savings" })).not.toBeInTheDocument();
      expect(screen.queryByText("No usage recorded for this user in this range.")).not.toBeInTheDocument();
      expect(mockUserDailyActivityAggregatedCall).not.toHaveBeenCalled();
      expect(mockUserDailyActivityCall).not.toHaveBeenCalled();
    },
  );

  it("loads selected user savings without a key filter, including losses", async () => {
    const firstDay: Partial<SpendMetrics> = {
      compression_savings_spend: 1.5,
      gateway_injected_caching_savings_spend: 0.1,
      prompt_caching_savings_spend: 0.25,
      autorouter_savings_spend: -1,
    };
    const secondDay: Partial<SpendMetrics> = {
      compression_savings_spend: 0.5,
      gateway_injected_caching_savings_spend: 0.3,
      prompt_caching_savings_spend: 0.75,
      autorouter_savings_spend: -2,
    };
    mockUserDailyActivityAggregatedCall.mockResolvedValue(
      savingsResponse([savingsDay("2026-09-18", firstDay), savingsDay("2026-09-19", secondDay)]),
    );
    const user = userEvent.setup();
    render(<UserInfoView {...props} />);
    const savingsTab = await screen.findByRole("tab", { name: "Savings" });
    expect(mockUserDailyActivityAggregatedCall).not.toHaveBeenCalled();
    expect(mockUserDailyActivityCall).not.toHaveBeenCalled();

    await user.click(savingsTab);

    expect(await screen.findByTestId("summary-card-total-recorded-savings")).toHaveTextContent("-$0.6000");
    expect(screen.getByTestId("summary-card-compression-savings")).toHaveTextContent("$2.00");
    expect(screen.getByTestId("summary-card-prompt-caching-savings")).toHaveTextContent("$0.4000");
    expect(screen.getByTestId("summary-card-prompt-caching-savings")).toHaveTextContent("$1.00Total");
    expect(screen.getByTestId("summary-card-auto-router-savings")).toHaveTextContent("-$3.00");
    expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalledExactlyOnceWith(
      "admin-token",
      expect.any(Date),
      expect.any(Date),
      "user-123",
      true,
      null,
    );
    expect(screen.getByTestId("user-savings-scope-note")).toHaveTextContent("JWT-authenticated requests");
    await user.click(screen.getByRole("tab", { name: "Per day" }));
    expect(screen.getByRole("tab", { name: "Per day" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByTestId("summary-card-total-recorded-savings")).toHaveTextContent("-$0.6000");
  });

  it("removes the prior user's savings while the newly selected user's results are loading", async () => {
    const nextUser = Promise.withResolvers<ReturnType<typeof savingsResponse>>();
    mockUserDailyActivityAggregatedCall
      .mockResolvedValueOnce(savingsResponse([savingsDay("2026-09-19", { compression_savings_spend: 42 })]))
      .mockReturnValueOnce(nextUser.promise);
    const user = userEvent.setup();
    const { rerender } = render(<UserInfoView {...props} />);
    await user.click(await screen.findByRole("tab", { name: "Savings" }));
    expect(await screen.findByTestId("summary-card-total-recorded-savings")).toHaveTextContent("$42.00");

    rerender(<UserInfoView {...props} userId="user-456" />);

    expect(await screen.findByTestId("user-savings-empty")).toHaveTextContent("Loading savings");
    expect(screen.queryByTestId("summary-card-total-recorded-savings")).not.toBeInTheDocument();
    expect(mockUserDailyActivityAggregatedCall).toHaveBeenLastCalledWith(
      "admin-token",
      expect.any(Date),
      expect.any(Date),
      "user-456",
      true,
      null,
    );
    await act(async () => {
      nextUser.resolve(savingsResponse([savingsDay("2026-09-19", { autorouter_savings_spend: -7 })]));
    });
    expect(await screen.findByTestId("summary-card-total-recorded-savings")).toHaveTextContent("-$7.00");
    expect(screen.queryByText("$42.00")).not.toBeInTheDocument();
  });

  it("never commits the previous range's savings under the newly selected dates", async () => {
    vi.stubGlobal("requestIdleCallback", (callback: IdleRequestCallback) =>
      window.setTimeout(() => callback({ didTimeout: false, timeRemaining: () => 0 }), 0),
    );
    const nextRange = Promise.withResolvers<ReturnType<typeof savingsResponse>>();
    mockUserDailyActivityAggregatedCall
      .mockResolvedValueOnce(savingsResponse([savingsDay("2026-09-19", { compression_savings_spend: 42 })]))
      .mockReturnValue(nextRange.promise);
    const committedTotals: Array<string | null> = [];
    const captureNewRange = () => {
      if (screen.queryByText("Running total saved · Sep 1 – Sep 2 (UTC)")) {
        committedTotals.push(screen.queryByTestId("summary-card-total-recorded-savings")?.textContent ?? null);
      }
    };
    const user = userEvent.setup();
    render(
      <Profiler id="user-savings" onRender={captureNewRange}>
        <UserInfoView {...props} />
      </Profiler>,
    );
    await user.click(await screen.findByRole("tab", { name: "Savings" }));
    expect(await screen.findByTestId("summary-card-total-recorded-savings")).toHaveTextContent("$42.00");

    await user.click(screen.getByRole("button", { name: / - / }));
    const [startDateInput, endDateInput] = screen.getAllByDisplayValue(/^\d{4}-\d{2}-\d{2}$/);
    fireEvent.change(startDateInput, { target: { value: "2026-09-01" } });
    fireEvent.change(endDateInput, { target: { value: "2026-09-02" } });
    await user.click(screen.getByRole("button", { name: "Apply" }));

    expect(committedTotals.length).toBeGreaterThan(0);
    expect(committedTotals.every((total) => total === null)).toBe(true);
    expect(screen.getByTestId("user-savings-empty")).toHaveTextContent("Loading savings");
    await act(async () => {
      nextRange.resolve(savingsResponse([savingsDay("2026-09-02", { autorouter_savings_spend: -7 })]));
    });
    expect(await screen.findByTestId("summary-card-total-recorded-savings")).toHaveTextContent("-$7.00");
  });

  it("reports an incomplete paginated read as unavailable instead of displaying a partial savings total", async () => {
    mockUserDailyActivityAggregatedCall.mockRejectedValue(new Error("aggregated unavailable"));
    mockUserDailyActivityCall
      .mockResolvedValueOnce({
        results: [savingsDay("2026-09-19", { compression_savings_spend: 42 })],
        metadata: { total_pages: 2, has_more: true, page: 1 },
      })
      .mockRejectedValueOnce(new Error("next page unavailable"));
    const user = userEvent.setup();
    render(<UserInfoView {...props} />);
    await user.click(await screen.findByRole("tab", { name: "Savings" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("Savings are unavailable for this range");
    expect(mockUserDailyActivityCall).toHaveBeenLastCalledWith(
      "admin-token",
      expect.any(Date),
      expect.any(Date),
      2,
      "user-123",
      true,
      null,
    );
    expect(screen.queryByTestId("summary-card-total-recorded-savings")).not.toBeInTheDocument();
    expect(screen.queryByText(/No usage recorded/)).not.toBeInTheDocument();
  });

  it("distinguishes a user with no usage from an unavailable read", async () => {
    const user = userEvent.setup();
    render(<UserInfoView {...props} />);
    await user.click(await screen.findByRole("tab", { name: "Savings" }));

    expect(await screen.findByTestId("user-savings-empty")).toHaveTextContent("No usage recorded for this user");
    expect(screen.getByTestId("summary-card-total-recorded-savings")).toHaveTextContent("$0.00");
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it.each(["", "   "])("never queries an absent selected user ID (%j)", async (userId) => {
    const user = userEvent.setup();
    render(<UserInfoView {...props} userId={userId} />);
    await user.click(await screen.findByRole("tab", { name: "Savings" }));

    expect(screen.getByRole("alert")).toHaveTextContent("this user has no ID");
    expect(mockUserDailyActivityAggregatedCall).not.toHaveBeenCalled();
    expect(mockUserDailyActivityCall).not.toHaveBeenCalled();
  });
});
