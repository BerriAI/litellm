import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { registerBaseUrlGetter } from "@/lib/http/runtime";
import { EMPTY_DAILY_ACTIVITY_METADATA } from "@/components/UsagePage/dailyActivityApi";
import type { SpendMetrics } from "@/components/UsagePage/types";
import {
  act,
  chooseSelectOption,
  fireEvent,
  renderWithProviders,
  screen,
  testQueryClient,
  waitFor,
  within,
} from "../../../../../tests/test-utils";
import CustomerHierarchyUsagePage from "./page";

const metrics = (spend: number, tokens: number, requests: number): SpendMetrics => ({
  spend,
  api_requests: requests,
  successful_requests: requests,
  failed_requests: 0,
  total_tokens: tokens,
  prompt_tokens: tokens,
  completion_tokens: 0,
  cache_read_input_tokens: 0,
  cache_creation_input_tokens: 0,
});
const audit = {
  created_at: "2026-11-01T00:00:00Z",
  created_by: "admin",
  updated_at: "2026-11-01T00:00:00Z",
  updated_by: "admin",
  spend: 40,
};
const organization = {
  ...audit,
  customer_org_id: "org-1",
  alias: "Customer organization",
  budget: { max_budget: 100 },
  default_budgets: [{ budget_id: "member-standard", name: "Member standard" }],
};
const teams = [
  { ...audit, customer_team_id: "team-a", parent_org_id: "org-1", alias: "Product", budget: null },
  { ...audit, customer_team_id: "team-b", parent_org_id: "org-1", alias: "Operations", budget: null },
];
const members = [
  {
    ...audit,
    customer_member_id: "member-a",
    parent_team_ids: ["team-a"],
    alias: "Member One",
    budget_id: "member-standard",
    effective_budget: { max_budget: 10 },
  },
  {
    ...audit,
    customer_member_id: "member-b",
    parent_team_ids: ["team-b"],
    alias: "Member Two",
    effective_budget: { max_budget: 20 },
  },
];
const records = [
  ...[
    { date: "2026-03-01", spend: 42 },
    { date: "2025-12-31", spend: 100 },
  ].map(({ date, spend }) => ({
    organization: "org-1",
    team: "team-a",
    member: "member-a",
    model: "model-a",
    key: "hash-a",
    alias: "Assistant key",
    metrics: metrics(spend, 50, 1),
    date,
  })),
  {
    organization: "org-1",
    team: "team-a",
    member: "member-a",
    model: "model-a",
    key: "hash-a",
    alias: "Assistant key",
    metrics: {
      ...metrics(12, 100, 2),
      cache_read_input_tokens: 40,
      cache_creation_input_tokens: 20,
      cache_cost: 0.0123,
      priced_cache_tokens: 60,
    },
    date: "2026-11-01",
  },
  {
    organization: "org-1",
    team: "team-b",
    member: "member-b",
    model: "model-b",
    key: "hash-b",
    alias: "Search key",
    metrics: { ...metrics(3, 600, 5), cache_read_input_tokens: 500 },
    date: "2026-11-02",
  },
  {
    organization: "org-1",
    team: "team-a",
    member: "member-a",
    model: "model-a",
    key: "hash-a",
    alias: "Assistant key",
    metrics: metrics(8, 300, 1),
    date: "2026-11-10",
  },
  {
    organization: "org-1",
    team: "",
    member: "",
    model: "model-a",
    key: "hash-c",
    alias: "Organization key",
    metrics: metrics(7, 200, 1),
    date: "2026-11-10",
  },
];
const gatewayApps = [
  { ...audit, app_id: "app-chat", resource_type: "team", resource_id: "gateway-chat", alias: "chat" },
  { ...audit, app_id: "app-search", resource_type: "team", resource_id: "gateway-search", alias: "search" },
];
const appForRecord = (record: (typeof records)[number]) => (record.key === "hash-b" ? "app-search" : "app-chat");
const fetchMock = vi.fn<typeof fetch>();
const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
const requestUrl = (input: Parameters<typeof fetch>[0]) =>
  new URL(input instanceof Request ? input.url : String(input), "http://localhost");
const requestedUrls = () => fetchMock.mock.calls.map(([input]) => requestUrl(input));
const initialUrl = "?org=org-1&from=2026-11-01&to=2026-11-03&source=portal";
const tableRows = (name: string) => within(screen.getByRole("table", { name })).getAllByRole("row").slice(1);
const list = <T,>(data: T[]) => ({ data, meta: { total_count: data.length, page: 1, page_size: 100, total_pages: 1 } });
const selectBreakdown = (user: ReturnType<typeof userEvent.setup>, name: string) =>
  chooseSelectOption(user, screen.getByRole("combobox", { name: "Break down by" }), name);
const activityDimension = (resource: string | undefined): "organization" | "team" | "member" => {
  if (resource === "customer_organizations") return "organization";
  if (resource === "customer_teams") return "team";
  return "member";
};
const filterFor = (resource: string | undefined) =>
  ({
    organization: "customer_org_ids",
    team: "customer_team_ids",
    member: "customer_member_ids",
  })[activityDimension(resource)];
const activityUrl = (resource: string, id: string, endDate = "2026-11-03") => {
  const filter = filterFor(resource);
  return requestedUrls().findLast(
    (url) =>
      url.pathname === `/management/v1/${resource}/daily/activity` &&
      url.searchParams.get(filter) === id &&
      url.searchParams.get("end_date") === endDate,
  );
};
const activityIds = (url: URL, resource: string | undefined, aggregated: boolean) => {
  if (aggregated)
    return url.searchParams.get("customer_org_ids")?.split(",") ?? [url.searchParams.get("customer_org_id")];
  return url.searchParams.get(filterFor(resource))?.split(",") ?? [];
};
const createUsageHandler =
  ({ appsEnabled = false }: Readonly<{ appsEnabled?: boolean }> = {}) =>
  (input: Parameters<typeof fetch>[0]): Response => {
    const url = requestUrl(input);
    if (url.pathname === "/management/v1/apps/definition") return json({ resource_type: appsEnabled ? "team" : null });
    if (url.pathname === "/management/v1/apps") return json(appsEnabled ? gatewayApps : []);
    if (url.pathname === "/management/v1/apps/resources")
      return json([
        ...gatewayApps.map((app) => ({
          resource_type: app.resource_type,
          resource_id: app.resource_id,
          app_id: app.app_id,
          alias: app.alias,
        })),
        { resource_type: "team", resource_id: "unbound-team", app_id: null, alias: "Unconfigured resource" },
      ]);
    if (url.pathname === "/management/v1/customer_organization_budget_groups")
      return json({ default_budgets: [], revision: "a".repeat(64) });
    const app = gatewayApps.find(
      (item) => url.pathname === `/management/v1/apps/${item.app_id}/customer_organizations/org-1`,
    );
    if (app) {
      const appBudget = {
        app_id: app.app_id,
        customer_org_id: "org-1",
        spend: app.app_id === "app-chat" ? 27 : 3,
        budget_override_id: `budget-${app.app_id}`,
        effective_budget: { max_budget: 100, budget_duration: "1mo" },
        budget_usage: [
          {
            source: "scalar",
            budget_duration: "1mo",
            spend: app.app_id === "app-chat" ? 27 : 3,
            max_budget: 100,
            reset_at: null,
          },
        ],
      };
      return json(appBudget);
    }
    const aggregated = url.pathname.endsWith("/daily/activity/aggregated");
    if (url.pathname.endsWith("/daily/activity") || aggregated) {
      const resource = url.pathname.split("/").at(aggregated ? -4 : -3);
      const dimension = activityDimension(resource);
      const ids = activityIds(url, resource, aggregated);
      const selected = records
        .filter((record) => ids.includes(record[dimension]))
        .filter((record) => !url.searchParams.get("app_id") || appForRecord(record) === url.searchParams.get("app_id"))
        .filter(
          (record) =>
            record.date >= (url.searchParams.get("start_date") ?? "") &&
            record.date <= (url.searchParams.get("end_date") ?? ""),
        );
      const results = selected.map((record) => {
        const key = { metrics: record.metrics, metadata: { key_alias: record.alias, team_id: null } };
        const entry = { metrics: record.metrics, metadata: {}, api_key_breakdown: { [record.key]: key } };
        return {
          date: record.date,
          metrics: record.metrics,
          breakdown: {
            model_groups: { [record.model]: entry },
            api_keys: { [record.key]: key },
            entities: { [record[dimension]]: entry },
          },
        };
      });
      return json({
        results,
        metadata: {
          ...EMPTY_DAILY_ACTIVITY_METADATA,
          total_spend: selected.reduce((total, record) => total + record.metrics.spend, 0),
        },
      });
    }
    if (url.pathname === "/management/v1/customer_organizations/org-1") return json(organization);
    if (url.pathname === "/management/v1/customer_teams") return json(list(teams));
    if (url.pathname === "/management/v1/customer_members")
      return json(
        list(members.filter((member) => member.parent_team_ids.includes(url.searchParams.get("parent_team_id") ?? ""))),
      );
    const team = teams.find((team) => url.pathname.endsWith(`/customer_teams/${team.customer_team_id}`));
    if (team) return json(team);
    const member = members.find((member) => url.pathname.endsWith(`/customer_members/${member.customer_member_id}`));
    if (member) return json(member);
    throw new Error(`Unexpected usage request ${url.pathname}`);
  };
const serve = createUsageHandler();
const mockUsageApi = (options: Parameters<typeof createUsageHandler>[0] = {}) => {
  const handler = createUsageHandler(options);
  fetchMock.mockImplementation((input) => Promise.resolve(handler(input)));
};

describe("CustomerHierarchyUsagePage", () => {
  beforeEach(() => {
    vi.useFakeTimers({ toFake: ["Date"] });
    vi.setSystemTime(new Date("2026-11-13T12:00:00Z"));
    testQueryClient.clear();
    registerBaseUrlGetter(() => "http://localhost");
    fetchMock.mockReset();
    mockUsageApi();
    vi.stubGlobal("fetch", fetchMock);
    vi.stubGlobal("requestIdleCallback", (callback: IdleRequestCallback) => {
      callback({ didTimeout: false, timeRemaining: () => 50 });
      return 1;
    });
  });

  afterEach(() => {
    vi.useRealTimers();
    testQueryClient.clear();
    vi.unstubAllGlobals();
  });

  it("should show app spend and budget actions on the customer landing below its global budget", async () => {
    mockUsageApi({ appsEnabled: true });
    renderWithProviders(<CustomerHierarchyUsagePage />, { searchParams: initialUrl });
    expect(await screen.findByRole("button", { name: "Edit budget for chat" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Edit budget for search" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Add app budget" })).toBeInTheDocument();
    const globalBudget = screen.getByRole("region", { name: "Current budget" });
    const appBudget = screen.getByRole("region", { name: "App budgets" });
    expect(globalBudget.compareDocumentPosition(appBudget) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    const chat = within(await within(appBudget).findByRole("row", { name: /^chat Monthly/ }));
    expect(await chat.findByRole("button", { name: "100 total tokens. View breakdown" })).toBeInTheDocument();
    expect(chat.getByText("$27.00 / $100.00")).toBeInTheDocument();
    expect(chat.getByText("$73.00 left")).toBeInTheDocument();
    expect(chat.getByText("27% used")).toBeInTheDocument();
    expect(chat.getByText("$69.00")).toBeInTheDocument();
    expect(chat.getByText("90%")).toBeInTheDocument();
  });

  it("should filter the usage graph by app while retaining the global customer budget and date range", async () => {
    mockUsageApi({ appsEnabled: true });
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn();
    renderWithProviders(<CustomerHierarchyUsagePage />, { searchParams: initialUrl, onUrlUpdate });
    await screen.findByRole("button", { name: "Edit budget for search" });
    await user.click(
      within(screen.getByRole("table", { name: "App budgets" })).getByRole("button", { name: "search", exact: true }),
    );
    await waitFor(() =>
      expect(activityUrl("customer_organizations", "org-1")?.searchParams.get("app_id")).toBe("app-search"),
    );
    await waitFor(() => expect(tableRows("Models")).toHaveLength(1));
    expect(await within(screen.getByRole("table", { name: "App budgets" })).findByText("90%")).toBeInTheDocument();
    expect(tableRows("Models")[0]).toHaveTextContent("model-b");
    expect(
      within(screen.getByRole("region", { name: "Current budget" })).getByText("$40.00 / $100.00"),
    ).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Customer teams", exact: true })).toHaveAttribute(
      "href",
      "/ui/customers/manage?org=org-1&from=2026-11-01&to=2026-11-03&app_id=app-search",
    );
    expect(onUrlUpdate).toHaveBeenLastCalledWith(
      expect.objectContaining({
        queryString: "?org=org-1&from=2026-11-01&to=2026-11-03&source=portal&app_id=app-search",
      }),
    );
    await chooseSelectOption(user, screen.getByRole("combobox", { name: "App" }), "All apps");
    await waitFor(() => expect(tableRows("Models")).toHaveLength(2));
  });

  it("should retain the global budget while an app filter error can be retried", async () => {
    const appHandler = createUsageHandler({ appsEnabled: true });
    fetchMock.mockImplementation((input) =>
      Promise.resolve(
        requestUrl(input).pathname === "/management/v1/apps"
          ? json({ detail: "App list unavailable" }, 503)
          : appHandler(input),
      ),
    );
    const user = userEvent.setup();
    renderWithProviders(<CustomerHierarchyUsagePage />, { searchParams: initialUrl });
    const chart = within(await screen.findByRole("region", { name: "Usage over time" }));
    expect(await chart.findByRole("alert")).toHaveTextContent("App list unavailable");
    expect(
      within(screen.getByRole("region", { name: "Current budget" })).getByText("$40.00 / $100.00"),
    ).toBeInTheDocument();
    mockUsageApi({ appsEnabled: true });
    await user.click(chart.getByRole("button", { name: "Retry" }));
    await waitFor(() => expect(chart.queryByRole("alert")).not.toBeInTheDocument());
    await chooseSelectOption(user, chart.getByRole("combobox", { name: "App" }), "search");
    await waitFor(() =>
      expect(activityUrl("customer_organizations", "org-1")?.searchParams.get("app_id")).toBe("app-search"),
    );
  });

  it.each([
    ["2026-11-15T00:30:00Z", "", "2026-11-01", "2026-11-15"],
    ["2027-01-01T00:30:00Z", "", "2027-01-01", "2027-01-01"],
    ["2026-11-15T00:30:00Z", "&range=month", "2026-11-01", "2026-11-30"],
    ["2026-11-15T00:30:00Z", "&range=MTD", "2026-11-01", "2026-11-15"],
    ["2026-11-30T23:30:00Z", "&range=today", "2026-11-30", "2026-11-30"],
  ])("should use UTC calendar dates at %s with %s", async (now, query, startDate, endDate) => {
    vi.setSystemTime(new Date(now));
    renderWithProviders(<CustomerHierarchyUsagePage />, { searchParams: `?org=org-1${query}` });
    await screen.findByRole("heading", { name: "Customer organization" });
    expect(
      requestedUrls()
        .find((url) => url.pathname.endsWith("/daily/activity"))
        ?.searchParams.get("start_date"),
    ).toBe(startDate);
    expect(
      requestedUrls()
        .find((url) => url.pathname.endsWith("/daily/activity"))
        ?.searchParams.get("end_date"),
    ).toBe(endDate);
  });

  it("should keep the selected dates in customer navigation and the return to all customers", async () => {
    renderWithProviders(<CustomerHierarchyUsagePage />, { searchParams: initialUrl });
    await screen.findByRole("heading", { name: "Customer organization" });
    for (const name of ["Customer teams", "Customer members", "Settings", "All customers"]) {
      const link = screen.getByRole("link", { name, exact: true });
      const url = new URL(link.getAttribute("href")!, "http://localhost");
      expect(url.searchParams.get("from")).toBe("2026-11-01");
      expect(url.searchParams.get("to")).toBe("2026-11-03");
    }
  });

  it("should render organization totals and model rankings from the selected UTC date range", async () => {
    renderWithProviders(<CustomerHierarchyUsagePage />, { searchParams: initialUrl });

    expect(await screen.findByRole("heading", { name: "Customer organization" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Customer teams", exact: true })).toHaveAttribute(
      "href",
      "/ui/customers/manage?org=org-1&from=2026-11-01&to=2026-11-03",
    );
    expect(screen.getByText("$15.00")).toBeInTheDocument();
    const budget = screen.getByRole("region", { name: "Current budget" });
    expect(within(budget).getByText("$40.00 / $100.00")).toBeInTheDocument();
    expect(within(budget).getByText("Recorded budget spend")).toBeInTheDocument();
    expect(within(budget).getByRole("button", { name: "Edit budget" })).toBeEnabled();
    expect(screen.getByTitle("700 tokens")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Usage over time" })).toBeInTheDocument();
    expect(tableRows("Models")[0]).toHaveTextContent("model-a");
    expect(within(tableRows("Models")[0]).getByText("80%")).toBeInTheDocument();
    expect(tableRows("Models")[1]).toHaveTextContent("model-b");
    expect(activityUrl("customer_organizations", "org-1")?.pathname).toBe(
      "/management/v1/customer_organizations/daily/activity",
    );
    expect(activityUrl("customer_organizations", "org-1")?.searchParams.get("start_date")).toBe("2026-11-01");
    expect(activityUrl("customer_organizations", "org-1")?.searchParams.get("end_date")).toBe("2026-11-03");
  });

  it("shows the customer-wide token breakdown in the Usage over time chart", async () => {
    const user = userEvent.setup();
    renderWithProviders(<CustomerHierarchyUsagePage />, { searchParams: initialUrl });
    await screen.findByRole("table", { name: "Models" });
    const trend = within(screen.getByRole("region", { name: "Usage over time" }));
    await user.click(trend.getByRole("radio", { name: "Token breakdown" }));
    expect(trend.getByRole("radio", { name: "Token breakdown" })).toBeChecked();
    expect(trend.getByText("Cache read tokens")).toBeInTheDocument();
    expect(trend.getByText("Cache write tokens")).toBeInTheDocument();
    expect(trend.getByText("Uncached input tokens")).toBeInTheDocument();
    expect(trend.getByText("Output tokens")).toBeInTheDocument();
    expect(trend.getByText("700")).toBeInTheDocument();
    expect(trend.getByText("$0.0123")).toBeInTheDocument();
    expect(trend.getByText("Recorded cache cost")).toBeInTheDocument();
    expect(trend.queryByText("Partial cost")).not.toBeInTheDocument();
    expect(trend.queryByText("model-a")).not.toBeInTheDocument();
    await user.click(trend.getByRole("radio", { name: "Tokens" }));
    expect(trend.getByText("model-a")).toBeInTheDocument();
  });

  it("should apply the global metric to sorting and shares across models, keys and teams", async () => {
    const user = userEvent.setup();
    renderWithProviders(<CustomerHierarchyUsagePage />, { searchParams: initialUrl });
    await screen.findByRole("table", { name: "Models" });

    await user.click(screen.getByRole("radio", { name: "Tokens" }));
    expect(tableRows("Models")[0]).toHaveTextContent("model-b");
    expect(within(tableRows("Models")[0]).getByText("85.7%")).toBeInTheDocument();

    for (const [tab, leader] of [
      ["API keys", "Search key"],
      ["Customer teams", "Operations"],
    ]) {
      await selectBreakdown(user, tab);
      expect(tableRows(tab)[0]).toHaveTextContent(leader);
      expect(within(tableRows(tab)[0]).getByText("85.7%")).toBeInTheDocument();
      expect(within(screen.getByRole("region", { name: "Usage over time" })).getByText(leader)).toBeInTheDocument();
    }

    await user.click(screen.getByRole("radio", { name: "Requests" }));
    expect(tableRows("Customer teams")[0]).toHaveTextContent("Operations");
    expect(within(tableRows("Customer teams")[0]).getByText("71.4%")).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "Requests" })).toHaveAttribute("aria-sort", "descending");
  });

  it("should drill into customer teams and members and retain dates and unrelated URL state", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn();
    renderWithProviders(<CustomerHierarchyUsagePage />, { searchParams: initialUrl, onUrlUpdate });
    await screen.findByRole("heading", { name: "Customer organization" });

    await selectBreakdown(user, "Customer teams");
    await user.click(
      within(screen.getByRole("table", { name: "Customer teams" })).getByRole("button", { name: "Product" }),
    );
    expect(await screen.findByRole("heading", { name: "Product" })).toBeInTheDocument();
    expect(
      within(screen.getByRole("table", { name: "Customer member spend" })).getByText("Member standard"),
    ).toBeInTheDocument();
    expect(activityUrl("customer_teams", "team-a")?.pathname).toBe("/management/v1/customer_teams/daily/activity");
    expect(activityUrl("customer_teams", "team-a")?.searchParams.get("customer_team_ids")).toBe("team-a");
    const teamBudget = screen.getByRole("region", { name: "Current budget" });
    expect(within(teamBudget).getByText("No budget set")).toBeInTheDocument();
    expect(within(teamBudget).queryByText(/\$40\.00/)).not.toBeInTheDocument();
    expect(within(teamBudget).getByRole("link", { name: "Set budget" })).toHaveAttribute(
      "href",
      "/ui/customers/manage?org=org-1&team=team-a&from=2026-11-01&to=2026-11-03&tab=settings",
    );

    await user.click(
      within(screen.getByRole("table", { name: "Customer member spend" })).getByRole("button", { name: "Member One" }),
    );
    expect(await screen.findByRole("heading", { name: "Member One" })).toBeInTheDocument();
    expect(
      within(screen.getByRole("table", { name: "Customer member spend" })).getByText("Member standard"),
    ).toBeInTheDocument();
    const memberBudget = screen.getByRole("region", { name: "Current budget" });
    expect(within(memberBudget).getByText("$40.00 / $10.00")).toBeInTheDocument();
    expect(within(memberBudget).getByRole("button", { name: "Edit budget" })).toBeEnabled();
    const memberUrl = activityUrl("customer_members", "member-a")!;
    expect(memberUrl.pathname).toBe("/management/v1/customer_members/daily/activity");
    expect(memberUrl.searchParams.get("customer_member_ids")).toBe("member-a");
    expect(memberUrl.searchParams.has("customer_org_ids")).toBe(false);
    expect(memberUrl.searchParams.has("customer_team_ids")).toBe(false);
    expect(memberUrl.searchParams.get("start_date")).toBe("2026-11-01");
    expect(memberUrl.searchParams.get("end_date")).toBe("2026-11-03");
    expect(onUrlUpdate).toHaveBeenLastCalledWith(
      expect.objectContaining({
        queryString: "?org=org-1&from=2026-11-01&to=2026-11-03&source=portal&team=team-a&member=member-a",
      }),
    );

    await user.click(
      within(screen.getByRole("navigation", { name: "Customer hierarchy" })).getByRole("link", {
        name: "Customer organization",
        exact: true,
      }),
    );
    expect(await screen.findByRole("heading", { name: "Customer organization" })).toBeInTheDocument();
    expect(onUrlUpdate).toHaveBeenLastCalledWith(expect.objectContaining({ queryString: initialUrl }));
  });

  it("shows the complete breadcrumb for a member deep link and returns to each parent with the same filters", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn();
    renderWithProviders(<CustomerHierarchyUsagePage />, {
      searchParams: `${initialUrl}&member=member-a`,
      onUrlUpdate,
    });
    await screen.findByRole("heading", { name: "Member One" });
    const path = within(screen.getByRole("navigation", { name: "Customer hierarchy" }));
    expect(path.getAllByRole("link").map((link) => link.textContent)).toEqual([
      "All customers",
      "Customer organization",
      "Product",
      "Member One",
    ]);
    expect(path.getByRole("link", { name: "Member One", exact: true })).toHaveAttribute("aria-current", "page");
    expect(path.getByRole("link", { name: "Product", exact: true })).toHaveAttribute(
      "href",
      "/ui/customers/usage?org=org-1&team=team-a&from=2026-11-01&to=2026-11-03",
    );
    await user.click(path.getByRole("link", { name: "Product", exact: true }));
    expect(await screen.findByRole("heading", { name: "Product" })).toBeInTheDocument();
    expect(path.getAllByRole("link").map((link) => link.textContent)).toEqual([
      "All customers",
      "Customer organization",
      "Product",
    ]);
    expect(path.getByRole("link", { name: "Product", exact: true })).toHaveAttribute("aria-current", "page");
    await user.click(path.getByRole("link", { name: "Customer organization", exact: true }));
    expect(await screen.findByRole("heading", { name: "Customer organization" })).toBeInTheDocument();
    expect(path.getByRole("link", { name: "Customer organization", exact: true })).toHaveAttribute(
      "aria-current",
      "page",
    );
    await waitFor(() =>
      expect(onUrlUpdate).toHaveBeenLastCalledWith(expect.objectContaining({ queryString: initialUrl })),
    );
  });

  it("should fall back to Models for a removed Members breakdown and offer only chart breakdowns", async () => {
    const user = userEvent.setup();
    renderWithProviders(<CustomerHierarchyUsagePage />, { searchParams: `${initialUrl}&breakdown=members` });

    expect(await screen.findByRole("table", { name: "Models" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Top models" })).toBeInTheDocument();
    expect(screen.getByRole("table", { name: "Customer member spend" })).toBeInTheDocument();
    await user.click(screen.getByRole("combobox", { name: "Break down by" }));
    expect((await screen.findAllByRole("option")).map((option) => option.textContent)).toEqual([
      "Models",
      "API keys",
      "Customer teams",
    ]);
  });

  it("should retain monthly member usage and whole-customer shares across chart breakdowns", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn();
    renderWithProviders(<CustomerHierarchyUsagePage />, { searchParams: initialUrl, onUrlUpdate });
    const table = await screen.findByRole("table", { name: "Customer member spend" });
    const firstMember = within(table).getAllByRole("row")[1];
    expect(firstMember).toHaveTextContent("Member One");
    expect(within(firstMember).getByText("Member standard")).toBeInTheDocument();
    expect(within(firstMember).getByText("66.7%")).toBeInTheDocument();
    expect(within(firstMember).getByRole("button", { name: "100 total tokens. View breakdown" })).toBeInTheDocument();
    expect(within(firstMember).getByRole("cell", { name: "$20.00 No monthly limit", exact: true })).toBeInTheDocument();
    expect(
      within(firstMember).getByRole("cell", { name: "$10.00 total No automatic reset", exact: true }),
    ).toBeInTheDocument();
    expect(within(firstMember).getByRole("meter", { name: "Share of customer spend" })).toHaveAttribute(
      "aria-valuetext",
      "66.7%",
    );
    expect(within(firstMember).getByText("Reached limit")).toBeInTheDocument();

    await selectBreakdown(user, "Customer teams");
    await user.click(screen.getByRole("radio", { name: "Tokens" }));

    expect(tableRows("Customer teams")[0]).toHaveTextContent("Operations");
    expect(within(table).getAllByRole("row")[1]).toHaveTextContent("Member One");
    const requestCount = fetchMock.mock.calls.length;
    act(() =>
      fireEvent.change(screen.getByRole("textbox", { name: "Search customer members" }), {
        target: { value: "Member One" },
      }),
    );
    expect(within(table).queryByText("Member Two")).not.toBeInTheDocument();
    expect(within(table).getByText("Member One")).toBeInTheDocument();
    await waitFor(() =>
      expect(new URLSearchParams(onUrlUpdate.mock.lastCall?.[0].queryString).get("memberSearch")).toBe("Member One"),
    );
    const updatedUrl = new URLSearchParams(onUrlUpdate.mock.lastCall?.[0].queryString);
    expect(updatedUrl.get("org")).toBe("org-1");
    expect(updatedUrl.get("breakdown")).toBe("teams");
    expect(updatedUrl.get("metric")).toBe("tokens");
    expect(updatedUrl.get("source")).toBe("portal");
    expect(fetchMock.mock.calls).toHaveLength(requestCount);
  });

  it("should clear the previous scope values while customer team usage loads", async () => {
    const user = userEvent.setup();
    renderWithProviders(<CustomerHierarchyUsagePage />, { searchParams: initialUrl });
    await screen.findByRole("heading", { name: "Customer organization" });
    await selectBreakdown(user, "Customer teams");
    fetchMock.mockImplementation(() => new Promise<Response>(() => {}));

    await user.click(
      within(screen.getByRole("table", { name: "Customer teams" })).getByRole("button", { name: "Product" }),
    );

    expect(screen.getByRole("status", { name: "Loading customer usage" })).toBeInTheDocument();
    expect(screen.queryByText("$15.00")).not.toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Customer organization" })).not.toBeInTheDocument();
    expect(screen.queryByRole("table")).not.toBeInTheDocument();
    await waitFor(() =>
      expect(requestedUrls().some((url) => url.pathname === "/management/v1/customer_teams/team-a")).toBe(true),
    );
  });

  it.each([
    { monthlySpend: 8, yearlySpend: 62, status: "Within budget" },
    { monthlySpend: 10, yearlySpend: 62, status: "Reached limit" },
    { monthlySpend: 8, yearlySpend: 100, status: "Reached limit" },
  ])(
    "should use each recorded budget window consistently across the table and summary (%#)",
    async ({ monthlySpend, yearlySpend, status }) => {
      const user = userEvent.setup();
      const budgetUsage = [
        { source: "scalar", budget_duration: "1mo", max_budget: 50, spend: 40, reset_at: "2026-12-01T00:00:00Z" },
        {
          source: "window",
          budget_duration: "1mo",
          max_budget: 10,
          spend: monthlySpend,
          window_start: "2026-11-01T17:00:00Z",
          reset_at: "2026-12-01T17:00:00Z",
        },
        {
          source: "window",
          budget_duration: "12mo",
          max_budget: 100,
          spend: yearlySpend,
          window_start: "2026-01-01T17:00:00Z",
          reset_at: "2027-01-01T17:00:00Z",
        },
      ];
      fetchMock.mockImplementation((input) => {
        const url = requestUrl(input);
        const member = {
          ...members[0],
          budget_usage: budgetUsage,
          effective_budget: {
            max_budget: 50,
            budget_duration: "1mo",
            budget_limits: [
              { budget_duration: "1mo", max_budget: 10 },
              { budget_duration: "12mo", max_budget: 100 },
            ],
          },
        };
        if (url.pathname === "/management/v1/customer_members")
          return Promise.resolve(json(list(url.searchParams.get("parent_team_id") === "team-a" ? [member] : [])));
        if (url.pathname.endsWith("/customer_members/member-a")) return Promise.resolve(json(member));
        return Promise.resolve(serve(input));
      });
      renderWithProviders(<CustomerHierarchyUsagePage />, { searchParams: initialUrl });
      const table = await screen.findByRole("table", { name: "Customer member spend" });
      const row = within(table).getAllByRole("row")[1];
      expect(within(row).getByText("$40.00 / $50.00")).toBeInTheDocument();
      expect(within(row).getByText(`$${monthlySpend.toFixed(2)} / $10.00`)).toBeInTheDocument();
      expect(within(row).getByText(`$${yearlySpend.toFixed(2)}`)).toBeInTheDocument();
      expect(within(row).getByText("$100.00 limit")).toBeInTheDocument();
      expect(
        within(row)
          .getAllByRole("progressbar")
          .map((bar) => bar.getAttribute("aria-valuenow")),
      ).toEqual(["80", String(monthlySpend * 10)]);
      expect(within(row).getByRole("meter", { name: "Share of customer spend" })).toHaveAttribute(
        "aria-valuetext",
        "66.7%",
      );
      expect(within(row).getByRole("cell", { name: status, exact: true })).toBeInTheDocument();
      await user.click(within(row).getByRole("button", { name: "Member One" }));
      expect(await screen.findByRole("heading", { name: "Member One" })).toBeInTheDocument();
      const summary = screen.getByRole("region", { name: "Current budget" });
      expect(within(summary).getByText("$40.00 / $50.00")).toBeInTheDocument();
      expect(within(summary).getByText(`$${monthlySpend.toFixed(2)} / $10.00`)).toBeInTheDocument();
      expect(within(summary).getByText(`$${yearlySpend.toFixed(2)} / $100.00`)).toBeInTheDocument();
      expect(
        within(summary)
          .getAllByRole("progressbar")
          .map((bar) => bar.getAttribute("aria-valuenow")),
      ).toEqual(["80", String(monthlySpend * 10)]);
    },
  );

  it("should retain limits but not invent window spend when an older backend omits usage", async () => {
    fetchMock.mockImplementation((input) => {
      const url = requestUrl(input);
      if (url.pathname === "/management/v1/customer_members")
        return Promise.resolve(
          json(
            list(
              url.searchParams.get("parent_team_id") === "team-a"
                ? [{ ...members[0], effective_budget: { budget_limits: [{ budget_duration: "1mo", max_budget: 10 }] } }]
                : [],
            ),
          ),
        );
      return Promise.resolve(serve(input));
    });
    renderWithProviders(<CustomerHierarchyUsagePage />, { searchParams: initialUrl });
    const table = await screen.findByRole("table", { name: "Customer member spend" });
    const row = within(table).getAllByRole("row")[1];
    expect(within(row).getByText("Unavailable / $10.00")).toBeInTheDocument();
    expect(within(row).getByText("Budget usage unavailable")).toBeInTheDocument();
    expect(within(row).queryByRole("progressbar")).not.toBeInTheDocument();
  });

  it("should request a changed date range without showing stale totals or losing the customer scope", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn();
    renderWithProviders(<CustomerHierarchyUsagePage />, { searchParams: initialUrl, onUrlUpdate });
    await screen.findByRole("heading", { name: "Customer organization" });
    fetchMock.mockImplementation((input) =>
      requestUrl(input).pathname.endsWith("/daily/activity") &&
      requestUrl(input).searchParams.get("start_date") === "2026-10-01"
        ? new Promise<Response>(() => {})
        : Promise.resolve(serve(input)),
    );

    await user.click(screen.getByRole("button", { name: / - / }));
    act(() => {
      fireEvent.change(screen.getByLabelText("Start date"), { target: { value: "2026-10-01" } });
      fireEvent.change(screen.getByLabelText("End date"), { target: { value: "2026-10-07" } });
    });
    await user.click(screen.getByRole("button", { name: "Apply" }));

    await waitFor(() =>
      expect(activityUrl("customer_organizations", "org-1", "2026-10-07")?.searchParams.get("start_date")).toBe(
        "2026-10-01",
      ),
    );
    expect(activityUrl("customer_organizations", "org-1", "2026-10-07")?.searchParams.get("end_date")).toBe(
      "2026-10-07",
    );
    expect(activityUrl("customer_organizations", "org-1", "2026-10-07")?.pathname).toBe(
      "/management/v1/customer_organizations/daily/activity",
    );
    expect(screen.getByRole("status", { name: "Loading customer usage" })).toBeInTheDocument();
    expect(screen.queryByText("$15.00")).not.toBeInTheDocument();
    expect(onUrlUpdate).toHaveBeenLastCalledWith(
      expect.objectContaining({ queryString: "?org=org-1&from=2026-10-01&to=2026-10-07&source=portal" }),
    );
  });

  it("should show a failed request as an error and load usage when retried", async () => {
    const user = userEvent.setup();
    fetchMock.mockResolvedValueOnce(json({ error: "Usage temporarily unavailable" }, 503));
    renderWithProviders(<CustomerHierarchyUsagePage />, { searchParams: initialUrl });

    expect(await screen.findByRole("alert")).toHaveTextContent("Could not load customer usage");
    expect(screen.getByRole("alert")).toHaveTextContent("Usage temporarily unavailable");
    expect(screen.queryByText("$0.00")).not.toBeInTheDocument();
    expect(screen.queryByText("No usage in this range")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Retry" }));

    expect(await screen.findByText("$15.00")).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("should show an empty date range after the scoped request succeeds without activity", async () => {
    fetchMock.mockImplementation((input) =>
      Promise.resolve(
        requestUrl(input).pathname.endsWith("/daily/activity")
          ? json({ results: [], metadata: EMPTY_DAILY_ACTIVITY_METADATA })
          : serve(input),
      ),
    );
    renderWithProviders(<CustomerHierarchyUsagePage />, { searchParams: initialUrl });

    expect(await screen.findByText("No usage in this range")).toBeInTheDocument();
    expect(screen.getByTitle("0 tokens")).toBeInTheDocument();
    expect(screen.queryByRole("status", { name: "Loading customer usage" })).not.toBeInTheDocument();
    expect(screen.getByText("No usage recorded for this breakdown.")).toBeInTheDocument();
  });

  it("should request an organization selection without issuing an unscoped usage request", () => {
    renderWithProviders(<CustomerHierarchyUsagePage />, { searchParams: "?source=portal" });

    expect(screen.getByText("Open a customer organization to explore its usage.")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Back to customer organizations" })).toHaveAttribute(
      "href",
      "/ui/customers",
    );
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("should show a zero effective member budget and its UTC reset without substituting the organization budget", async () => {
    fetchMock.mockImplementation((input) =>
      Promise.resolve(
        requestUrl(input).pathname.endsWith("/customer_members/member-a")
          ? json({ ...members[0], effective_budget: { max_budget: 0 }, budget_reset_at: "2026-12-01T00:00:00Z" })
          : serve(input),
      ),
    );
    renderWithProviders(<CustomerHierarchyUsagePage />, {
      searchParams: `${initialUrl}&team=team-a&member=member-a`,
    });
    const budget = await screen.findByRole("region", { name: "Current budget" });
    expect(within(budget).getByText("$40.00 / $0.00")).toBeInTheDocument();
    expect(within(budget).getByRole("progressbar")).toHaveAttribute("aria-valuenow", "100");
    expect(within(budget).getByTitle(/Configured reset:.*UTC/)).toBeInTheDocument();
    expect(within(budget).queryByRole("link", { name: "Set budget" })).not.toBeInTheDocument();
  });
});
