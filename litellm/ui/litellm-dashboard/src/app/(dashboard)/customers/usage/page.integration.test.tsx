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
  default_budgets: [],
};
const teams = [
  { ...audit, customer_team_id: "team-a", parent_org_id: "org-1", alias: "Product", budget: null },
  { ...audit, customer_team_id: "team-b", parent_org_id: "org-1", alias: "Operations", budget: null },
];
const members = [
  {
    ...audit,
    customer_member_id: "member-a",
    parent_team_id: "team-a",
    alias: "Member One",
    effective_budget: { max_budget: 10 },
  },
  {
    ...audit,
    customer_member_id: "member-b",
    parent_team_id: "team-b",
    alias: "Member Two",
    effective_budget: { max_budget: 20 },
  },
];
const records = [
  {
    organization: "org-1",
    team: "team-a",
    member: "member-a",
    model: "model-a",
    key: "hash-a",
    alias: "Assistant key",
    metrics: metrics(12, 100, 2),
    date: "2026-11-01",
  },
  {
    organization: "org-1",
    team: "team-b",
    member: "member-b",
    model: "model-b",
    key: "hash-b",
    alias: "Search key",
    metrics: metrics(3, 600, 5),
    date: "2026-11-02",
  },
];
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
const activityUrl = (resource: string, id: string) => {
  const filter = filterFor(resource);
  return requestedUrls().findLast(
    (url) => url.pathname === `/management/v1/${resource}/daily/activity` && url.searchParams.get(filter) === id,
  );
};
const serve = (input: Parameters<typeof fetch>[0]): Response => {
  const url = requestUrl(input);
  if (url.pathname.endsWith("/daily/activity")) {
    const resource = url.pathname.split("/").at(-3);
    const dimension = activityDimension(resource);
    const filter = filterFor(resource);
    const ids = url.searchParams.get(filter)?.split(",") ?? [];
    const selected = records.filter((record) => ids.includes(record[dimension]));
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
    return json({ results, metadata: EMPTY_DAILY_ACTIVITY_METADATA });
  }
  if (url.pathname === "/management/v1/customer_organizations/org-1") return json(organization);
  if (url.pathname === "/management/v1/customer_teams") return json(list(teams));
  if (url.pathname === "/management/v1/customer_members")
    return json(list(members.filter((member) => member.parent_team_id === url.searchParams.get("parent_team_id"))));
  const team = teams.find((team) => url.pathname.endsWith(`/customer_teams/${team.customer_team_id}`));
  if (team) return json(team);
  const member = members.find((member) => url.pathname.endsWith(`/customer_members/${member.customer_member_id}`));
  if (member) return json(member);
  throw new Error(`Unexpected usage request ${url.pathname}`);
};

describe("CustomerHierarchyUsagePage", () => {
  beforeEach(() => {
    testQueryClient.clear();
    registerBaseUrlGetter(() => "http://localhost");
    fetchMock.mockReset();
    fetchMock.mockImplementation((input) => Promise.resolve(serve(input)));
    vi.stubGlobal("fetch", fetchMock);
    vi.stubGlobal("requestIdleCallback", (callback: IdleRequestCallback) => {
      callback({ didTimeout: false, timeRemaining: () => 50 });
      return 1;
    });
  });

  afterEach(() => {
    testQueryClient.clear();
    vi.unstubAllGlobals();
  });

  it("should render organization totals and model rankings from the selected UTC date range", async () => {
    renderWithProviders(<CustomerHierarchyUsagePage />, { searchParams: initialUrl });

    expect(await screen.findByRole("heading", { name: "Customer organization" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Teams", exact: true })).toHaveAttribute(
      "href",
      "/ui/customers/manage?org=org-1",
    );
    expect(screen.getByText("$15.00")).toBeInTheDocument();
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

  it("should apply the global metric to sorting and shares across models, keys and teams", async () => {
    const user = userEvent.setup();
    renderWithProviders(<CustomerHierarchyUsagePage />, { searchParams: initialUrl });
    await screen.findByRole("table", { name: "Models" });

    await user.click(screen.getByRole("radio", { name: "Tokens" }));
    expect(tableRows("Models")[0]).toHaveTextContent("model-b");
    expect(within(tableRows("Models")[0]).getByText("85.7%")).toBeInTheDocument();

    for (const [tab, leader] of [
      ["API keys", "Search key"],
      ["Teams", "Operations"],
    ]) {
      await selectBreakdown(user, tab);
      expect(tableRows(tab)[0]).toHaveTextContent(leader);
      expect(within(tableRows(tab)[0]).getByText("85.7%")).toBeInTheDocument();
      expect(within(screen.getByRole("region", { name: "Usage over time" })).getByText(leader)).toBeInTheDocument();
    }

    await user.click(screen.getByRole("radio", { name: "Requests" }));
    expect(tableRows("Teams")[0]).toHaveTextContent("Operations");
    expect(within(tableRows("Teams")[0]).getByText("71.4%")).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "Requests" })).toHaveAttribute("aria-sort", "descending");
  });

  it("should drill into customer teams and members and retain dates and unrelated URL state", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn();
    renderWithProviders(<CustomerHierarchyUsagePage />, { searchParams: initialUrl, onUrlUpdate });
    await screen.findByRole("heading", { name: "Customer organization" });

    await selectBreakdown(user, "Teams");
    await user.click(within(screen.getByRole("table", { name: "Teams" })).getByRole("button", { name: "Product" }));
    expect(await screen.findByRole("heading", { name: "Product" })).toBeInTheDocument();
    expect(activityUrl("customer_teams", "team-a")?.pathname).toBe("/management/v1/customer_teams/daily/activity");
    expect(activityUrl("customer_teams", "team-a")?.searchParams.get("customer_team_ids")).toBe("team-a");

    await user.click(
      within(screen.getByRole("table", { name: "Member spend" })).getByRole("button", { name: "Member One" }),
    );
    expect(await screen.findByRole("heading", { name: "Member One" })).toBeInTheDocument();
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

    await user.click(screen.getByRole("button", { name: "Customer organization" }));
    expect(await screen.findByRole("heading", { name: "Customer organization" })).toBeInTheDocument();
    expect(onUrlUpdate).toHaveBeenLastCalledWith(expect.objectContaining({ queryString: initialUrl }));
  });

  it("should fall back to Models for a removed Members breakdown and offer only chart breakdowns", async () => {
    const user = userEvent.setup();
    renderWithProviders(<CustomerHierarchyUsagePage />, { searchParams: `${initialUrl}&breakdown=members` });

    expect(await screen.findByRole("table", { name: "Models" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Top models" })).toBeInTheDocument();
    expect(screen.getByRole("table", { name: "Member spend" })).toBeInTheDocument();
    await user.click(screen.getByRole("combobox", { name: "Break down by" }));
    expect((await screen.findAllByRole("option")).map((option) => option.textContent)).toEqual([
      "Models",
      "API keys",
      "Teams",
    ]);
  });

  it("should retain a separate member spend table with range usage and current budget status across chart breakdowns", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn();
    renderWithProviders(<CustomerHierarchyUsagePage />, { searchParams: initialUrl, onUrlUpdate });
    const table = await screen.findByRole("table", { name: "Member spend" });
    const firstMember = within(table).getAllByRole("row")[1];
    expect(firstMember).toHaveTextContent("Member One");
    expect(within(firstMember).getByText("$12.00")).toBeInTheDocument();
    expect(within(firstMember).getByText("$40.00 / $10.00")).toBeInTheDocument();
    expect(within(firstMember).getByText("Over budget")).toBeInTheDocument();

    await selectBreakdown(user, "Teams");
    await user.click(screen.getByRole("radio", { name: "Tokens" }));

    expect(tableRows("Teams")[0]).toHaveTextContent("Operations");
    expect(within(table).getAllByRole("row")[1]).toHaveTextContent("Member One");
    const requestCount = fetchMock.mock.calls.length;
    act(() =>
      fireEvent.change(screen.getByRole("textbox", { name: "Search members" }), { target: { value: "Member One" } }),
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
    await selectBreakdown(user, "Teams");
    fetchMock.mockImplementation(() => new Promise<Response>(() => {}));

    await user.click(within(screen.getByRole("table", { name: "Teams" })).getByRole("button", { name: "Product" }));

    expect(screen.getByRole("status", { name: "Loading customer usage" })).toBeInTheDocument();
    expect(screen.queryByText("$15.00")).not.toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Customer organization" })).not.toBeInTheDocument();
    expect(screen.queryByRole("table")).not.toBeInTheDocument();
    await waitFor(() =>
      expect(requestedUrls().some((url) => url.pathname === "/management/v1/customer_teams/team-a")).toBe(true),
    );
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
      expect(activityUrl("customer_organizations", "org-1")?.searchParams.get("start_date")).toBe("2026-10-01"),
    );
    expect(activityUrl("customer_organizations", "org-1")?.searchParams.get("end_date")).toBe("2026-10-07");
    expect(activityUrl("customer_organizations", "org-1")?.pathname).toBe(
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
    expect(screen.getByRole("link", { name: "Back to customers" })).toHaveAttribute("href", "/ui/customers");
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
