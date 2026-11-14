import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { registerBaseUrlGetter } from "@/lib/http/runtime";
import { EMPTY_DAILY_ACTIVITY_METADATA } from "@/components/UsagePage/dailyActivityApi";
import type { DailyData, SpendMetrics } from "@/components/UsagePage/types";
import { customerHierarchyUsageCall } from "./customerUsageApi";

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
  budget_reset_at: "2026-12-01T00:00:00Z",
  default_budgets: [],
};
const teams = [
  { ...audit, customer_team_id: "team-a", parent_org_id: "org-1", alias: "Product", budget: { max_budget: 50 } },
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
const metrics = (spend: number, tokens: number, requests: number): SpendMetrics => ({
  spend,
  prompt_tokens: tokens,
  completion_tokens: 0,
  total_tokens: tokens,
  api_requests: requests,
  successful_requests: requests,
  failed_requests: 0,
  cache_read_input_tokens: 0,
  cache_creation_input_tokens: 0,
});
const records = [
  {
    organization: "org-1",
    team: "team-a",
    member: "member-a",
    model: "model-a",
    key: "hash-a",
    alias: "Assistant key",
    metrics: metrics(3, 100, 8),
  },
  {
    organization: "org-1",
    team: "team-b",
    member: "member-b",
    model: "model-b",
    key: "hash-b",
    alias: "Search key",
    metrics: metrics(1, 500, 2),
  },
];
type UsageRecord = (typeof records)[number];
const daily = (record: UsageRecord, entityId: string): DailyData => {
  const key = { metrics: record.metrics, metadata: { key_alias: record.alias, team_id: null } };
  const entity = { metrics: record.metrics, metadata: {}, api_key_breakdown: { [record.key]: key } };
  return {
    date: "2026-11-01",
    metrics: record.metrics,
    breakdown: {
      models: { [record.model]: entity },
      model_groups: { [record.model]: entity },
      providers: {},
      mcp_servers: {},
      api_keys: { [record.key]: key },
      entities: { [entityId]: entity },
    },
  };
};
const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
const requestUrl = (input: Parameters<typeof fetch>[0]) =>
  new URL(input instanceof Request ? input.url : String(input), "http://localhost");
const fetchMock = vi.fn<typeof fetch>();
const urls = () => fetchMock.mock.calls.map(([input]) => requestUrl(input));
const dailyUrls = () => urls().filter((url) => url.pathname.endsWith("/daily/activity"));
const request = {
  scope: { org: "org-1", team: null, member: null },
  accessToken: "test-session",
  startDate: "2026-11-01",
  endDate: "2026-11-03",
};
const list = <T>(data: T[], page = 1, totalPages = 1) => ({
  data,
  meta: { total_count: data.length, page, page_size: 100, total_pages: totalPages },
});

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
const activityResponse = (url: URL) => {
  const collection = url.pathname.split("/").at(-3);
  const dimension = activityDimension(collection);
  const filter = filterFor(collection);
  const ids = url.searchParams.get(filter)?.split(",") ?? [];
  const selected = records.filter((row) => ids.includes(row[dimension]));
  const page = Number(url.searchParams.get("page"));
  const current = selected.slice(page - 1, page);
  const values = current[0]?.metrics ?? metrics(0, 0, 0);
  return json({
    results: current.map((row) => daily(row, row[dimension])),
    metadata: {
      ...EMPTY_DAILY_ACTIVITY_METADATA,
      total_spend: values.spend,
      total_tokens: values.total_tokens,
      total_api_requests: values.api_requests,
      page,
      total_pages: selected.length,
      has_more: page < selected.length,
    },
  });
};

const serve = (input: Parameters<typeof fetch>[0]): Response => {
  const url = requestUrl(input);
  if (url.pathname.endsWith("/daily/activity")) return activityResponse(url);
  if (url.pathname === "/management/v1/customer_organizations/org-1") return json(organization);
  if (url.pathname === "/management/v1/customer_teams") {
    const page = Number(url.searchParams.get("page"));
    return json(list([teams[page - 1]], page, 2));
  }
  if (url.pathname === "/management/v1/customer_members")
    return json(list(members.filter((member) => member.parent_team_id === url.searchParams.get("parent_team_id"))));
  const team = teams.find((team) => url.pathname.endsWith(`/customer_teams/${team.customer_team_id}`));
  if (team) return json(team);
  const member = members.find((member) => url.pathname.endsWith(`/customer_members/${member.customer_member_id}`));
  if (member) return json(member);
  throw new Error(`Unexpected usage request ${url.pathname}`);
};

beforeEach(() => {
  registerBaseUrlGetter(() => "http://localhost");
  fetchMock.mockReset();
  fetchMock.mockImplementation((input) => Promise.resolve(serve(input)));
  vi.stubGlobal("fetch", fetchMock);
});
afterEach(() => vi.unstubAllGlobals());

describe("customerHierarchyUsageCall", () => {
  it("should read exact daily activity routes, all pages, and scoped descendant lists", async () => {
    fetchMock.mockImplementation((input) => {
      const url = requestUrl(input);
      if (url.pathname === "/management/v1/customer_members" && url.searchParams.get("parent_team_id") === "team-a") {
        const page = Number(url.searchParams.get("page"));
        const member = page === 1 ? members[0] : { ...members[0], customer_member_id: "member-c" };
        return Promise.resolve(json(list([member], page, 2)));
      }
      return Promise.resolve(serve(input));
    });
    const result = await customerHierarchyUsageCall(request);
    expect(result.metadata.total_spend).toBe(4);
    expect(result.metadata.total_tokens).toBe(600);
    expect(result.metadata.total_api_requests).toBe(10);
    expect(result.results).toHaveLength(1);
    expect(result.results[0].metrics.spend).toBe(4);
    expect(result.results[0].breakdown.model_groups["model-b"].metrics.total_tokens).toBe(500);
    expect(result.api_keys.map((row) => row.label).sort()).toEqual(["Assistant key", "Search key"]);
    expect(result.teams.map((row) => [row.id, row.spend])).toEqual([
      ["team-a", 3],
      ["team-b", 1],
    ]);
    expect(result.members.map((row) => [row.id, row.tokens])).toEqual([
      ["member-a", 100],
      ["member-b", 500],
    ]);
    expect(result.teamResults).toHaveLength(1);
    expect(result.teamResults[0].metrics.spend).toBe(4);
    expect(result.teamResults[0].breakdown.entities["team-a"]).toMatchObject({
      metadata: { alias: "Product" },
      metrics: { spend: 3 },
    });
    expect(result.memberResults).toHaveLength(1);
    expect(result.memberResults[0].breakdown.entities["member-b"].metrics.total_tokens).toBe(500);
    expect(result.memberBudgets[0]).toMatchObject({ id: "member-a", spend: 40, budgetLimit: 10 });
    for (const url of dailyUrls()) {
      expect(url.pathname).toMatch(/^\/management\/v1\/customer_(organizations|teams|members)\/daily\/activity$/);
      expect(url.searchParams.get("start_date")).toBe(request.startDate);
      expect(url.searchParams.get("end_date")).toBe(request.endDate);
      expect(url.searchParams.get("page_size")).toBe("100");
      expect([...url.searchParams.keys()].filter((name) => name.endsWith("_ids"))).toHaveLength(1);
    }
    const organizationPages = dailyUrls().filter((url) => url.searchParams.get("customer_org_ids") === "org-1");
    expect(organizationPages.map((url) => url.searchParams.get("page"))).toEqual(["1", "2"]);
    expect(
      urls()
        .filter((url) => url.pathname === "/management/v1/customer_teams")
        .map((url) => [url.searchParams.get("parent_org_id"), url.searchParams.get("page")]),
    ).toEqual([
      ["org-1", "1"],
      ["org-1", "2"],
    ]);
    expect(
      dailyUrls()
        .find((url) => url.searchParams.has("customer_team_ids"))
        ?.searchParams.get("customer_team_ids"),
    ).toBe("team-a,team-b");
    expect(
      dailyUrls()
        .find((url) => url.searchParams.has("customer_member_ids"))
        ?.searchParams.get("customer_member_ids"),
    ).toBe("member-a,member-c,member-b");
    expect(
      urls().some(
        (url) =>
          url.pathname === "/management/v1/customer_members" &&
          url.searchParams.get("parent_team_id") === "team-a" &&
          url.searchParams.get("page") === "2",
      ),
    ).toBe(true);
  });

  it("should infer member ancestry and retain live spend and effective budget separately from range totals", async () => {
    const result = await customerHierarchyUsageCall({
      ...request,
      scope: { org: "org-1", team: null, member: "member-a" },
    });
    expect(result.team?.id).toBe("team-a");
    expect(result.member).toMatchObject({ id: "member-a", spend: 40, budgetLimit: 10 });
    expect(result.organization).toMatchObject({ spend: 40, budgetLimit: 100, budgetResetAt: "2026-12-01T00:00:00Z" });
    expect(result.metadata.total_spend).toBe(3);
    expect(result.teams).toEqual([{ id: "team-a", label: "Product", spend: 3, tokens: 100, requests: 8 }]);
    expect(Object.keys(result.teamResults[0].breakdown.entities)).toEqual(["team-a"]);
    expect(result.teamResults[0].breakdown.entities["team-a"].metrics.spend).toBe(3);
    expect(result.memberResults[0].breakdown.entities["member-a"].metrics.total_tokens).toBe(100);
    expect(dailyUrls()).toHaveLength(1);
    expect(dailyUrls()[0].searchParams.get("customer_member_ids")).toBe("member-a");
    expect(urls().some((url) => url.pathname === "/management/v1/customer_members")).toBe(false);
  });

  it("should reject mismatched ancestry before requesting any usage", async () => {
    fetchMock.mockImplementation((input) =>
      Promise.resolve(
        requestUrl(input).pathname.endsWith("/customer_teams/team-a")
          ? json({ ...teams[0], parent_org_id: "another-org" })
          : serve(input),
      ),
    );
    await expect(
      customerHierarchyUsageCall({ ...request, scope: { org: "org-1", team: "team-a", member: null } }),
    ).rejects.toThrow("does not belong");
    expect(dailyUrls()).toHaveLength(0);
  });

  it("should never send an empty descendant filter and should retain unassigned usage", async () => {
    fetchMock.mockImplementation((input) =>
      Promise.resolve(requestUrl(input).pathname === "/management/v1/customer_teams" ? json(list([])) : serve(input)),
    );
    const result = await customerHierarchyUsageCall(request);
    expect(dailyUrls().every((url) => url.searchParams.get("customer_org_ids") === "org-1")).toBe(true);
    expect(result.teams).toEqual([
      { id: "", label: "Unassigned or removed team", spend: 4, tokens: 600, requests: 10 },
    ]);
    expect(result.members).toEqual([
      { id: "", label: "Unassigned or removed member", spend: 4, tokens: 600, requests: 10 },
    ]);
  });

  it("should fail a snapshot when a later page fails instead of presenting partial totals", async () => {
    fetchMock.mockImplementation((input) => {
      const url = requestUrl(input);
      return Promise.resolve(
        url.searchParams.get("customer_org_ids") === "org-1" && url.searchParams.get("page") === "2"
          ? json({ error: "Second page unavailable" }, 503)
          : serve(input),
      );
    });
    await expect(customerHierarchyUsageCall(request)).rejects.toThrow("Second page unavailable");
  });

  it("should reject identifiers the comma-separated API cannot represent instead of broadening filters", async () => {
    await expect(
      customerHierarchyUsageCall({ ...request, scope: { org: "org-1,org-2", team: null, member: null } }),
    ).rejects.toThrow();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("should chunk descendant filters and keep every request phase within four concurrent reads", async () => {
    const manyTeams = Array.from({ length: 205 }, (_, index) => ({ ...teams[0], customer_team_id: `team-${index}` }));
    let active = 0;
    let maximumActive = 0;
    fetchMock.mockImplementation(async (input) => {
      active++;
      maximumActive = Math.max(maximumActive, active);
      await Promise.resolve();
      const url = requestUrl(input);
      active--;
      if (url.pathname === "/management/v1/customer_teams") {
        const page = Number(url.searchParams.get("page"));
        return json(list(manyTeams.slice((page - 1) * 100, page * 100), page, 3));
      }
      if (url.pathname === "/management/v1/customer_members") return json(list([]));
      if (url.pathname === "/management/v1/customer_teams/daily/activity") {
        return json({ results: [], metadata: EMPTY_DAILY_ACTIVITY_METADATA });
      }
      return serve(input);
    });

    await customerHierarchyUsageCall(request);

    expect(maximumActive).toBeLessThanOrEqual(4);
    const teamFilters = dailyUrls()
      .filter((url) => url.searchParams.has("customer_team_ids"))
      .map((url) => url.searchParams.get("customer_team_ids")!.split(","));
    expect(teamFilters.map((ids) => ids.length)).toEqual([100, 100, 5]);
    expect(new Set(teamFilters.flat()).size).toBe(manyTeams.length);
    expect(dailyUrls().some((url) => url.searchParams.has("customer_member_ids"))).toBe(false);
  });
});
