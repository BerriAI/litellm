import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import {
  act,
  fireEvent,
  renderWithProviders,
  screen,
  testQueryClient,
  waitFor,
  within,
} from "../../../tests/test-utils";
import IssuesInbox from "./IssuesInbox";
import IssueDetail from "./IssueDetail";
import { FAMILY_SETTINGS, RETIREMENT_ISSUE, SPEND_ISSUE } from "./issueFixtures";
import { EMPTY_DAILY_ACTIVITY_METADATA } from "@/components/UsagePage/dailyActivityApi";
import type { Issue } from "./issueApi";
import type { components } from "@/lib/http/schema";

const auth = vi.hoisted(() => ({
  accessToken: "test-session",
  userId: "test-admin",
  userRole: "Admin",
  isAuthorized: true,
  isViewOnly: false,
}));
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({ default: () => auth }));
const calls: { path: string; query: URLSearchParams; method: string; body: unknown }[] = [];
let detail: Issue;
let failList = false;
let failActivity = false;
let staleStatus = false;
let empty = false;
let allFailed = false;
let deniedDetail = false;
let resourceTotal = 1;
let modelBreakdown: NonNullable<components["schemas"]["BoundedDailyActivityResponse"]["model_breakdown"]> | null = null;
let families = FAMILY_SETTINGS;
const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

function activity() {
  const successful = allFailed ? 0 : 40;
  const metrics = {
    api_requests: 42,
    successful_requests: successful,
    failed_requests: 42 - successful,
    spend: 12,
    total_tokens: 4200,
    prompt_tokens: 4000,
    completion_tokens: 200,
    cache_read_input_tokens: 0,
    cache_creation_input_tokens: 0,
    total_response_time_ms: 8400,
    timed_requests: 42,
  };
  return {
    model_breakdown: modelBreakdown,
    results: [{ date: "2026-11-04", metrics, breakdown: {} }],
    metadata: {
      ...EMPTY_DAILY_ACTIVITY_METADATA,
      total_api_requests: 42,
      total_successful_requests: successful,
      total_failed_requests: 42 - successful,
      total_spend: 12,
      total_tokens: 4200,
      total_response_time_ms: 8400,
      total_timed_requests: 42,
    },
  };
}

function listResponse(query: URLSearchParams) {
  if (failList) return json({ detail: "Unavailable" }, 503);
  const offset = Number(query.get("offset") ?? 0);
  const emptyPage = { data: [], total: 0, limit: 25, offset };
  if (empty) return json(emptyPage);
  const next = { ...SPEND_ISSUE, id: "second-page", scope_label: "Second key", title: "Second key spend increased" };
  let data = offset === 0 ? [SPEND_ISSUE, RETIREMENT_ISSUE] : [next];
  const family = query.get("family");
  if (family) data = data.filter((issue) => issue.family === family);
  const page = { data, total: family ? data.length : 26, limit: 25, offset };
  return json(page);
}

function resourceKeyPage(query: URLSearchParams) {
  const offset = Number(query.get("offset") ?? 0);
  const total = query.get("start_date") === detail.range_start ? resourceTotal : 1;
  const key = {
    api_key: "resource-key",
    metadata: { key_alias: "Related checkout key", user_email: "owner@example.com" },
    metrics: { spend: 12, api_requests: 42, failed_requests: 2 },
  };
  const page = { api_keys: offset < total ? [key] : [], total_api_keys: total, limit: 20, offset };
  return json(page);
}

describe("unified issue investigations", () => {
  beforeEach(() => {
    testQueryClient.clear();
    calls.length = 0;
    detail = structuredClone(SPEND_ISSUE);
    failList = false;
    failActivity = false;
    staleStatus = false;
    empty = false;
    allFailed = false;
    deniedDetail = false;
    resourceTotal = 1;
    modelBreakdown = null;
    families = FAMILY_SETTINGS;
    auth.isViewOnly = false;
    auth.isAuthorized = true;
    auth.userRole = "Admin";
    auth.userId = "test-admin";
    vi.stubGlobal("requestIdleCallback", (callback: IdleRequestCallback) =>
      setTimeout(() => callback({ didTimeout: false, timeRemaining: () => 50 }), 0),
    );
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const request = input instanceof Request ? input : new Request(input);
      const url = new URL(request.url);
      const body = request.method === "GET" ? null : await request.clone().json();
      const call = { path: url.pathname, query: url.searchParams, method: request.method, body };
      calls.push(call);
      if (url.pathname === "/management/v1/users/me")
        return json({ user_id: auth.userId, user_role: auth.userRole === "Admin" ? "proxy_admin" : "internal_user" });
      if (url.pathname === "/issues/foreign-issue") return json({ detail: "Forbidden private-project metadata" }, 403);
      if (url.pathname === "/issues") return listResponse(url.searchParams);
      if (url.pathname === "/issues/monitoring")
        return json({ enabled: true, incidents_used: 0, incident_limit: null });
      if (url.pathname === "/issues/families") return json(families);
      if (url.pathname === "/issues/families/spend") return json(FAMILY_SETTINGS.data[0]);
      if (url.pathname.endsWith("/activity"))
        return failActivity ? json({ detail: "Unavailable" }, 503) : json(activity());
      if (url.pathname.endsWith("/keys")) return resourceKeyPage(url.searchParams);
      if (url.pathname.endsWith("/users")) {
        const page = {
          users: [
            { user_id: "resource-user", user_alias: "Checkout owner", spend: 12, api_requests: 42, failed_requests: 2 },
          ],
          total_users: 1,
          limit: 20,
          offset: 0,
        };
        return json(page);
      }
      if (url.pathname.endsWith("/signals")) {
        const page = { data: [], total: 0, limit: 20, offset: 0 };
        return json(page);
      }
      if (url.pathname.startsWith("/issues/")) {
        if (deniedDetail) return json({ detail: "Access revoked" }, 403);
        if (request.method === "PATCH") {
          if (staleStatus) return json({ detail: "Issue changed" }, 409);
          detail = { ...detail, ...body };
        }
        return json(detail);
      }
      return json({ detail: "Unexpected request" }, 404);
    });
  });
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("shows retirement and key spend in one server-paginated inbox with issue destinations", async () => {
    const user = userEvent.setup();
    renderWithProviders(<IssuesInbox />);
    expect(await screen.findByRole("link", { name: "Spend on Checkout key rose to 20× normal" })).toHaveAttribute(
      "href",
      "/ui/alerts?issue=issue-checkout-spend",
    );
    expect(screen.getByRole("link", { name: RETIREMENT_ISSUE.title })).toBeInTheDocument();
    expect(screen.getByText("$600.00 on Nov 4, 2026 UTC, compared with the usual $30.00.")).toBeInTheDocument();
    expect(screen.getByRole("list", { name: "Alerts", exact: true })).toBeInTheDocument();
    expect(screen.queryByRole("table")).not.toBeInTheDocument();
    expect(calls.filter((call) => call.path === "/issues")).toHaveLength(1);
    expect(calls.some((call) => call.path.includes("activity") || call.path.includes("deprecations"))).toBe(false);
    await user.click(screen.getByRole("button", { name: "Go to next page", exact: true }));
    expect(await screen.findByRole("link", { name: "Spend on Second key rose to 20× normal" })).toBeInTheDocument();
    expect(
      calls
        .filter((call) => call.path === "/issues")
        .at(-1)
        ?.query.get("offset"),
    ).toBe("25");
    fireEvent.change(screen.getByLabelText("Alert type"), { target: { value: "retirement" } });
    expect(await screen.findByRole("link", { name: RETIREMENT_ISSUE.title })).toBeInTheDocument();
    expect(
      calls
        .filter((call) => call.path === "/issues")
        .at(-1)
        ?.query.get("offset"),
    ).toBe("0");
  });

  it("keeps old performance list links usable", async () => {
    renderWithProviders(<IssuesInbox />, { searchParams: "type=usage&usage_status=resolved" });
    await screen.findByRole("link", { name: "Spend on Checkout key rose to 20× normal" });
    const request = calls.find((call) => call.path === "/issues");
    expect(request?.query.get("kind")).toBe("performance");
    expect(request?.query.get("status")).toBe("resolved");
  });

  it("hides cached global issues when the reader loses global access", async () => {
    renderWithProviders(<IssuesInbox />);
    await screen.findByRole("link", { name: "Spend on Checkout key rose to 20× normal" });
    await act(async () => {
      testQueryClient.setQueriesData(
        { queryKey: ["currentUser"] },
        { user_id: "test-admin", user_role: "internal_user_viewer" },
      );
    });
    await waitFor(() =>
      expect(screen.queryByRole("link", { name: "Spend on Checkout key rose to 20× normal" })).not.toBeInTheDocument(),
    );
  });

  it("renders a clear empty state and retries list failures", async () => {
    failList = true;
    const user = userEvent.setup();
    renderWithProviders(<IssuesInbox />);
    expect(await screen.findByRole("alert")).toHaveTextContent("Could not load alerts");
    failList = false;
    empty = true;
    await user.click(screen.getByRole("button", { name: "Retry" }));
    expect(await screen.findByText("No open alerts.")).toBeInTheDocument();
    expect(screen.getByText(/enough recorded usage history/)).toBeInTheDocument();
  });

  it("resolves directly from the feed without fetching individual alerts", async () => {
    renderWithProviders(<IssuesInbox />);
    await screen.findByRole("link", { name: "Spend on Checkout key rose to 20× normal" });
    const feed = screen.getByRole("list", { name: "Alerts", exact: true });
    await userEvent.click(within(feed).getAllByRole("button", { name: "Mark resolved" })[0]);
    await waitFor(() =>
      expect(calls.some((call) => call.method === "PATCH" && call.path === `/issues/${SPEND_ISSUE.id}`)).toBe(true),
    );
    expect(
      calls.filter(
        (call) =>
          call.method === "GET" &&
          call.path.startsWith("/issues/") &&
          !call.path.endsWith("/families") &&
          !call.path.endsWith("/monitoring"),
      ),
    ).toHaveLength(0);
    expect(calls.find((call) => call.method === "PATCH")?.body).toEqual({
      status: "resolved",
      expected_status: "open",
    });
  });

  it("leaves missing history unavailable instead of fabricating a chart", async () => {
    detail = {
      ...SPEND_ISSUE,
      comparisons: SPEND_ISSUE.comparisons?.map((comparison) => ({ ...comparison, history: [] })),
    };
    renderWithProviders(<IssueDetail id={detail.id} />);
    expect(await screen.findByText(/Daily history was not saved/)).toBeInTheDocument();
    expect(screen.queryByRole("figure")).not.toBeInTheDocument();
  });

  it("explains the change and links to entity Activity without loading investigation data", async () => {
    const user = userEvent.setup();
    renderWithProviders(<IssueDetail id={SPEND_ISSUE.id} />);
    expect(
      await screen.findByRole("heading", { name: "Spend on Checkout key rose to 20× normal" }),
    ).toBeInTheDocument();
    expect(screen.getByText("$600.00 on Nov 4, 2026 UTC, compared with the usual $30.00.")).toBeInTheDocument();
    expect(screen.getByRole("figure", { name: "Daily spend" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "View key activity" })).toHaveAttribute(
      "href",
      "/ui/api-keys?key=reporting-test-id&detail_tab=activity&activity_from=2026-10-28&activity_to=2026-11-04",
    );
    expect(calls.some((call) => /\/(activity|keys|users|signals)$/.test(call.path))).toBe(false);
    await user.click(screen.getByText("Alert details", { selector: "summary" }));
    const table = await screen.findByRole("table", { name: "Saved alert comparisons" });
    expect(within(table).getByText("$600.00")).toBeInTheDocument();
    expect(within(table).getByText("$30.00")).toBeInTheDocument();
    expect(within(table).getByText("At least $90.00")).toBeInTheDocument();
    await user.click(screen.getByText("Resources in this traffic", { selector: "summary" }));
    expect(await screen.findByRole("link", { name: "Related checkout key" })).toHaveAttribute(
      "href",
      "/ui/api-keys?key=resource-key&detail_tab=activity&activity_from=2026-10-28&activity_to=2026-11-04",
    );
    await user.click(screen.getByRole("tab", { name: "Key owners", exact: true }));
    expect(await screen.findByRole("link", { name: "Checkout owner" })).toBeInTheDocument();
    expect(calls.some((call) => call.path.includes("/user/daily"))).toBe(false);
  });

  it.each([false, null, undefined])(
    "keeps saved evidence without fetching unavailable activity (%s)",
    async (canView) => {
      detail = { ...SPEND_ISSUE, can_view_activity: canView };
      renderWithProviders(<IssueDetail id={SPEND_ISSUE.id} />);
      await userEvent.click(await screen.findByText("Alert details", { selector: "summary" }));
      expect(await screen.findByRole("table", { name: "Saved alert comparisons" })).toBeInTheDocument();
      expect(screen.getByText("You don’t have access to this alert’s activity.")).toBeInTheDocument();
      expect(screen.queryByText("Resources in this traffic")).not.toBeInTheDocument();
      expect(calls.some((call) => /\/(activity|keys|users|impact)$/.test(call.path))).toBe(false);
    },
  );

  it("keeps exact totals and shows the explicit remainder from bounded model contributions", async () => {
    modelBreakdown = {
      limit: 20,
      total_models: 137,
      models: Array.from({ length: 20 }, (_, index) => ({
        model: index === 0 ? "Other" : `model-${index}`,
        metrics: { spend: 0.1, api_requests: 1, successful_requests: 1, failed_requests: 0, total_tokens: 10 },
      })),
      other: {
        model_count: 117,
        metrics: { spend: 10, api_requests: 22, successful_requests: 20, failed_requests: 2, total_tokens: 4000 },
      },
    };
    detail = { ...detail, cohort: { scope_type: "provider", scope_id: "provider", model_pairs: [] } };
    renderWithProviders(<IssueDetail id={SPEND_ISSUE.id} />);
    await userEvent.click(await screen.findByText("View activity for this alert", { selector: "summary" }));
    const remainder = await screen.findByRole("row", { name: /Other \(117 models\)/ });
    expect(within(remainder).getByText("$10.00")).toBeInTheDocument();
    expect(within(remainder).getByText("4,000")).toBeInTheDocument();
    expect(screen.getByText("Other")).toBeInTheDocument();
    expect(screen.getByText("42")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "$12.00" })).toBeInTheDocument();
    expect(calls.filter((call) => call.path.endsWith("/activity"))).toHaveLength(1);
  });

  it("preserves saved evidence when the Activity date range changes", async () => {
    resourceTotal = 41;
    detail = { ...detail, cohort: { scope_type: "provider", scope_id: "provider", model_pairs: [] } };
    renderWithProviders(<IssueDetail id={SPEND_ISSUE.id} />, { searchParams: "resourcespage=3" });
    await userEvent.click(await screen.findByText("Alert details", { selector: "summary" }));
    await userEvent.click(await screen.findByText("View activity for this alert", { selector: "summary" }));
    await screen.findByText("Total Requests");
    await userEvent.click(screen.getByText("Resources in this traffic", { selector: "summary" }));
    await screen.findByRole("link", { name: "Related checkout key" });
    expect(calls.find((call) => call.path.endsWith("/keys"))?.query.get("offset")).toBe("40");
    fireEvent.click(screen.getByRole("button", { name: /28 Oct.*4 Nov/ }));
    fireEvent.change(screen.getByDisplayValue("2026-10-28"), { target: { value: "2026-10-30" } });
    await userEvent.click(screen.getByRole("button", { name: "Apply" }));
    await waitFor(() =>
      expect(
        calls
          .filter((call) => call.path.endsWith("/activity"))
          .at(-1)
          ?.query.get("start_date"),
      ).toBe("2026-10-30"),
    );
    expect(screen.getByText("Observed · 2026-11-04 UTC")).toBeInTheDocument();
    expect(screen.getByText("At least $90.00")).toBeInTheDocument();
    await waitFor(() =>
      expect(
        calls
          .filter((call) => call.path.endsWith("/keys"))
          .at(-1)
          ?.query.get("offset"),
      ).toBe("0"),
    );
    expect(await screen.findByRole("link", { name: "Related checkout key" })).toBeInTheDocument();
    const firstNarrowedPage = calls.find(
      (call) => call.path.endsWith("/keys") && call.query.get("start_date") === "2026-10-30",
    );
    expect(firstNarrowedPage?.query.get("offset")).toBe("0");
    await userEvent.click(screen.getByRole("button", { name: "Reset range" }));
    expect(screen.queryByRole("button", { name: "Reset range" })).not.toBeInTheDocument();
  });

  it("updates the issue with expected status and recovers from a stale change", async () => {
    const user = userEvent.setup();
    staleStatus = true;
    renderWithProviders(<IssueDetail id={SPEND_ISSUE.id} />);
    await user.click(await screen.findByRole("button", { name: "Mark resolved" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("This alert changed");
    expect(calls.find((call) => call.method === "PATCH")?.body).toEqual({
      status: "resolved",
      expected_status: "open",
    });
    staleStatus = false;
    await user.click(screen.getByRole("button", { name: "Refresh", exact: true }));
    await user.click(screen.getByRole("button", { name: "Mark resolved" }));
    expect(await screen.findByRole("button", { name: "Reopen" })).toBeInTheDocument();
  });

  it("shows the saved status while dependent refreshes are stalled", async () => {
    const user = userEvent.setup();
    renderWithProviders(<IssuesInbox />);
    await screen.findByRole("link", { name: "Spend on Checkout key rose to 20× normal" });
    renderWithProviders(<IssueDetail id={SPEND_ISSUE.id} />);
    const context = within(await screen.findByRole("complementary", { name: "Alert context" }));
    await context.findByRole("button", { name: "Mark resolved" });
    const fetchBefore = vi.mocked(globalThis.fetch).getMockImplementation()!;
    let resume: (() => void) | undefined;
    const stalled = new Promise<void>((resolve) => {
      resume = resolve;
    });
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const request = input instanceof Request ? input : new Request(input, init);
      if (request.method === "GET" && new URL(request.url).pathname.startsWith("/issues")) await stalled;
      return fetchBefore(input, init);
    });
    try {
      await user.click(context.getByRole("button", { name: "Mark resolved" }));
      expect(await context.findByRole("button", { name: "Reopen" })).toBeInTheDocument();
      expect(context.queryByRole("button", { name: "Mark resolved" })).not.toBeInTheDocument();
    } finally {
      await act(async () => {
        resume?.();
      });
    }
  });

  it("uses the same retirement source/provenance while keeping migration details available", async () => {
    detail = RETIREMENT_ISSUE;
    renderWithProviders(<IssueDetail id={detail.id} />);
    expect(await screen.findByText("recommended-model")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /Provider deprecation notice/ })).toHaveAttribute(
      "href",
      "https://provider.example/model-retirements",
    );
    expect(screen.getByRole("link", { name: "Migration details" })).toHaveAttribute(
      "href",
      "/ui/alerts?type=model-deprecations&model=provider%2Fretiring-model",
    );
    expect(screen.getByRole("link", { name: "View model activity" })).toHaveAttribute(
      "href",
      "/ui/models-and-endpoints?activity_model=provider%2Fretiring-model&detail_tab=activity&activity_from=2026-10-28&activity_to=2026-11-04",
    );
    expect(calls.some((call) => call.path === `/issues/${RETIREMENT_ISSUE.id}/activity`)).toBe(false);
  });

  it("does not show invalid per-success averages when all requests failed", async () => {
    allFailed = true;
    detail = { ...detail, cohort: { scope_type: "provider", scope_id: "provider", model_pairs: [] } };
    renderWithProviders(<IssueDetail id={detail.id} />);
    await userEvent.click(await screen.findByText("View activity for this alert", { selector: "summary" }));
    await screen.findByText("Total Requests");
    expect(screen.getByLabelText("Alert activity")).not.toHaveTextContent(/NaN|Infinity|∞/);
  });

  it("retries failed Activity without losing saved issue evidence", async () => {
    failActivity = true;
    detail = { ...detail, cohort: { scope_type: "provider", scope_id: "provider", model_pairs: [] } };
    renderWithProviders(<IssueDetail id={detail.id} />);
    await userEvent.click(await screen.findByText("Alert details", { selector: "summary" }));
    await userEvent.click(await screen.findByText("View activity for this alert", { selector: "summary" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Could not load activity");
    expect(screen.getByText("At least $90.00")).toBeInTheDocument();
    failActivity = false;
    await userEvent.click(screen.getByRole("button", { name: "Retry" }));
    expect(await screen.findByText("Total Requests")).toBeInTheDocument();
    expect(calls.filter((call) => call.path.endsWith("/activity"))).toHaveLength(2);
  });

  it("does not invent user attribution for historical team activity", async () => {
    detail = { ...SPEND_ISSUE, cohort: { scope_type: "team", scope_id: "team-example", model_pairs: [] } };
    renderWithProviders(<IssueDetail id={detail.id} />, { searchParams: "resources=users" });
    await userEvent.click(await screen.findByText("Resources in this traffic", { selector: "summary" }));
    expect(await screen.findByRole("link", { name: "Related checkout key" })).toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: "Key owners", exact: true })).not.toBeInTheDocument();
    expect(calls.some((call) => call.path.endsWith("/users"))).toBe(false);
  });

  it("lets a read-only admin inspect an issue without mutation controls", async () => {
    auth.isViewOnly = true;
    renderWithProviders(<IssueDetail id={detail.id} />);
    await screen.findByRole("heading", {
      name: detail.kind === "model_deprecation" ? detail.title : "Spend on Checkout key rose to 20× normal",
    });
    expect(screen.queryByRole("button", { name: "Mark resolved" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Dismiss" })).not.toBeInTheDocument();
    expect(calls.some((call) => call.method === "PATCH")).toBe(false);
  });

  it.each([false, null, undefined])(
    "hides status actions without explicit write permission (%s)",
    async (permission) => {
      auth.userRole = "Internal User";
      auth.userId = "team-member";
      detail = {
        ...detail,
        cohort: { scope_type: "team", scope_id: "team-example", model_pairs: [] },
        can_update_status: permission,
      };
      renderWithProviders(<IssueDetail id={detail.id} />);
      await screen.findByRole("heading", {
        name: detail.kind === "model_deprecation" ? detail.title : "Spend on Checkout key rose to 20× normal",
      });
      expect(screen.queryByRole("button", { name: "Mark resolved" })).not.toBeInTheDocument();
      expect(screen.queryByRole("button", { name: "Dismiss" })).not.toBeInTheDocument();
    },
  );

  it("opens an authorized key issue for an internal user without reading global issues", async () => {
    auth.userRole = "Internal User";
    auth.userId = "member-owner";
    renderWithProviders(<IssueDetail id={detail.id} />);
    await screen.findByRole("heading", {
      name: detail.kind === "model_deprecation" ? detail.title : "Spend on Checkout key rose to 20× normal",
    });
    expect(screen.getByRole("heading", { name: "Spend on Checkout key rose to 20× normal" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Resource activity" })).toHaveAttribute(
      "href",
      "/ui/api-keys?key=reporting-test-id&detail_tab=activity&activity_from=2026-10-28&activity_to=2026-11-04",
    );
    expect(calls.some((call) => call.path === "/issues")).toBe(false);
    expect(calls.some((call) => call.path.includes("/user/daily"))).toBe(false);
    await userEvent.click(screen.getByRole("button", { name: "Mark resolved" }));
    await waitFor(() =>
      expect(calls.find((call) => call.method === "PATCH")?.body).toEqual({
        status: "resolved",
        expected_status: "open",
      }),
    );
  });

  it("shows a scoped permission failure without exposing foreign metadata or fetching activity", async () => {
    auth.userRole = "Internal User";
    auth.userId = "member-owner";
    renderWithProviders(<IssueDetail id="foreign-issue" />);
    expect(await screen.findByRole("alert")).toHaveTextContent("You don’t have access to this alert.");
    expect(screen.getByRole("link", { name: "My usage" })).toHaveAttribute("href", "/ui/usage?view=my-usage");
    expect(screen.queryByText(/private-project/)).not.toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Spend on Checkout key rose to 20× normal" })).not.toBeInTheDocument();
    expect(calls.filter((call) => call.path.startsWith("/issues"))).toHaveLength(1);
    expect(calls.some((call) => call.path.endsWith("/activity") || call.path.endsWith("/keys"))).toBe(false);
  });

  it("removes cached issue details after a scoped permission revocation", async () => {
    auth.userRole = "Internal User";
    auth.userId = "member-owner";
    renderWithProviders(<IssueDetail id={detail.id} />);
    await screen.findByRole("heading", {
      name: detail.kind === "model_deprecation" ? detail.title : "Spend on Checkout key rose to 20× normal",
    });
    deniedDetail = true;
    await act(async () => {
      await testQueryClient.invalidateQueries({ queryKey: ["issues"] });
    });
    expect(await screen.findByRole("alert")).toHaveTextContent("You don’t have access to this alert.");
    expect(screen.queryByRole("heading", { name: "Spend on Checkout key rose to 20× normal" })).not.toBeInTheDocument();
    expect(screen.queryByText("Total Requests")).not.toBeInTheDocument();
    expect(screen.queryByRole("link", { name: "Resource activity" })).not.toBeInTheDocument();
  });

  it("does not present supporting request context as another triggered condition", async () => {
    detail = {
      ...SPEND_ISSUE,
      comparisons: SPEND_ISSUE.comparisons?.map((comparison) =>
        comparison.metric === "requests" ? { ...comparison, threshold: null } : comparison,
      ),
    };
    renderWithProviders(<IssueDetail id={detail.id} />);
    await userEvent.click(await screen.findByText("Alert details", { selector: "summary" }));
    const table = await screen.findByRole("table", { name: "Saved alert comparisons" });
    const row = within(table).getByRole("row", { name: /^Requests/ });
    expect(row).toHaveTextContent("1,200");
    expect(row).toHaveTextContent("Context only");
    expect(row).not.toHaveTextContent(/At least|At most/);
    expect(within(table).getByRole("row", { name: /^Spend/ })).toHaveTextContent("At least $90.00");
  });

  it.each(["resolved", "dismissed"] as const)("only reopens active %s retirement issues", async (status) => {
    detail = { ...RETIREMENT_ISSUE, status, active: false };
    const view = renderWithProviders(<IssueDetail id={detail.id} />);
    await screen.findByRole("heading", {
      name: detail.kind === "model_deprecation" ? detail.title : "Spend on Checkout key rose to 20× normal",
    });
    expect(screen.queryByRole("button", { name: "Reopen" })).not.toBeInTheDocument();
    view.unmount();
    testQueryClient.clear();
    detail = { ...RETIREMENT_ISSUE, status, active: true };
    renderWithProviders(<IssueDetail id={detail.id} />);
    expect(await screen.findByRole("button", { name: "Reopen" })).toBeInTheDocument();
    expect(calls.some((call) => call.method === "PATCH")).toBe(false);
  });

  it.each([
    { total: 0, label: "No default policies yet" },
    { total: 5, label: "Thresholds vary by scope" },
  ])("describes incomplete or differing defaults without a fabricated multiplier: $total", async ({ total, label }) => {
    families = {
      ...FAMILY_SETTINGS,
      data: FAMILY_SETTINGS.data.map((family) => ({
        ...family,
        total_rules: total,
        enabled: null,
        policies: family.policies.map((policy) => ({ ...policy, multiplier: null })),
      })),
    };
    renderWithProviders(<IssuesInbox />);
    await userEvent.click(await screen.findByRole("button", { name: "Settings" }));
    expect((await screen.findAllByText(new RegExp(label))).length).toBeGreaterThan(0);
    expect(screen.getByRole("dialog")).not.toHaveTextContent("Mixed×");
  });

  it("uses four family controls and sends only changed policy fields atomically", async () => {
    const user = userEvent.setup();
    renderWithProviders(<IssuesInbox />);
    await user.click(await screen.findByRole("button", { name: "Settings" }));
    expect(screen.getByRole("link", { name: "Email delivery settings" })).toHaveAttribute(
      "href",
      "/ui/logging-and-alerts?tab=email-alerts",
    );
    await user.click(await screen.findByRole("button", { name: "Edit Unexpected spend" }));
    fireEvent.change(screen.getByLabelText("Increase factor"), { target: { value: "10" } });
    await user.click(screen.getByRole("button", { name: "Save changes" }));
    await waitFor(() =>
      expect(calls.find((call) => call.path === "/issues/families/spend")?.body).toEqual({
        policies: [{ direction: "increase", multiplier: 10 }],
      }),
    );
    expect(calls.some((call) => call.path.includes("alert-rules"))).toBe(false);
  });

  it.each([0, 1, 3])("distinguishes %i delayed monitoring checks from insufficient history", async (delayed) => {
    families = {
      ...FAMILY_SETTINGS,
      data: FAMILY_SETTINGS.data.map((family) => ({
        ...family,
        evaluation_status_counts: {
          pending: 0,
          complete: 0,
          delayed: family.family === "spend" ? delayed : 0,
          insufficient_coverage: 0,
          insufficient_history: family.family === "latency" ? 5 : 0,
        },
      })),
    };
    renderWithProviders(<IssuesInbox />);
    await userEvent.click(await screen.findByRole("button", { name: "Settings" }));
    await screen.findByRole("button", { name: "Edit Unexpected spend" });
    if (delayed === 0) {
      expect(screen.queryByText(/catching up/)).not.toBeInTheDocument();
    } else {
      expect(screen.getByRole("status")).toHaveTextContent(
        `${delayed} ${delayed === 1 ? "check is" : "checks are"} catching up and will resume automatically.`,
      );
      expect(screen.getByRole("status")).toHaveTextContent("Some resources have not been checked yet.");
    }
  });

  it("shows alert processing backlog even when no rule evaluation is delayed", async () => {
    families = { ...FAMILY_SETTINGS, projection_pending: true };
    renderWithProviders(<IssuesInbox />);
    await userEvent.click(await screen.findByRole("button", { name: "Settings" }));
    expect(
      await screen.findByText("Alert processing is catching up and will resume automatically."),
    ).toBeInTheDocument();
    expect(screen.queryByText("Some resources have not been checked yet.")).not.toBeInTheDocument();
  });
});
