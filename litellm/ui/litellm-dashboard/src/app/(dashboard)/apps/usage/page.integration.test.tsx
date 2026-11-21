import { afterEach, beforeEach, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import { EMPTY_DAILY_ACTIVITY_METADATA } from "@/components/UsagePage/dailyActivityApi";
import { uiHref } from "@/utils/uiHref";
import { fireEvent, jsonResponse, renderWithProviders, screen, waitFor, within } from "../../../../../tests/test-utils";
import { calendarDate, customerPresetRange } from "../../customers/_components/customerUsageRange";
import AppUsagePage from "./page";
import { appFixture, mockAppRequests } from "../_components/appTestUtils";

const app = appFixture({ alias: "Assistant app", resource_id: "team-a" });
const initialUrl = "?app_id=app-chat&from=2026-11-01&to=2026-11-03";
const urls = () => network.requests.map((request) => new URL(request.url));
let budgetFailed: boolean;
let budgetAmount: number;
let customerCount: number;
let unassigned: boolean;
let network: ReturnType<typeof mockAppRequests>;

async function customerBudget(request: Request) {
  if (budgetFailed) return jsonResponse({ detail: "Customer budget unavailable" }, 503);
  if (request.method === "PUT") budgetAmount = (await request.json()).budget.max_budget;
  const orgId = new URL(request.url).pathname.split("/").at(-1);
  const budget =
    orgId === "org-1" ? { max_budget: budgetAmount, budget_duration: "1mo", tpm_limit: 1000, rpm_limit: 50 } : null;
  return {
    app_id: app.app_id,
    customer_org_id: orgId,
    spend: 500,
    budget,
    effective_budget: budget,
    budget_usage: budget
      ? [
          { source: "scalar", budget_duration: "1mo", max_budget: budgetAmount, spend: 80 },
          { source: "window", budget_duration: "12mo", max_budget: 1000, spend: 500 },
        ]
      : [],
  };
}

beforeEach(() => {
  budgetFailed = false;
  budgetAmount = 100;
  customerCount = 2;
  unassigned = false;
  const routes: Parameters<typeof mockAppRequests>[0] = {
    "GET /management/v1/apps/app-chat": () => app,
    "GET /management/v1/app_customer_organization_budget_groups": () => ({ default_budgets: [] }),
    "GET /management/v1/apps/resources/team/team-a/models": () => [],
    ...Object.fromEntries(
      Array.from({ length: 6 }, (_, index) => [
        `GET /management/v1/apps/app-chat/customer_organizations/org-${index + 1}`,
        customerBudget,
      ]),
    ),
    "PUT /management/v1/apps/app-chat/customer_organizations/org-1": customerBudget,
    "GET /management/v1/customer_organizations/daily/activity": (request) => {
      const url = new URL(request.url);
      const page = Number(url.searchParams.get("page"));
      const month = url.searchParams.get("end_date") !== "2026-11-03";
      const metrics = {
        spend: (page === 1 ? 30 : 10) * (month ? 2 : 1),
        total_tokens: page === 1 ? 300 : 100,
        prompt_tokens: page === 1 ? 300 : 100,
        completion_tokens: 0,
        api_requests: page === 1 ? 3 : 1,
        successful_requests: page === 1 ? 3 : 1,
        failed_requests: 0,
      };
      const isUnassigned = unassigned && page === 2;
      const entity = { metrics, metadata: isUnassigned ? {} : { alias: `Customer ${page}` }, api_key_breakdown: {} };
      return {
        results: [
          {
            date: "2026-11-01",
            metrics,
            breakdown: {
              entities: { [isUnassigned ? "Unassigned" : `org-${page}`]: entity },
              model_groups: { "model-a": entity },
            },
          },
        ],
        metadata: {
          ...EMPTY_DAILY_ACTIVITY_METADATA,
          page,
          has_more: page < customerCount,
          total_pages: customerCount,
        },
      };
    },
  };
  network = mockAppRequests(routes);
});
afterEach(() => vi.unstubAllGlobals());

it("shows app totals across paginated customers and keeps the app and dates in customer links", async () => {
  renderWithProviders(<AppUsagePage />, { searchParams: initialUrl });
  const first = await screen.findByRole("link", { name: "Customer 1" });
  expect(first).toHaveAttribute(
    "href",
    `${uiHref("customers/usage")}?org=org-1&from=2026-11-01&to=2026-11-03&app_id=app-chat`,
  );
  expect(await screen.findByRole("heading", { name: "Assistant app" })).toBeInTheDocument();
  expect(screen.getByRole("link", { name: "Settings" })).toHaveAttribute(
    "href",
    `${uiHref("apps/settings")}?app_id=app-chat&from=2026-11-01&to=2026-11-03`,
  );
  expect(screen.getByRole("link", { name: "Budget groups" })).toHaveAttribute(
    "href",
    `${uiHref("apps/budget-groups")}?app_id=app-chat&from=2026-11-01&to=2026-11-03`,
  );
  expect(
    within(screen.getByRole("region", { name: "Application usage totals" })).getByText("$40.00"),
  ).toBeInTheDocument();
  const table = within(screen.getByRole("table", { name: "Customer organizations" }));
  await table.findByText("Within budget");
  const rows = table.getAllByRole("row").slice(1);
  expect(rows[0]).toHaveTextContent("Customer 1");
  expect(rows[0]).toHaveTextContent("75%");
  expect(rows[0]).toHaveTextContent("$80.00 / $100.00");
  expect(within(rows[0]).getAllByRole("progressbar")).toHaveLength(1);
  expect(rows[0]).toHaveTextContent("1,000 TPM");
  expect(rows[0]).toHaveTextContent("50 RPM");
  expect(rows[1]).toHaveTextContent("Customer 2");
  expect(rows[1]).toHaveTextContent("25%");
  expect(rows[1]).toHaveTextContent("$20.00");
  const activity = urls().filter((url) => url.pathname.endsWith("/daily/activity"));
  const selected = activity.filter((url) => url.searchParams.get("end_date") === "2026-11-03");
  expect(selected.map((url) => url.searchParams.get("page"))).toEqual(["1", "2"]);
  for (const url of activity) {
    expect(url.searchParams.get("app_id")).toBe("app-chat");
  }
  const month = customerPresetRange("month");
  const monthly = activity.filter((url) => url.searchParams.get("end_date") !== "2026-11-03");
  expect(monthly.map((url) => url.searchParams.get("page"))).toEqual(["1", "2"]);
  expect(monthly[0].searchParams.get("start_date")).toBe(calendarDate(month.from));
  expect(monthly[0].searchParams.get("end_date")).toBe(calendarDate(month.to));
});

it("edits only the selected app and customer budget and refreshes its table row", async () => {
  const user = userEvent.setup();
  renderWithProviders(<AppUsagePage />, { searchParams: initialUrl });
  await user.click(await screen.findByRole("button", { name: "Edit budget for Customer 1" }));
  const dialog = within(await screen.findByRole("dialog", { name: "Customer 1's app budget" }));
  const amount = await dialog.findByRole("spinbutton", { name: "Max Budget (USD)" });
  fireEvent.change(amount, { target: { value: "120" } });
  await user.click(dialog.getByRole("button", { name: "Save app budget" }));
  await waitFor(() => expect(network.writes).toHaveLength(1));
  const [request] = network.writes;
  expect(request.method).toBe("PUT");
  expect(new URL(request.url).pathname).toBe("/management/v1/apps/app-chat/customer_organizations/org-1");
  expect(await request.json()).toEqual({ budget: { max_budget: 120, budget_duration: "1mo" } });
  expect(await screen.findByText("$80.00 / $120.00")).toBeInTheDocument();
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  expect(urls().some((url) => url.pathname === "/management/v1/customer_organizations/org-1")).toBe(false);
});

it("shows budget load failures without claiming customers have no limit and supports retry", async () => {
  budgetFailed = true;
  const user = userEvent.setup();
  renderWithProviders(<AppUsagePage />, { searchParams: initialUrl });
  const table = within(await screen.findByRole("table", { name: "Customer organizations" }));
  expect(await table.findAllByRole("alert")).toHaveLength(2);
  expect(table.queryByText("No limit")).not.toBeInTheDocument();
  expect(table.getByText("75%")).toBeInTheDocument();
  expect(table.getByRole("button", { name: /300 total tokens/ })).toBeInTheDocument();
  expect(table.getByText("$60.00")).toBeInTheDocument();
  budgetFailed = false;
  await user.click(table.getAllByRole("button", { name: /retry/i })[0]);
  expect(await table.findByText("$80.00 / $100.00")).toBeInTheDocument();
});

it("keeps unassigned spend and tokens without offering organization navigation or editing", async () => {
  unassigned = true;
  renderWithProviders(<AppUsagePage />, { searchParams: initialUrl });
  const row = await screen.findByRole("row", { name: /Unassigned customer/ });
  expect(row).toHaveTextContent("$20.00");
  expect(row).toHaveTextContent("25%");
  expect(within(row).getByRole("button", { name: /100 total tokens/ })).toBeInTheDocument();
  expect(within(row).queryByRole("link")).not.toBeInTheDocument();
  expect(within(row).queryByRole("button", { name: /edit/i })).not.toBeInTheDocument();
  expect(urls().some((url) => url.pathname.includes("/customer_organizations/Unassigned"))).toBe(false);
});

it("loads budgets only for visible customers and loads the next page on demand", async () => {
  customerCount = 6;
  const user = userEvent.setup();
  renderWithProviders(<AppUsagePage />, { searchParams: initialUrl });
  await screen.findByText("Within budget");
  const associationUrls = () => urls().filter((url) => url.pathname.includes("/apps/app-chat/customer_organizations/"));
  expect(associationUrls()).toHaveLength(5);
  expect(associationUrls().some((url) => url.pathname.endsWith("/org-6"))).toBe(false);
  await user.click(screen.getByRole("button", { name: /next page/i }));
  expect(await screen.findByRole("link", { name: "Customer 6" })).toBeInTheDocument();
  await waitFor(() => expect(associationUrls()).toHaveLength(6));
});

it("does not fetch unscoped customer usage when no app is selected", () => {
  renderWithProviders(<AppUsagePage />);
  expect(screen.getByRole("heading", { name: "Select an app" })).toBeInTheDocument();
  expect(fetch).not.toHaveBeenCalled();
});

it("shows an app access error instead of an empty usage report", async () => {
  network.routes["GET /management/v1/apps/app-chat"] = () => jsonResponse({ detail: "App access denied" }, 403);
  renderWithProviders(<AppUsagePage />, { searchParams: initialUrl });
  expect(await screen.findByRole("alert")).toHaveTextContent("App access denied");
  expect(urls().every((url) => url.pathname.endsWith("/apps/app-chat"))).toBe(true);
});
