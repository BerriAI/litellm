import { appBudgetPolicyCases, chooseAppOption, saveAppBudget } from "./appTestUtils";
import userEvent, { type UserEvent } from "@testing-library/user-event";
import { beforeEach, afterEach, describe, expect, it, vi } from "vitest";
import { registerBaseUrlGetter } from "@/lib/http/runtime";
import { uiHref } from "@/utils/uiHref";
import {
  jsonResponse as json,
  renderWithProviders as render,
  screen,
  fireEvent,
  act,
  waitFor,
  within,
  testQueryClient,
  chooseSelectOption,
} from "../../../../../tests/test-utils";
import CustomerOrganizationApps from "./CustomerOrganizationApps";
import { AppOrganizationBudgetDialog } from "./AppOrganizationBudgetDialog";
import { organizationAppKey } from "./appApi";

const resource = { resource_type: "team" as const, resource_id: "team-chat", alias: "chat", app_id: null };
const app = {
  ...resource,
  app_id: "app-chat",
  auto_create_customers: true,
  spend: 0,
  created_at: "",
  updated_at: "",
  created_by: "",
  updated_by: "",
};
const assignedBudget = {
  app_id: app.app_id,
  customer_org_id: "org-1",
  tier_budget_id: "standard",
  spend: 5,
  effective_budget: {
    budget_limits: [
      { budget_duration: "1mo", max_budget: 100 },
      { budget_duration: "12mo", max_budget: 1000 },
    ],
  },
};
const fetchMock = vi.fn<typeof fetch>();
let configured: boolean;
let bound: boolean;
let assigned: boolean;
let requests: Request[];
let associationStatus: number;
let customAssociation: object | null;

beforeEach(() => {
  configured = true;
  bound = false;
  assigned = false;
  associationStatus = 404;
  customAssociation = null;
  requests = [];
  testQueryClient.clear();
  registerBaseUrlGetter(() => "http://localhost");
  fetchMock.mockReset();
  fetchMock.mockImplementation(async (input) => {
    const request = new Request(input);
    requests.push(request.clone());
    const path = new URL(request.url).pathname;
    if (path.includes("/daily/activity")) return json({ results: [], metadata: { has_more: false, total_pages: 1 } });
    if (path.endsWith("/apps/definition")) return json({ resource_type: configured ? "team" : null });
    if (path.endsWith("/apps/resources"))
      return json([
        { ...resource, app_id: bound ? app.app_id : null },
        { ...resource, resource_id: "team-search", alias: "search", app_id: null },
      ]);
    if (path.endsWith("/apps/app-chat")) return json(app);
    if (path.endsWith("/apps")) {
      if (request.method === "POST") {
        bound = true;
        return json(app);
      }
      return json(bound ? [app] : []);
    }
    if (path.endsWith("/customer_organization_budget_groups"))
      return json({
        default_budgets: [
          { budget_id: "org-standard", name: "Standard", max_budget: 1000 },
          { budget_id: "org-enterprise", name: "Enterprise", max_budget: 5000 },
        ],
      });
    if (path.endsWith("/app_customer_organization_budget_groups"))
      return json({
        default_budgets: [{ budget_id: "standard", name: "Tier 1", max_budget: 100 }],
        revision: "a".repeat(64),
      });
    if (request.method === "PUT") {
      assigned = true;
      return json({ app_id: app.app_id, customer_org_id: "org-1", spend: 0 });
    }
    if (path.endsWith("/customer_organizations/org-1"))
      return assigned
        ? json(customAssociation ?? assignedBudget)
        : json({ detail: "Association unavailable" }, associationStatus);
    if (path.endsWith("/models")) return json({ data: [] });
    throw new Error(`Unexpected request ${path}`);
  });
  vi.stubGlobal("fetch", fetchMock);
});
afterEach(() => vi.unstubAllGlobals());

async function openBudget() {
  const user = userEvent.setup();
  render(<CustomerOrganizationApps organizationId="org-1" />);
  await user.click(await screen.findByRole("button", { name: "Edit budget for chat" }));
  return user;
}

const saveBudget = (user: UserEvent) => saveAppBudget(user, requests, "Save app budget");

describe("customer app budgets", () => {
  it("paginates app budgets in five-row pages without changing member pagination or customer filters", async () => {
    bound = true;
    assigned = true;
    const apps = Array.from({ length: 6 }, (_, index) => ({
      ...app,
      app_id: `app-${index + 1}`,
      alias: `App ${index + 1}`,
    }));
    const serve = fetchMock.getMockImplementation()!;
    fetchMock.mockImplementation(async (input, init) => {
      const path = new URL(new Request(input, init).url).pathname;
      if (path === "/management/v1/apps") return json(apps);
      const requestedApp = apps.find((app) => path === `/management/v1/apps/${app.app_id}`);
      if (requestedApp) return json(requestedApp);
      const selected = apps.find((app) => path === `/management/v1/apps/${app.app_id}/customer_organizations/org-1`);
      if (selected) return json({ ...assignedBudget, app_id: selected.app_id });
      return serve(input, init);
    });
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn();
    render(<CustomerOrganizationApps organizationId="org-1" />, {
      searchParams: "?org=org-1&from=2026-11-01&to=2026-11-18&memberPage=2",
      onUrlUpdate,
    });
    const table = within(await screen.findByRole("table", { name: "App budgets" }));
    await table.findByRole("button", { name: "Edit budget for App 5" });
    expect(table.getAllByRole("row")).toHaveLength(6);
    expect(table.queryByText("App 6")).not.toBeInTheDocument();
    expect(screen.getByTestId("pagination-page-size")).toHaveTextContent("5");
    await user.click(screen.getByRole("button", { name: "Go to next page" }));
    expect(await table.findByText("App 6")).toBeInTheDocument();
    expect(table.getAllByRole("row")).toHaveLength(2);
    expect(table.queryByText("App 1")).not.toBeInTheDocument();
    await user.click(table.getByRole("button", { name: "Edit budget for App 6" }));
    expect(await screen.findByRole("textbox", { name: "App", exact: true })).toHaveValue("App 6");
    await user.click(await screen.findByRole("button", { name: "Cancel", exact: true }));
    await chooseSelectOption(user, screen.getByTestId("pagination-page-size"), "10");
    expect(await table.findByText("App 1")).toBeInTheDocument();
    expect(table.getAllByRole("row")).toHaveLength(7);
    expect(screen.getByRole("button", { name: "Go to previous page" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Go to next page" })).toBeDisabled();
    await waitFor(() =>
      expect(onUrlUpdate).toHaveBeenLastCalledWith(
        expect.objectContaining({
          queryString: "?org=org-1&from=2026-11-01&to=2026-11-18&memberPage=2&org_apps_page_size=10",
        }),
      ),
    );
  });

  it("opens the reusable dialog for one pair, validates periods and preserves temporary increases", async () => {
    assigned = true;
    customAssociation = {
      ...assignedBudget,
      budget: { max_budget: 50, budget_duration: "1mo", rpm_limit: 10 },
      budget_override: { max_budget: 100 },
      budget_override_expires_at: "2099-01-01T00:00:00Z",
    };
    const user = userEvent.setup();
    const onClose = vi.fn();
    render(
      <AppOrganizationBudgetDialog app={app} organizationId="org-1" organizationAlias="Customer Y" onClose={onClose} />,
    );
    const dialog = within(screen.getByRole("dialog", { name: "Customer Y's app budget" }));
    expect(dialog.getByRole("textbox", { name: "App" })).toHaveValue("chat");
    await user.click(await dialog.findByRole("button", { name: "Add another Max Budget" }));
    fireEvent.change(dialog.getByRole("spinbutton", { name: "Additional Max Budget (USD)" }), {
      target: { value: "25" },
    });
    await chooseAppOption(user, dialog.getAllByRole("combobox", { name: "Reset period" })[1], "Custom period");
    fireEvent.change(dialog.getByRole("textbox", { name: "Custom period" }), { target: { value: "30d" } });
    fireEvent.submit(dialog.getByRole("form"));
    await waitFor(() =>
      expect(dialog.getAllByRole("alert").at(-1)).toHaveTextContent("A max budget already uses this period."),
    );
    expect(requests.some((request) => request.method === "PUT")).toBe(false);
    await user.click(dialog.getByRole("button", { name: "Remove limit" }));
    fireEvent.change(dialog.getByRole("spinbutton", { name: "Requests per minute" }), { target: { value: "11" } });
    expect(await saveBudget(user)).toEqual({ budget: { rpm_limit: 11 } });
    await waitFor(() => expect(onClose).toHaveBeenCalledOnce());
    expect(new URL(requests.find((request) => request.method === "PUT")!.url).pathname).toBe(
      "/management/v1/apps/app-chat/customer_organizations/org-1",
    );
    expect(requests.some((request) => new URL(request.url).pathname === "/management/v1/apps")).toBe(false);
  });

  it("offers only the app's models and clears the optional allowlist without removing other limits", async () => {
    bound = true;
    assigned = true;
    customAssociation = {
      ...assignedBudget,
      budget: { rpm_limit: 10, allowed_models: ["app-model"] },
    };
    const serve = fetchMock.getMockImplementation()!;
    fetchMock.mockImplementation(async (input, init) => {
      const url = new URL((input as Request).url);
      if (url.pathname === `/management/v1/apps/resources/team/${resource.resource_id}/models`)
        return json(["app-model"]);
      if (url.pathname.endsWith("/models")) return json({ data: [{ id: "other-model" }] });
      return serve(input, init);
    });
    const user = await openBudget();
    const search = screen.getByRole("combobox", { name: "Search app models" });
    await user.click(search);
    expect(await screen.findByRole("option", { name: "app-model" })).toBeInTheDocument();
    expect(screen.queryByRole("option", { name: "other-model" })).not.toBeInTheDocument();
    await user.keyboard("{Escape}");
    await user.click(within(screen.getByLabelText("app-model")).getByRole("button"));
    expect(await saveBudget(user)).toEqual({ budget: { allowed_models: null } });
  });

  it.each([null, { "test-model": 10000 }])(
    "copies the selected preset's hidden policy into Custom %j",
    async (rates) => {
      bound = true;
      assigned = true;
      const initialPolicy = { model_tpm_limit: { "test-model": 1000 }, model_rpm_limit: { "test-model": 10 } };
      const selectedPolicy = { model_tpm_limit: rates, model_rpm_limit: null, cap_policy: "pause_allow_increase" };
      const serve = fetchMock.getMockImplementation()!;
      fetchMock.mockImplementation(async (input, init) =>
        new URL((input as Request).url).pathname.endsWith("/app_customer_organization_budget_groups")
          ? json({
              default_budgets: [
                { budget_id: "standard", name: "Tier 1", max_budget: 100, budget_duration: "1mo", ...initialPolicy },
                { budget_id: "scale", name: "Tier 2", max_budget: 200, budget_duration: "1mo", ...selectedPolicy },
              ],
            })
          : serve(input, init),
      );
      const user = await openBudget();
      const source = screen.getByRole("combobox", { name: "Budget source" });
      await chooseAppOption(user, source, /^Tier 2/);
      await chooseAppOption(user, source, "Enter custom limits instead");
      fireEvent.change(screen.getByRole("spinbutton", { name: "Max Budget (USD)" }), { target: { value: "250" } });
      expect(await saveBudget(user)).toEqual({
        budget: { max_budget: 250, budget_duration: "1mo", ...selectedPolicy },
      });
    },
  );

  it.each(["custom", "legacy"])("allows a %s policy edit when the catalog fails", async (source) => {
    bound = true;
    assigned = true;
    customAssociation = {
      ...assignedBudget,
      ...(source === "custom" ? { budget: { rpm_limit: 10 } } : { budget_override: { rpm_limit: 10 } }),
    };
    const serve = fetchMock.getMockImplementation()!;
    fetchMock.mockImplementation(async (input, init) =>
      new URL((input as Request).url).pathname.endsWith("/app_customer_organization_budget_groups")
        ? json({ detail: "Catalog unavailable" }, 503)
        : serve(input, init),
    );
    const user = await openBudget();
    fireEvent.change(await screen.findByRole("spinbutton", { name: "Requests per minute" }), {
      target: { value: "11" },
    });
    expect(await saveBudget(user)).toMatchObject({
      budget: { rpm_limit: 11 },
    });
  });

  it("keeps the budget baseline stable and leaves member defaults untouched across a background refresh", async () => {
    bound = true;
    assigned = true;
    const policy = { max_budget: 5, budget_duration: "1mo", rpm_limit: 10 };
    customAssociation = {
      ...assignedBudget,
      budget: policy,
      customer_member_budget_id: "member-policy",
      effective_customer_member_budget: policy,
    };
    const user = await openBudget();
    customAssociation = {
      ...customAssociation,
      budget: { ...policy, max_budget: 25 },
      effective_customer_member_budget: { ...policy, max_budget: 25 },
    };
    await act(() => testQueryClient.refetchQueries({ queryKey: organizationAppKey(app.app_id, "org-1"), exact: true }));
    for (const input of screen.getAllByRole("spinbutton", { name: "Requests per minute" }))
      fireEvent.change(input, { target: { value: "11" } });
    expect(await saveBudget(user)).toEqual({
      budget: { rpm_limit: 11 },
    });
  });

  it("should send only entered first-custom fields when leaving an app organization preset", async () => {
    bound = true;
    assigned = true;
    customAssociation = {
      ...assignedBudget,
      budget_id: "standard",
      budget: null,
      effective_budget: {
        ...assignedBudget.effective_budget,
        tpm_limit: 1700,
        rpm_limit: 17,
        allowed_models: ["test-model"],
      },
    };
    const user = await openBudget();
    await chooseAppOption(user, screen.getByRole("combobox", { name: "Budget source" }), "Enter custom limits instead");
    fireEvent.change(screen.getByRole("spinbutton", { name: "Max Budget (USD)" }), {
      target: { value: "0" },
    });
    expect(await saveBudget(user)).toEqual({
      budget: { max_budget: 0, budget_duration: null },
    });
  });

  it.each([
    { source: "expired grant", expiry: "2000-01-01T00:00:00Z", label: "Tier 1" },
    { source: "active grant", expiry: "2099-01-01T00:00:00Z", label: "Tier 1" },
    { source: "legacy override", expiry: null, label: "Custom" },
  ])("labels a $source as $label", async ({ expiry, label }) => {
    bound = true;
    assigned = true;
    customAssociation = {
      ...assignedBudget,
      ...(expiry === null ? {} : { budget_id: "standard", budget: null }),
      budget_override_id: expiry === null ? "legacy-1" : "grant-1",
      budget_override: { max_budget: 200 },
      budget_override_expires_at: expiry,
    };
    render(<CustomerOrganizationApps organizationId="org-1" />);
    const table = within(await screen.findByRole("table", { name: "App budgets" }));
    expect(await table.findByText(label)).toBeInTheDocument();
    expect(table.queryAllByText("Custom")).toHaveLength(expiry === null ? 1 : 0);
  });

  it("shows current-month app tokens and share of the entire customer spend in the compact table", async () => {
    bound = true;
    assigned = true;
    const serve = fetchMock.getMockImplementation()!;
    const usageRequests: URL[] = [];
    let totalAvailable = false;
    fetchMock.mockImplementation(async (input, init) => {
      const url = new URL(new Request(input, init).url);
      if (!url.pathname.includes("/daily/activity")) return serve(input, init);
      usageRequests.push(url);
      if (url.pathname.endsWith("/aggregated") && !totalAvailable)
        return json({ detail: "Organization total unavailable" }, 503);
      const appScoped = url.searchParams.get("app_id") === app.app_id;
      const metrics = {
        spend: appScoped ? 20 : 100,
        total_tokens: 1500,
        prompt_tokens: 1000,
        completion_tokens: 500,
        api_requests: 3,
      };
      return json({
        results: [
          { date: url.searchParams.get("start_date"), metrics, breakdown: { entities: { "org-1": { metrics } } } },
        ],
        metadata: { has_more: false, total_pages: 1, total_spend: metrics.spend },
      });
    });
    const user = userEvent.setup();
    render(<CustomerOrganizationApps organizationId="org-1" />);
    const table = within(await screen.findByRole("table", { name: "App budgets" }));
    const tokens = await table.findByRole("button", { name: "1,500 total tokens. View breakdown" });
    expect(table.getByRole("columnheader", { name: "Budget & usage" })).toBeInTheDocument();
    expect(table.getByRole("columnheader", { name: "Tokens" })).toBeInTheDocument();
    const retry = await table.findByRole("button", { name: /Retry/ });
    expect(retry).toHaveAttribute("title", "Organization total unavailable");
    expect(tokens).toHaveTextContent("1.5K");
    totalAvailable = true;
    await user.click(retry);
    expect(await table.findByText("20%")).toBeInTheDocument();
    expect(table.getByRole("meter", { name: "Share of customer spend" })).toHaveAttribute("aria-valuenow", "20");
    expect(
      usageRequests.every(
        (url) => (url.searchParams.get("customer_org_ids") ?? url.searchParams.get("customer_org_id")) === "org-1",
      ),
    ).toBe(true);
    expect(usageRequests.filter((url) => url.searchParams.get("app_id") === app.app_id)).toHaveLength(1);
    const monthlyTotal = usageRequests.find((url) => url.pathname.endsWith("/aggregated"));
    expect(monthlyTotal?.searchParams.has("app_id")).toBe(false);
    expect(
      usageRequests.some(
        (url) =>
          url.searchParams.get("app_id") === app.app_id &&
          url.searchParams.get("start_date") === monthlyTotal?.searchParams.get("start_date") &&
          url.searchParams.get("end_date") === monthlyTotal?.searchParams.get("end_date"),
      ),
    ).toBe(true);
    await user.click(tokens);
    expect(await screen.findByText("Input tokens")).toBeInTheDocument();
    expect(screen.getByText("1,000")).toBeInTheDocument();
    expect(screen.getByText("500")).toBeInTheDocument();
  });

  it.each(["loading", "error"] as const)("keeps a legacy assignment while the catalog is %s", async (status) => {
    bound = true;
    assigned = true;
    const serve = fetchMock.getMockImplementation()!;
    let legacyAvailable = false;
    const { promise, resolve } = Promise.withResolvers<void>();
    fetchMock.mockImplementation(async (input, init) => {
      const request = input as Request;
      const path = new URL(request.url).pathname;
      if (request.method === "GET" && path.endsWith("/customer_organizations/org-1"))
        return json({ ...assignedBudget, tier_budget_id: "org-standard" });
      if (path === "/management/v1/customer_organization_budget_groups") {
        if (status === "loading") await promise;
        if (status === "error" && !legacyAvailable) return json({ detail: "Legacy catalog unavailable" }, 503);
      }
      return serve(input, init);
    });
    const user = await openBudget();
    if (status === "error") expect(await screen.findByText("Legacy catalog unavailable")).toBeInTheDocument();
    expect(screen.queryByRole("combobox", { name: "Budget source" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Save app budget" })).not.toBeInTheDocument();
    if (status === "loading") await act(async () => resolve());
    else {
      legacyAvailable = true;
      await user.click(screen.getByRole("button", { name: "Retry" }));
    }
    const source = await screen.findByRole("combobox", { name: "Budget source" });
    const save = screen.getByRole("button", { name: "Save app budget" });
    expect(source).toHaveTextContent("Standard");
    if (status === "loading") {
      await user.click(source);
      expect(await screen.findByRole("option", { name: /^Tier 1/ })).toBeInTheDocument();
      expect(screen.queryByRole("option", { name: /^Enterprise/ })).not.toBeInTheDocument();
      expect(screen.getByRole("option", { name: /^Standard/ })).toHaveAttribute("aria-disabled", "true");
      await user.keyboard("{Escape}");
    }
    await user.click(save);
    await waitFor(() => expect(requests.some((request) => request.method === "PUT")).toBe(true));
    expect(await requests.find((request) => request.method === "PUT")!.json()).toEqual({
      budget_id: "org-standard",
    });
  });

  it("should show recorded period usage separately from cumulative spend", async () => {
    bound = true;
    assigned = true;
    const usage = [
      { source: "window", budget_duration: "1mo", max_budget: 100, spend: 7, reset_at: "2026-12-01T00:00:00Z" },
      { source: "window", budget_duration: "12mo", max_budget: 1000, spend: 12, reset_at: "2027-01-01T00:00:00Z" },
    ];
    const serve = fetchMock.getMockImplementation()!;
    fetchMock.mockImplementation(async (input, init) => {
      const request = input as Request;
      if (new URL(request.url).pathname.endsWith("/customer_organizations/org-1"))
        return json({ ...assignedBudget, spend: 500, budget_usage: usage });
      return serve(input, init);
    });
    render(<CustomerOrganizationApps organizationId="org-1" />);
    await screen.findByRole("button", { name: "Edit budget for chat" });
    const table = within(screen.getByRole("table", { name: "App budgets" }));
    expect(table.queryByText("$500.00")).not.toBeInTheDocument();
    expect(table.queryByRole("columnheader", { name: "Tracked spend (USD)" })).not.toBeInTheDocument();
    expect(table.getByText("$7.00 of $100.00 used")).toBeInTheDocument();
    const monthlyProgress = table.getByRole("progressbar", { name: "App budget used" });
    expect(Number(monthlyProgress.getAttribute("aria-valuenow"))).toBeCloseTo(7);
    expect(table.queryByText("$12.00 / $1,000.00")).not.toBeInTheDocument();
    expect(table.getByRole("button", { name: "Budget details" })).toBeInTheDocument();
    expect(table.getByText("$93.00 remaining")).toBeInTheDocument();
  });
  it("should summarize the monthly group limit with additional limits in details", async () => {
    bound = true;
    assigned = true;
    render(<CustomerOrganizationApps organizationId="org-1" />);
    await screen.findByRole("button", { name: "Edit budget for chat" });
    const table = within(screen.getByRole("table", { name: "App budgets" }));
    expect(await table.findByRole("cell", { name: "Tier 1" })).toBeInTheDocument();
    expect(table.getByText("$100.00")).toBeInTheDocument();
    expect(table.queryByText("$1,000.00")).not.toBeInTheDocument();
    expect(table.getByRole("button", { name: "Budget details" })).toBeInTheDocument();
    expect(table.queryByText("No max budget")).not.toBeInTheDocument();
  });
  it("should route Add app budget to Gateway setup when the app definition is unset", async () => {
    configured = false;
    render(<CustomerOrganizationApps organizationId="org-1" />);
    expect(await screen.findByText(/Define what an app is in your Gateway/)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Add app budget" })).toHaveAttribute("href", uiHref("apps"));
    expect(requests.every((request) => request.method === "GET")).toBe(true);
    expect(screen.queryByRole("combobox")).not.toBeInTheDocument();
  });

  it("selects an app for the current customer and assigns only the shared app budget", async () => {
    const user = userEvent.setup();
    render(<CustomerOrganizationApps organizationId="org-1" organizationAlias="Customer Y" />);
    await user.click(await screen.findByRole("button", { name: "Add app budget" }));
    const dialog = within(await screen.findByRole("dialog", { name: "Customer Y's app budget" }));
    await chooseAppOption(user, dialog.getByRole("combobox", { name: "Select an app" }), "chat");
    expect(dialog.queryByRole("textbox", { name: "Customer organization" })).not.toBeInTheDocument();
    await user.click(screen.getByRole("combobox", { name: "Budget source" }));
    expect(await screen.findByRole("option", { name: /^Tier 1/ })).toBeInTheDocument();
    expect(screen.queryByRole("option", { name: /^Standard/ })).not.toBeInTheDocument();
    await user.click(await screen.findByRole("option", { name: /^Tier 1/ }));
    expect(screen.queryByRole("switch", { name: "Set per-member limits" })).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Save app budget" }));
    await waitFor(() => expect(requests.some((request) => request.method === "PUT")).toBe(true));
    const mutations = requests.filter((request) => request.method !== "GET");
    expect(mutations.map((request) => request.method)).toEqual(["POST", "PUT"]);
    expect(await mutations[0].json()).toEqual({ resource_id: "team-chat", resource_type: "team" });
    expect(new URL(mutations[1].url).pathname).toBe("/management/v1/apps/app-chat/customer_organizations/org-1");
    expect(await mutations[1].json()).toEqual({
      budget_id: "standard",
    });
    expect(await screen.findByRole("button", { name: "Edit budget for chat" })).toBeInTheDocument();
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("should edit in a modal and start a fresh selection after canceling", async () => {
    bound = true;
    assigned = true;
    const user = userEvent.setup();
    render(<CustomerOrganizationApps organizationId="org-1" organizationAlias="Customer Y" />);
    await user.click(await screen.findByRole("button", { name: "Edit budget for chat" }));
    const dialog = within(await screen.findByRole("dialog", { name: "Customer Y's app budget" }));
    expect(dialog.queryByRole("textbox", { name: "Customer organization" })).not.toBeInTheDocument();
    expect(dialog.getByRole("textbox", { name: "App" })).toHaveValue("chat");
    expect(dialog.getByRole("combobox", { name: "Budget source" })).toHaveTextContent("Tier 1");
    await user.click(dialog.getByRole("button", { name: "Cancel" }));
    await user.click(screen.getByRole("button", { name: "Add app budget" }));
    const fresh = within(await screen.findByRole("dialog", { name: "Customer Y's app budget" }));
    expect(fresh.getByRole("combobox", { name: "Select an app" })).toHaveValue("");
    expect(fresh.queryByRole("textbox", { name: "Customer organization" })).not.toBeInTheDocument();
    expect(requests.every((request) => request.method === "GET")).toBe(true);
  });

  it("should save a custom budget without creating a tier or changing the global customer budget", async () => {
    const user = userEvent.setup();
    render(<CustomerOrganizationApps organizationId="org-1" />);
    await user.click(await screen.findByRole("button", { name: "Add app budget" }));
    await chooseAppOption(user, await screen.findByRole("combobox", { name: "Select an app" }), "search");
    await user.click(screen.getByRole("combobox", { name: "Budget source" }));
    await user.click(await screen.findByRole("option", { name: "Enter custom limits instead" }));
    await chooseAppOption(user, screen.getByRole("combobox", { name: "Reset period" }), "/ year");
    act(() => {
      fireEvent.change(screen.getByRole("spinbutton", { name: "Max Budget (USD)" }), { target: { value: "1000" } });
      fireEvent.change(screen.getByRole("spinbutton", { name: "Tokens per minute" }), { target: { value: "50000" } });
      fireEvent.change(screen.getByRole("spinbutton", { name: "Requests per minute" }), { target: { value: "120" } });
    });

    expect(screen.queryByRole("switch", { name: "Set per-member limits" })).not.toBeInTheDocument();
    expect(await saveBudget(user)).toEqual({
      budget: {
        max_budget: 1000,
        budget_duration: "12mo",
        tpm_limit: 50000,
        rpm_limit: 120,
      },
    });
    expect(requests.some((request) => request.method === "PATCH")).toBe(false);
  });

  it("switches the organization budget to a preset without clearing an existing member default", async () => {
    bound = true;
    assigned = true;
    const serve = fetchMock.getMockImplementation()!;
    const customBudget = {
      ...assignedBudget,
      tier_budget_id: null,
      budget_override_id: "custom-budget",
      budget_override: assignedBudget.effective_budget,
      customer_member_budget_id: "member-budget",
      effective_customer_member_budget: { max_budget: 20, budget_duration: "1mo", rpm_limit: 60 },
    };
    fetchMock.mockImplementation(async (input, init) => {
      const request = input as Request;
      if (request.method === "GET" && new URL(request.url).pathname.endsWith("/customer_organizations/org-1"))
        return json(customBudget);
      return serve(input, init);
    });
    const user = await openBudget();
    const limits = within(screen.getByRole("region", { name: "Budget source" }));
    expect(limits.getByRole("spinbutton", { name: "Max Budget (USD)" })).toHaveValue(null);
    const caps = limits.getAllByRole("spinbutton", { name: "Additional Max Budget (USD)" });
    expect(caps[0]).toHaveValue(100);
    expect(caps[1]).toHaveValue(1000);
    await user.click(limits.getByRole("combobox", { name: "Budget source" }));
    await user.click(await screen.findByRole("option", { name: /^Tier 1/ }));
    expect(screen.queryByRole("spinbutton", { name: "Max Budget (USD)" })).not.toBeInTheDocument();
    expect(await saveBudget(user)).toEqual({
      budget_id: "standard",
      budget_override: null,
    });
  });

  it.each(appBudgetPolicyCases)(
    "should save a private org budget and preserve temporary overrides $source",
    async ({ legacy, temporary, policy, expectedPatch }) => {
      bound = true;
      assigned = true;
      const serve = fetchMock.getMockImplementation()!;
      fetchMock.mockImplementation(async (input, init) =>
        new URL((input as Request).url).pathname.endsWith("/app_customer_organization_budget_groups")
          ? json({
              default_budgets: [
                { budget_id: "standard", name: "Tier 1", rpm_limit: 99, allowed_models: ["inherited-model"] },
              ],
            })
          : serve(input, init),
      );
      customAssociation = {
        ...assignedBudget,
        ...policy,
        budget_id: legacy ? null : "private",
        budget_override_id: legacy || temporary ? "override" : null,
      };
      const user = await openBudget();
      expect(screen.getByRole("spinbutton", { name: "Max Budget (USD)" })).toHaveValue(5);
      fireEvent.change(screen.getByRole("spinbutton", { name: "Max Budget (USD)" }), { target: { value: "7" } });
      expect(await saveBudget(user)).toEqual({
        ...expectedPatch,
      });
    },
  );

  it("should keep association read failures visible instead of overwriting an unknown budget", async () => {
    bound = true;
    associationStatus = 503;
    const user = userEvent.setup();
    render(<CustomerOrganizationApps organizationId="org-1" />);
    expect(await screen.findByRole("alert")).toHaveTextContent("Association unavailable");
    await user.click(screen.getByRole("button", { name: "Add app budget" }));
    await chooseAppOption(user, await screen.findByRole("combobox", { name: "Select an app" }), "chat");
    expect(screen.queryByRole("button", { name: "Save app budget" })).not.toBeInTheDocument();
    expect(requests.every((request) => request.method === "GET")).toBe(true);
  });
  it("starts first Custom from the preset and leaves untouched limits and active grants alone", async () => {
    bound = true;
    assigned = true;
    const preset = {
      budget_id: "standard",
      name: "Tier 1",
      max_budget: 500,
      budget_duration: "1mo",
      tpm_limit: 10000,
      rpm_limit: 50,
      budget_limits: [{ budget_duration: "12mo", max_budget: 5000 }],
    };
    customAssociation = {
      ...assignedBudget,
      budget: null,
      budget_id: "standard",
      budget_override: { max_budget: 900 },
      budget_override_expires_at: "2027-01-01T00:00:00Z",
      effective_budget: { ...preset, max_budget: 900 },
    };
    const serve = fetchMock.getMockImplementation()!;
    fetchMock.mockImplementation(async (input, init) =>
      new URL((input as Request).url).pathname.endsWith("/app_customer_organization_budget_groups")
        ? json({ default_budgets: [preset] })
        : serve(input, init),
    );
    const user = await openBudget();
    await user.click(await screen.findByRole("combobox", { name: "Budget source" }));
    await user.click(await screen.findByRole("option", { name: "Enter custom limits instead" }));
    expect(screen.getByRole("spinbutton", { name: "Max Budget (USD)" })).toHaveValue(500);
    expect(screen.getByRole("spinbutton", { name: "Additional Max Budget (USD)" })).toHaveValue(5000);
    expect(screen.getByRole("spinbutton", { name: "Tokens per minute" })).toHaveValue(10000);
    fireEvent.change(screen.getByRole("spinbutton", { name: "Max Budget (USD)" }), { target: { value: "750" } });
    expect(await saveBudget(user)).toEqual({
      budget: { max_budget: 750, budget_duration: "1mo" },
    });
  });
});
