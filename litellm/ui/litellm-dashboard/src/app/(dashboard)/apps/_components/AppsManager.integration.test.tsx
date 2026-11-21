import { appFixture, chooseAppOption, mockAppRequests } from "./appTestUtils";
import userEvent, { type UserEvent } from "@testing-library/user-event";
import { beforeEach, afterEach, describe, expect, it, vi } from "vitest";
import { uiHref } from "@/utils/uiHref";
import {
  jsonResponse as json,
  fireEvent,
  renderWithProviders as render,
  screen,
  waitFor,
  within,
} from "../../../../../tests/test-utils";
import AppsManager from "./AppsManager";
import AppGroupsPage from "../../app-groups/page";
import AppSettingsPage from "../settings/page";
import type { App, AppResource, Budget } from "./appForms";
import type { BudgetGroups } from "../../customers/_components/customerBudgetGroupsPlan";
import type { AppGroup } from "./appGroupApi";

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => ({ accessToken: "test-session", userRole: "Admin", isLoading: false }),
}));
vi.mock("@/app/(dashboard)/hooks/models/useModels", () => ({
  useAllProxyModels: () => ({ data: { data: [] }, isLoading: false, isError: false }),
}));
const resource: AppResource = { resource_type: "team", resource_id: "team-chat", alias: "chat", app_id: null };
const app = appFixture();
const overallPreset: BudgetGroups["default_budgets"][number] = {
  budget_id: "overall-standard",
  name: "App Standard",
  max_budget: 150,
  budget_duration: "1mo",
  rpm_limit: 20,
  tpm_limit: 1000,
  allowed_models: ["model-a"],
  budget_limits: [{ max_budget: 1200, budget_duration: "12mo" }],
};
const yearly = { max_budget: 1200, budget_duration: "12mo", reset_at: "2027-01-01T00:00:00Z" };
const paginatedResources: AppResource[] = Array.from({ length: 30 }, (_, index) => ({
  resource_type: "team",
  resource_id: `team-${index}`,
  alias: `app-${String(index).padStart(2, "0")}`,
  app_id: null,
}));
let definition: { resource_type: string | null };
let configuredApps: App[];
let overallPresets: BudgetGroups["default_budgets"];
let resources: AppResource[];
let requests: Request[];
let network: ReturnType<typeof mockAppRequests>;

beforeEach(() => {
  definition = { resource_type: "team" };
  configuredApps = [];
  overallPresets = [overallPreset];
  resources = [resource];
  const routes: Parameters<typeof mockAppRequests>[0] = {
    "GET /management/v1/app_groups": () => [],
    "GET /management/v1/apps/definition": () => definition,
    "PUT /management/v1/apps/definition": async (request) => {
      definition = await request.json();
      return definition;
    },
    "GET /management/v1/apps/resources": () => resources,
    "GET /management/v1/apps/app-chat": () => configuredApps[0],
    "PATCH /management/v1/apps/app-chat": async (request) => {
      configuredApps = [{ ...configuredApps[0], ...(await request.json()) }];
      return configuredApps[0];
    },
    "GET /management/v1/app_group_budget_groups": () => ({
      default_budgets: overallPresets,
      budget_groups_revision: "g".repeat(64),
    }),
    "GET /management/v1/app_budget_groups": () => ({
      default_budgets: overallPresets,
      budget_groups_revision: "a".repeat(64),
    }),
    "POST /management/v1/apps": async (request) => {
      const created = { ...app, ...(await request.clone().json()) };
      configureApp(created);
      return created;
    },
    "GET /management/v1/customer_organizations/daily/activity": (request) => {
      const url = new URL(request.url);
      const metrics = {
        spend: url.searchParams.has("app_id") || url.searchParams.has("app_group_id") ? 12.5 : 50,
        total_tokens: 1500,
        prompt_tokens: 1000,
        completion_tokens: 500,
        api_requests: 3,
      };
      return {
        results: [{ date: url.searchParams.get("start_date"), metrics }],
        metadata: { has_more: false, total_pages: 1 },
      };
    },
    "GET /management/v1/apps": () => configuredApps,
    "GET /models": () => ({ data: [] }),
    "GET /management/v1/apps/resources/team/team-chat/models": () => [],
  };
  routes["GET /management/v1/customer_teams/daily/activity"] =
    routes["GET /management/v1/customer_organizations/daily/activity"];
  network = mockAppRequests(routes);
  requests = network.requests;
});
afterEach(() => vi.unstubAllGlobals());

const appGroup: AppGroup = {
  app_group_id: "group-suite",
  name: "Product suite",
  budget_id: "group-custom",
  budget: { max_budget: 10000 },
  budget_usage: [{ source: "scalar", max_budget: 10000, budget_duration: null, spend: 0, reset_at: null }],
  spend: 0,
  created_at: "",
  updated_at: "",
  created_by: "",
  updated_by: "",
};

describe("App groups", () => {
  it("shows group usage and filters an app within the same group and date range", async () => {
    network.routes["GET /management/v1/app_groups"] = () => [appGroup];
    configureApp({
      app_group_id: appGroup.app_group_id,
      budget: { max_budget: 2000, budget_duration: "1mo" },
      budget_usage: [{ source: "scalar", max_budget: 2000, budget_duration: "1mo", spend: 1200, reset_at: null }],
    });
    network.routes["GET /management/v1/customer_organizations/daily/activity"] = (request) => {
      const query = new URL(request.url).searchParams;
      const scoped =
        query.get("app_group_id") === appGroup.app_group_id &&
        query.get("start_date") === "2026-11-01" &&
        query.get("end_date") === "2026-11-03";
      const groupSpend = query.has("app_id") ? 2.5 : 12.5;
      const metrics = {
        spend: scoped ? groupSpend : 900,
        total_tokens: 1500,
        prompt_tokens: 1000,
        completion_tokens: 500,
        api_requests: 3,
      };
      return {
        results: [
          {
            date: query.get("start_date"),
            metrics,
            breakdown: { model_groups: { "group-model": { metrics, metadata: {} } } },
          },
        ],
        metadata: { has_more: false, total_pages: 1 },
      };
    };
    const user = userEvent.setup();
    render(<AppGroupsPage />, {
      searchParams: "?app_group=group-suite&from=2026-11-01&to=2026-11-03",
    });
    const models = within(await screen.findByRole("table", { name: "Models" }));
    expect(await models.findByText("$12.50")).toBeInTheDocument();
    const applications = within(screen.getByRole("table", { name: "Applications" }));
    expect(await applications.findByText("$2.50")).toBeInTheDocument();
    expect(applications.getByText("$1,200.00 / $2,000.00")).toBeInTheDocument();
    expect(applications.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "60");
    expect(applications.getByRole("meter", { name: "Overall spend share" })).toHaveAttribute("aria-valuenow", "20");
    expect(screen.getByTitle("1,500 tokens")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Usage over time" })).toBeInTheDocument();
    expect(
      within(screen.getByRole("region", { name: "Current budget" })).getByText("$0.00 / $10,000.00"),
    ).toBeInTheDocument();
    await chooseAppOption(user, screen.getByRole("combobox", { name: "App", exact: true }), "chat");
    await waitFor(() =>
      expect(within(screen.getByRole("table", { name: "Models" })).getByText("$2.50")).toBeInTheDocument(),
    );
    const activity = requests
      .map((request) => new URL(request.url))
      .filter(
        (url) =>
          url.pathname.endsWith("/daily/activity") && url.searchParams.get("app_group_id") === appGroup.app_group_id,
      );
    expect(
      activity.some(
        (url) =>
          url.searchParams.get("app_id") === app.app_id &&
          url.searchParams.get("start_date") === "2026-11-01" &&
          url.searchParams.get("end_date") === "2026-11-03",
      ),
    ).toBe(true);
    const customerLink = new URL(
      screen.getByRole("link", { name: "Customer organization budgets" }).getAttribute("href")!,
      "http://localhost",
    );
    const expectedScope = {
      app_group: "group-suite",
      group_tab: "customers",
      from: "2026-11-01",
      to: "2026-11-03",
      app_id: app.app_id,
    };
    expect(Object.fromEntries(customerLink.searchParams)).toEqual(expectedScope);
  });

  it("assigns an app to a group while preserving its own budget preset", async () => {
    network.routes["GET /management/v1/app_groups"] = () => [appGroup];
    const user = await openAppSettings({ budget_id: overallPreset.budget_id, budget: overallPreset });
    await chooseAppOption(user, await screen.findByRole("combobox", { name: "App group" }), "Product suite");
    expect(await saveAppSettings(user)).toMatchObject({
      app_group_id: "group-suite",
      budget_id: overallPreset.budget_id,
    });
  });

  it("lists groups separately, searches them, and edits a child app without losing the group", async () => {
    network.routes["GET /management/v1/app_groups"] = () => [
      appGroup,
      { ...appGroup, app_group_id: "group-other", name: "Other suite" },
    ];
    network.routes["GET /management/v1/app_groups/group-suite/customer_organizations"] = () => [];
    network.routes["GET /management/v1/customer_organizations"] = () => ({ data: [], meta: { total_pages: 1 } });
    configureApp({ app_group_id: appGroup.app_group_id, budget: { max_budget: 2000 } });
    const user = userEvent.setup();
    render(<AppGroupsPage />);
    const table = within(await screen.findByRole("table", { name: "App groups" }));
    expect(await table.findByRole("button", { name: "Product suite" })).toBeInTheDocument();
    expect(table.getByRole("button", { name: "Other suite" })).toBeInTheDocument();
    fireEvent.change(screen.getByRole("textbox", { name: "Search app groups" }), { target: { value: "Product" } });
    await waitFor(() => expect(table.queryByRole("button", { name: "Other suite" })).not.toBeInTheDocument());
    await user.click(table.getByRole("button", { name: "Product suite" }));
    expect(await screen.findByRole("heading", { name: "Product suite" })).toBeInTheDocument();
    const applications = within(await screen.findByRole("table", { name: "Applications" }));
    await user.click(await applications.findByRole("button", { name: "Edit chat" }));
    const editor = within(await screen.findByRole("region", { name: "Overall app budget" }));
    fireEvent.change(editor.getByRole("spinbutton", { name: "Max Budget (USD)" }), { target: { value: "2500" } });
    expect(await saveAppSettings(user)).toMatchObject({ budget: { max_budget: 2500 } });
    expect(screen.getByRole("heading", { name: "Product suite" })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "App groups" }));
    expect(await screen.findByRole("table", { name: "App groups" })).toBeInTheDocument();
    expect(screen.getByRole("textbox", { name: "Search app groups" })).toHaveValue("Product");
  });

  it("shows ten app groups per page and opens a group from the next page", async () => {
    network.routes["GET /management/v1/app_groups"] = () =>
      Array.from({ length: 11 }, (_, index) => ({
        ...appGroup,
        app_group_id: `group-${index + 1}`,
        name: `Group ${String(index + 1).padStart(2, "0")}`,
      }));
    const user = userEvent.setup();
    render(<AppGroupsPage />);
    const table = within(await screen.findByRole("table", { name: "App groups" }));
    await table.findByRole("button", { name: "Group 10", exact: true });
    expect(table.getAllByRole("row")).toHaveLength(11);
    expect(table.queryByRole("button", { name: "Group 11", exact: true })).not.toBeInTheDocument();
    expect(screen.getByTestId("pagination-page-size")).toHaveTextContent("10");
    await user.click(screen.getByRole("button", { name: "Go to next page" }));
    const group = await table.findByRole("button", { name: "Group 11", exact: true });
    expect(table.getAllByRole("row")).toHaveLength(2);
    await user.click(group);
    expect(await screen.findByRole("heading", { name: "Group 11", exact: true })).toBeInTheDocument();
  });

  it("preselects the catalog default when creating a group with selected apps", async () => {
    network.routes["POST /management/v1/app_groups"] = () => appGroup;
    network.routes["GET /management/v1/app_group_budget_groups"] = () => ({
      default_budgets: [overallPreset],
      default_budget_template_id: overallPreset.budget_id,
      budget_groups_revision: "g".repeat(64),
    });
    resources = [resource, { resource_type: "team", resource_id: "team-search", alias: "search", app_id: null }];
    const user = userEvent.setup();
    render(<AppGroupsPage />);
    await user.click(await screen.findByRole("button", { name: "Create app group" }));
    const dialog = within(await screen.findByRole("dialog", { name: "Create app group" }));
    fireEvent.change(await dialog.findByRole("textbox", { name: "Group name" }), {
      target: { value: "Product suite" },
    });
    await user.click(dialog.getByRole("combobox", { name: "Select apps" }));
    await user.click(await screen.findByRole("option", { name: /chat/ }));
    await user.click(await screen.findByRole("option", { name: /search/ }));
    await user.keyboard("{Escape}");
    expect(dialog.getByRole("combobox", { name: "Budget source" })).toHaveTextContent("App Standard");
    await user.click(dialog.getByRole("button", { name: "Save changes" }));
    await waitFor(() => expect(network.writes).toHaveLength(1));
    const expected = {
      name: "Product suite",
      description: null,
      apps: [
        { resource_type: "team", resource_id: "team-chat" },
        { resource_type: "team", resource_id: "team-search" },
      ],
      budget_id: overallPreset.budget_id,
    };
    expect(await network.writes[0].json()).toEqual(expected);
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
  });

  it("sets a customer cap across the group using the existing customer budget catalog", async () => {
    network.routes["GET /management/v1/app_groups"] = () => [appGroup];
    network.routes["GET /management/v1/app_groups/group-suite/customer_organizations"] = () => [];
    network.routes["GET /management/v1/customer_organizations"] = () => ({
      data: [{ customer_org_id: "customer-one", alias: "Customer One" }],
      meta: { total_pages: 1 },
    });
    network.routes["GET /management/v1/app_customer_organization_budget_groups"] = () => ({
      default_budgets: [{ ...overallPreset, budget_id: "customer-standard", name: "Customer Standard" }],
      budget_groups_revision: "b".repeat(64),
    });
    network.routes["PUT /management/v1/app_groups/group-suite/customer_organizations/customer-one"] = () => ({
      app_group_id: "group-suite",
      customer_org_id: "customer-one",
      budget_id: "customer-standard",
    });
    const user = userEvent.setup();
    render(<AppGroupsPage />, { searchParams: "?app_group=group-suite&group_tab=customers" });
    await user.click(await screen.findByRole("button", { name: "Set customer budget" }));
    const dialog = within(await screen.findByRole("dialog", { name: "Customer budget for app group" }));
    await chooseAppOption(user, await dialog.findByRole("combobox", { name: "Customer organization" }), "Customer One");
    await chooseAppOption(user, dialog.getByRole("combobox", { name: "Budget source" }), /^Customer Standard/);
    await user.click(dialog.getByRole("button", { name: "Save changes" }));
    await waitFor(() => expect(network.writes).toHaveLength(1));
    expect(await network.writes[0].json()).toEqual({ budget_id: "customer-standard" });
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
  });
});

function configureApp(settings: Partial<App>) {
  resources = [{ ...resource, app_id: app.app_id }];
  configuredApps = [{ ...app, ...settings }];
}

async function openAppSettings(settings?: Partial<App>) {
  if (settings) configureApp(settings);
  const user = userEvent.setup();
  render(<AppsManager />);
  await user.click(await screen.findByRole("button", { name: "Edit chat" }));
  return user;
}

it.each(["default", "none", "existing"])("respects the app catalog default with a %s choice", async (choice) => {
  network.routes["GET /management/v1/app_budget_groups"] = () => ({
    default_budgets: [overallPreset],
    default_budget_template_id: overallPreset.budget_id,
    budget_groups_revision: "a".repeat(64),
  });
  const user = await openAppSettings(choice === "existing" ? { budget_id: null, budget: null } : undefined);
  const source = await screen.findByRole("combobox", { name: "Overall app budget" });
  expect(source).toHaveTextContent(choice === "existing" ? "No limit" : "App Standard");
  if (choice === "none") await chooseAppOption(user, source, /^No limit/);
  expect(await saveAppSettings(user, choice === "existing" ? "PATCH" : "POST")).toMatchObject({
    budget_id: choice === "default" ? overallPreset.budget_id : null,
  });
});

it("saves the selected app on its Settings page and keeps member defaults untouched", async () => {
  configureApp({ auto_create_customers: true, customer_margin: 0.025 });
  const user = userEvent.setup();
  render(<AppSettingsPage />, { searchParams: "?app_id=app-chat" });
  const form = within(await screen.findByRole("form", { name: "Defaults for chat" }));
  fireEvent.change(form.getByRole("spinbutton", { name: "Surcharge on customer requests (%)" }), {
    target: { value: "3.5" },
  });
  await user.click(form.getByRole("checkbox", { name: "This app may create new customers" }));
  await user.click(form.getByRole("button", { name: "Save app settings" }));
  await waitFor(() => expect(requests.filter((request) => request.method === "PATCH")).toHaveLength(1));
  const patch = requests.find((request) => request.method === "PATCH")!;
  expect(await patch.json()).toEqual({ budget_id: null, auto_create_customers: false, customer_margin: 0.035 });
  await waitFor(() => expect(form.getByRole("button", { name: "Save app settings" })).toBeDisabled());
  expect(form.getByRole("spinbutton", { name: "Surcharge on customer requests (%)" })).toHaveValue(3.5);
  expect(screen.getByRole("link", { name: "Usage" })).toHaveAttribute(
    "href",
    `${uiHref("apps/usage")}?app_id=app-chat`,
  );
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
});

async function saveAppSettings(user: UserEvent, method: "PATCH" | "POST" = "PATCH", index = 0) {
  await user.click(screen.getByRole("button", { name: "Save app settings" }));
  await waitFor(() => expect(requests.filter((request) => request.method === method)).toHaveLength(index + 1));
  await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
  const request = requests.filter((request) => request.method === method)[index];
  expect(new URL(request.url).pathname).toBe(`/management/v1/apps${method === "PATCH" ? "/app-chat" : ""}`);
  return request.clone().json();
}

async function expectSavedBudget(user: UserEvent, budget: Budget, index = 0) {
  expect(await saveAppSettings(user, "PATCH", index)).toEqual({
    budget,
    auto_create_customers: true,
    customer_margin: null,
  });
}

describe("Gateway app definition", () => {
  it("should render an unset choice without creating apps or selecting a resource layer", async () => {
    definition = { resource_type: null };
    render(<AppsManager />);
    expect(await screen.findByRole("combobox", { name: "Apps are represented by" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Save app definition" })).toBeDisabled();
    expect(screen.queryByRole("button", { name: "Create app" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Add tier" })).not.toBeInTheDocument();
    expect(requests.every((request) => request.method === "GET")).toBe(true);
    expect(requests.some((request) => new URL(request.url).pathname.endsWith("/apps/resources"))).toBe(false);
  });

  it("should preview existing resources and save only the chosen Gateway resource type", async () => {
    definition = { resource_type: null };
    const user = userEvent.setup();
    render(<AppsManager />);
    await chooseAppOption(user, await screen.findByRole("combobox", { name: "Apps are represented by" }), "Teams");
    expect(await screen.findByText("1 existing resources found in Teams")).toBeInTheDocument();
    expect(screen.getByText("chat")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Save app definition" }));
    await waitFor(() => expect(requests.some((request) => request.method === "PUT")).toBe(true));
    expect(await requests.find((request) => request.method === "PUT")!.json()).toEqual({
      resource_type: "team",
      auto_create_customers_without_app: true,
    });
    expect(await screen.findByText("Current definition: Teams")).toBeInTheDocument();
  });

  it.each(["POST", "PATCH"] as const)(
    "saves app settings via %s without changing relocated organization or member defaults",
    async (method) => {
      const user = await openAppSettings(method === "PATCH" ? { default_tier_budget_id: "standard" } : undefined);
      const form = within(await screen.findByRole("form", { name: "Defaults for chat" }));
      expect(form.queryByRole("combobox", { name: "Budget group" })).not.toBeInTheDocument();
      const body = await saveAppSettings(user, method);
      expect(body).not.toHaveProperty("default_tier_budget_id");
      expect(body).not.toHaveProperty("default_customer_member_budget");
      expect(body).not.toHaveProperty("default_customer_member_budget_id");
      expect(body).toMatchObject({
        ...(method === "POST" ? { resource_id: "team-chat", resource_type: "team" } : {}),
        auto_create_customers: true,
      });
      expect(requests.some((request) => new URL(request.url).pathname.includes("/tiers"))).toBe(false);
    },
  );

  it("changes app rates without persisting unchanged caps or overwriting member defaults", async () => {
    const user = await openAppSettings({
      budget: overallPreset,
      default_customer_member_budget: overallPreset,
    });
    const overall = within(await screen.findByRole("region", { name: "Overall app budget" }));
    expect(overall.getByRole("spinbutton", { name: "Max Budget (USD)" })).toHaveValue(150);
    fireEvent.change(overall.getByRole("spinbutton", { name: "Requests per minute" }), { target: { value: "0" } });
    expect(screen.getByRole("link", { name: "Manage per-member defaults and budget groups" })).toHaveAttribute(
      "href",
      `${uiHref("apps/budget-groups")}?app_id=app-chat`,
    );
    await expectSavedBudget(user, { rpm_limit: 0 });
  });

  it("keeps overall presets linked until a custom edit", async () => {
    const user = await openAppSettings({
      budget_id: overallPreset.budget_id,
      budget: { ...overallPreset, max_budget: 250 },
    });
    const overall = within(await screen.findByRole("region", { name: "Overall app budget" }));
    const source = overall.getByRole("combobox", { name: "Overall app budget" });
    expect(source).toHaveTextContent("App Standard");
    expect(overall.queryByRole("spinbutton")).not.toBeInTheDocument();
    await user.click(source);
    expect(await screen.findByRole("option", { name: /^App Standard/ })).toBeInTheDocument();
    expect(screen.queryByRole("option", { name: /^Tier 1/ })).not.toBeInTheDocument();
    await user.keyboard("{Escape}");
    const linked = await saveAppSettings(user);
    expect(linked.budget_id).toBe(overallPreset.budget_id);
    expect(linked).not.toHaveProperty("budget");
    await user.click(screen.getByRole("button", { name: "Edit chat" }));
    const customEditor = within(await screen.findByRole("region", { name: "Overall app budget" }));
    await chooseAppOption(
      user,
      customEditor.getByRole("combobox", { name: "Overall app budget" }),
      "Enter custom limits instead",
    );
    expect(customEditor.getByRole("spinbutton", { name: "Max Budget (USD)" })).toHaveValue(150);
    fireEvent.change(customEditor.getByRole("spinbutton", { name: "Requests per minute" }), { target: { value: "0" } });
    await expectSavedBudget(user, { max_budget: 150, budget_duration: "1mo", rpm_limit: 0 }, 1);
  });

  it("clears an overall preset with No limit without changing member defaults", async () => {
    const user = await openAppSettings({ budget_id: overallPreset.budget_id, budget: overallPreset });
    await chooseAppOption(user, screen.getByRole("combobox", { name: "Overall app budget" }), /^No limit/);
    const body = await saveAppSettings(user);
    expect(body.budget_id).toBeNull();
    expect(body).not.toHaveProperty("budget");
    expect(body).not.toHaveProperty("default_customer_member_budget");
  });

  it("copies the newly selected preset into Custom and preserves its draft through No limit", async () => {
    const copiedPolicy = {
      max_budget: 250,
      rpm_limit: 25,
      allowed_models: ["model-b"],
      budget_limits: [{ max_budget: 3000, budget_duration: "12mo" }],
      model_tpm_limit: { "model-b": 200 },
      cap_policy: "pause_allow_increase" as const,
    };
    const selected = { ...overallPreset, ...copiedPolicy, budget_id: "overall-scale", name: "App Scale" };
    overallPresets = [overallPreset, selected];
    const user = await openAppSettings({
      budget_id: overallPreset.budget_id,
      budget: {
        ...overallPreset,
        soft_budget: 50,
        model_tpm_limit: { "model-a": 100 },
        model_rpm_limit: { "model-a": 10 },
        cap_policy: "pause_until_reset",
      },
    });
    const overall = within(await screen.findByRole("region", { name: "Overall app budget" }));
    const source = overall.getByRole("combobox", { name: "Overall app budget" });
    await chooseAppOption(user, source, /^App Scale/);
    await chooseAppOption(user, source, "Enter custom limits instead");
    expect(overall.getByRole("spinbutton", { name: "Max Budget (USD)" })).toHaveValue(250);
    expect(overall.getByRole("spinbutton", { name: "Requests per minute" })).toHaveValue(25);
    fireEvent.change(overall.getByRole("spinbutton", { name: "Requests per minute" }), { target: { value: "0" } });
    await chooseAppOption(user, source, /^No limit/);
    await chooseAppOption(user, source, "Enter custom limits instead");
    expect(overall.getByRole("spinbutton", { name: "Requests per minute" })).toHaveValue(0);
    const expectedBudget: Budget = {
      ...copiedPolicy,
      budget_duration: "1mo",
      soft_budget: null,
      rpm_limit: 0,
      model_rpm_limit: null,
    };
    await expectSavedBudget(user, expectedBudget);
  });

  it("recalculates the reset when changing a stored app spend period", async () => {
    const budget = { rpm_limit: 20, budget_limits: [yearly] };
    const user = await openAppSettings({ budget, default_customer_member_budget: budget });
    const editor = within(await screen.findByRole("region", { name: "Overall app budget" }));
    const periods = editor.getAllByRole("combobox", { name: "Reset period" });
    await user.click(periods[0]);
    expect((await screen.findAllByRole("option")).map((option) => option.textContent)).toEqual([
      "/ month",
      "/ year",
      "No reset",
      "Custom period",
    ]);
    await user.keyboard("{Escape}");
    await user.click(periods[1]);
    expect((await screen.findAllByRole("option")).map((option) => option.textContent)).toEqual([
      "/ month",
      "/ year",
      "Custom period",
    ]);
    await user.click(screen.getByRole("option", { name: "/ month" }));
    await expectSavedBudget(user, { budget_limits: [{ ...yearly, budget_duration: "1mo", reset_at: null }] });
  });

  it.each(["clear", "remove"])("removes only the selected app spend window using %s", async (action) => {
    const budget = {
      tpm_limit: 1000,
      rpm_limit: 20,
      budget_limits: [{ max_budget: 100, budget_duration: "1mo" }, yearly],
    };
    const user = await openAppSettings({ budget, default_customer_member_budget: budget });
    const editor = within(await screen.findByRole("region", { name: "Overall app budget" }));
    await user.click(editor.getAllByRole("combobox", { name: "Reset period" })[1]);
    expect(await screen.findByRole("option", { name: "/ year" })).toHaveAttribute("aria-disabled", "true");
    await user.keyboard("{Escape}");
    const amount = editor.getAllByRole("spinbutton", { name: "Additional Max Budget (USD)" })[0];
    expect(amount).toHaveAttribute("placeholder", "No limit");
    expect(amount).toHaveAccessibleDescription(
      "Leave an amount blank to remove that max budget. Other max budgets and TPM/RPM limits stay unchanged.",
    );
    if (action === "clear") {
      fireEvent.change(amount, { target: { value: "" } });
      await user.click(editor.getAllByRole("combobox", { name: "Reset period" })[2]);
      expect(await screen.findByRole("option", { name: "/ month" })).toHaveAttribute("aria-disabled", "true");
      await user.keyboard("{Escape}");
    } else await user.click(editor.getAllByRole("button", { name: "Remove limit" })[0]);
    await expectSavedBudget(user, { budget_limits: [yearly] });
  });

  it("adds multiple spend limits and can add again after removing every additional row", async () => {
    const user = await openAppSettings({ budget: { max_budget: 500, tpm_limit: 2000, rpm_limit: 100 } });
    const editor = within(await screen.findByRole("region", { name: "Overall app budget" }));
    const add = editor.getByRole("button", { name: "Add another Max Budget" });
    expect(editor.queryByText("Additional Max Budget (USD)")).not.toBeInTheDocument();
    await user.click(add);
    fireEvent.change(editor.getByRole("spinbutton", { name: "Additional Max Budget (USD)" }), {
      target: { value: "1300" },
    });
    await user.click(add);
    expect(editor.getByText("Additional Max Budget (USD)")).toBeInTheDocument();
    expect(editor.getAllByRole("spinbutton", { name: "Additional Max Budget (USD)" })).toHaveLength(2);
    await user.click(editor.getAllByRole("combobox", { name: "Reset period" })[2]);
    await user.click(await screen.findByRole("option", { name: "Custom period" }));
    const duplicatePeriod = editor.getByRole("textbox", { name: "Custom period" });
    fireEvent.change(duplicatePeriod, { target: { value: "30d" } });
    expect(duplicatePeriod).toBeInvalid();
    expect(editor.getByRole("alert")).toHaveTextContent("A max budget already uses this period.");
    await user.click(screen.getByRole("button", { name: "Save app settings" }));
    expect(requests.filter((request) => request.method === "PATCH")).toHaveLength(0);
    await user.click(editor.getAllByRole("button", { name: "Remove limit" })[1]);
    await user.click(editor.getByRole("button", { name: "Remove limit" }));
    expect(editor.queryByText("Additional Max Budget (USD)")).not.toBeInTheDocument();
    await user.click(add);
    fireEvent.change(editor.getByRole("spinbutton", { name: "Additional Max Budget (USD)" }), {
      target: { value: "1300" },
    });
    await user.click(add);
    fireEvent.change(editor.getAllByRole("spinbutton", { name: "Additional Max Budget (USD)" })[1], {
      target: { value: "9000" },
    });
    await expectSavedBudget(user, {
      budget_limits: [
        { max_budget: 1300, budget_duration: "1mo" },
        { max_budget: 9000, budget_duration: "12mo" },
      ],
    });
  });

  it("should send the app's customer creation switch turned off when the admin unchecks it", async () => {
    const user = await openAppSettings();
    await user.click(await screen.findByRole("checkbox", { name: "This app may create new customers" }));
    expect(await saveAppSettings(user, "POST")).toMatchObject({
      resource_id: "team-chat",
      auto_create_customers: false,
    });
  });

  it("should send the app's surcharge as a fraction and clear it when the field is emptied", async () => {
    const user = await openAppSettings();
    const field = await screen.findByRole("spinbutton", { name: "Surcharge on customer requests (%)" });
    fireEvent.change(field, { target: { value: "2.5" } });
    expect(await saveAppSettings(user, "POST")).toMatchObject({
      customer_margin: 0.025,
    });
    await user.click(screen.getByRole("button", { name: "Edit chat" }));
    fireEvent.change(await screen.findByRole("spinbutton", { name: "Surcharge on customer requests (%)" }), {
      target: { value: "" },
    });
    expect(await saveAppSettings(user, "PATCH")).toMatchObject({
      customer_margin: null,
    });
  });

  it.each(["1001", "-1"])("should refuse a surcharge of %s percent with a message and send nothing", async (value) => {
    const user = await openAppSettings();
    fireEvent.change(await screen.findByRole("spinbutton", { name: "Surcharge on customer requests (%)" }), {
      target: { value },
    });
    await user.click(screen.getByRole("button", { name: "Save app settings" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Enter a surcharge between 0 and 1000 percent, or leave it empty.",
    );
    expect(requests.some((request) => request.method === "POST")).toBe(false);
  });

  it("should save the no-app customer creation switch without changing the app definition", async () => {
    const user = userEvent.setup();
    render(<AppsManager />);
    await user.click(await screen.findByRole("checkbox", { name: "Requests with no app may create new customers" }));
    await waitFor(() => expect(requests.some((request) => request.method === "PUT")).toBe(true));
    expect(await requests.find((request) => request.method === "PUT")!.json()).toEqual({
      resource_type: "team",
      auto_create_customers_without_app: false,
    });
  });

  it("shows monthly usage, tokens and overall share and opens app controls from Edit", async () => {
    resources = [
      { ...resource, app_id: "app-chat" },
      { resource_type: "team", resource_id: "team-search", alias: "search", app_id: null },
    ];
    configuredApps = [{ ...app, spend: 900, customer_margin: 0.05, auto_create_customers: false }];
    const user = userEvent.setup();
    render(<AppsManager />);
    await screen.findByRole("button", { name: "1,500 total tokens. View breakdown" });
    const table = screen.getByRole("table", { name: "Applications" });
    const [chat, search] = within(table).getAllByRole("row").slice(1);
    expect(within(chat).getByRole("link", { name: "chat" })).toHaveAttribute(
      "href",
      `${uiHref("apps/usage")}?app_id=app-chat`,
    );
    expect(chat).toHaveTextContent("$12.50");
    expect(within(chat).getByRole("meter", { name: "Overall spend share" })).toHaveAttribute("aria-valuenow", "25");
    expect(within(chat).getByRole("button", { name: "1,500 total tokens. View breakdown" })).toHaveTextContent("1.5K");
    expect(search).toHaveTextContent("search");
    await user.click(within(chat).getByRole("button", { name: "Edit chat" }));
    const dialog = within(await screen.findByRole("dialog", { name: "Edit chat" }));
    expect(await dialog.findByRole("spinbutton", { name: "Surcharge on customer requests (%)" })).toHaveValue(5);
    expect(dialog.getByRole("checkbox", { name: "This app may create new customers" })).not.toBeChecked();
  });

  it("does not label a linked budget Custom when its catalog is unavailable", async () => {
    configureApp({ budget_id: overallPreset.budget_id, budget: overallPreset });
    network.routes["GET /management/v1/app_budget_groups"] = () => json({ detail: "Catalog unavailable" }, 503);
    render(<AppsManager />);
    await screen.findByRole("button", { name: "Edit chat" });
    expect(within(screen.getByRole("table", { name: "Applications" })).queryByText("Custom")).not.toBeInTheDocument();
  });

  it("finishes saving app settings while monthly usage is still loading", async () => {
    const activity = Promise.withResolvers<void>();
    network.routes["GET /management/v1/customer_organizations/daily/activity"] = () =>
      activity.promise.then(() => ({ results: [], metadata: { has_more: false, total_pages: 1 } }));
    try {
      const user = await openAppSettings({ budget_id: overallPreset.budget_id, budget: overallPreset });
      fireEvent.change(await screen.findByRole("spinbutton", { name: "Surcharge on customer requests (%)" }), {
        target: { value: "2.5" },
      });
      expect(await saveAppSettings(user)).toMatchObject({ customer_margin: 0.025 });
    } finally {
      activity.resolve();
    }
  });

  it("should show the current budget window's spend and cap instead of lifetime spend", async () => {
    configureApp({
      spend: 900,
      budget: { max_budget: 1000, budget_limits: [{ budget_duration: "1mo", max_budget: 100 }] },
      budget_usage: [
        { source: "scalar", budget_duration: null, max_budget: 1000, spend: 900, reset_at: null },
        { source: "window", budget_duration: "1mo", max_budget: 100, spend: 40, reset_at: null },
      ],
    });
    render(<AppsManager />);
    await screen.findByRole("link", { name: "chat" });
    const usage = within(screen.getByRole("table", { name: "Applications" })).getByRole("progressbar");
    expect(usage).toHaveAttribute("aria-valuenow", "40");
    expect(usage).toHaveAttribute("aria-valuetext", expect.stringContaining("$100"));
    expect(within(screen.getByRole("table", { name: "Applications" })).queryByText("$900.00")).not.toBeInTheDocument();
  });

  it.each(["next page", "page 2 link"])("shows the second page of applications through %s", async (navigation) => {
    resources = paginatedResources;
    const user = userEvent.setup();
    render(<AppsManager />, { searchParams: navigation === "page 2 link" ? "?page=2" : "" });
    if (navigation === "next page") {
      await screen.findByRole("button", { name: "Edit app-00" });
      expect(screen.queryByRole("button", { name: "Edit app-25" })).not.toBeInTheDocument();
      await user.click(screen.getByRole("button", { name: "Go to next page" }));
    }
    expect(await screen.findByRole("button", { name: "Edit app-25" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Edit app-00" })).not.toBeInTheDocument();
  });

  it("should open the edit modal from an application link", async () => {
    render(<AppsManager />, { searchParams: "?app=team-chat" });
    expect(await screen.findByRole("form", { name: "Defaults for chat" })).toBeInTheDocument();
    expect(requests.some((request) => new URL(request.url).pathname.endsWith("/apps/resources"))).toBe(true);
    expect(screen.queryByRole("table", { name: "Applications" })).not.toBeInTheDocument();
  });

  it("should show the definition setting when a linked application has no app definition to load", async () => {
    definition = { resource_type: null };
    render(<AppsManager />, { searchParams: "?app=team-chat" });
    expect(await screen.findByRole("combobox")).toBeInTheDocument();
    await waitFor(() => expect(screen.queryByRole("status")).not.toBeInTheDocument());
  });

  it.each(["?page_size=0", "?page=-3", "?page=0", "?page_size=-5&page=9"])(
    "should still list applications when the URL carries %s",
    async (searchParams) => {
      render(<AppsManager />, { searchParams });
      expect(await screen.findByRole("button", { name: "Edit chat" })).toBeInTheDocument();
    },
  );

  it("should narrow the list with the search box and say when nothing matches", async () => {
    render(<AppsManager />);
    await screen.findByRole("button", { name: "Edit chat" });
    fireEvent.change(screen.getByRole("textbox", { name: "Search applications" }), { target: { value: "billing" } });
    expect(screen.queryByRole("button", { name: "Edit chat" })).not.toBeInTheDocument();
    expect(screen.getByText("No applications match your search.")).toBeInTheDocument();
  });

  it("should return from an application's settings to the list", async () => {
    const user = await openAppSettings();
    expect(await screen.findByRole("form", { name: "Defaults for chat" })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Cancel" }));
    expect(await screen.findByRole("table", { name: "Applications" })).toBeInTheDocument();
  });

  it("should show definition failures without presenting an empty resource list", async () => {
    definition = { resource_type: null };
    network.routes["GET /management/v1/apps/definition"] = () => json({ detail: "Definition unavailable" }, 503);
    render(<AppsManager />);
    expect(await screen.findByRole("alert")).toHaveTextContent("Definition unavailable");
    expect(screen.queryByRole("combobox")).not.toBeInTheDocument();
  });
});
