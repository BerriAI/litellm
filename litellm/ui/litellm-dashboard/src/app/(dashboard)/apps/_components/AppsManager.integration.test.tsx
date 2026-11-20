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
import AppSettingsPage from "../settings/page";
import type { App, AppResource, Budget } from "./appForms";
import type { BudgetGroups } from "../../customers/_components/customerBudgetGroupsPlan";

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
        spend: url.searchParams.has("app_id") ? 12.5 : 50,
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
  network = mockAppRequests(routes);
  requests = network.requests;
});
afterEach(() => vi.unstubAllGlobals());

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
    expect(overall.getByRole("spinbutton", { name: "Spend limit (USD)" })).toHaveValue(150);
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
    expect(customEditor.getByRole("spinbutton", { name: "Spend limit (USD)" })).toHaveValue(150);
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
    expect(overall.getByRole("spinbutton", { name: "Spend limit (USD)" })).toHaveValue(250);
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
    const amount = editor.getAllByRole("spinbutton", { name: "Additional spend limit (USD)" })[0];
    expect(amount).toHaveAttribute("placeholder", "No limit");
    expect(amount).toHaveAccessibleDescription(
      "Leave an amount blank to remove that spend limit. Other spend limits and TPM/RPM limits stay unchanged.",
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
    const add = editor.getByRole("button", { name: "Add another spend limit" });
    expect(editor.queryByText("Additional spend limit (USD)")).not.toBeInTheDocument();
    await user.click(add);
    fireEvent.change(editor.getByRole("spinbutton", { name: "Additional spend limit (USD)" }), {
      target: { value: "1300" },
    });
    await user.click(add);
    expect(editor.getByText("Additional spend limit (USD)")).toBeInTheDocument();
    expect(editor.getAllByRole("spinbutton", { name: "Additional spend limit (USD)" })).toHaveLength(2);
    await user.click(editor.getAllByRole("combobox", { name: "Reset period" })[2]);
    await user.click(await screen.findByRole("option", { name: "Custom period" }));
    const duplicatePeriod = editor.getByRole("textbox", { name: "Custom period" });
    fireEvent.change(duplicatePeriod, { target: { value: "30d" } });
    expect(duplicatePeriod).toBeInvalid();
    expect(editor.getByRole("alert")).toHaveTextContent("A spend limit already uses this period.");
    await user.click(screen.getByRole("button", { name: "Save app settings" }));
    expect(requests.filter((request) => request.method === "PATCH")).toHaveLength(0);
    await user.click(editor.getAllByRole("button", { name: "Remove limit" })[1]);
    await user.click(editor.getByRole("button", { name: "Remove limit" }));
    expect(editor.queryByText("Additional spend limit (USD)")).not.toBeInTheDocument();
    await user.click(add);
    fireEvent.change(editor.getByRole("spinbutton", { name: "Additional spend limit (USD)" }), {
      target: { value: "1300" },
    });
    await user.click(add);
    fireEvent.change(editor.getAllByRole("spinbutton", { name: "Additional spend limit (USD)" })[1], {
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
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
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
