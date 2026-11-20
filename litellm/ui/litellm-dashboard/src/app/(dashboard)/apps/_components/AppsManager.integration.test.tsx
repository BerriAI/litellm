import { chooseAppOption } from "./appTestUtils";
import userEvent, { type UserEvent } from "@testing-library/user-event";
import { beforeEach, afterEach, describe, expect, it, vi } from "vitest";
import { registerBaseUrlGetter } from "@/lib/http/runtime";
import {
  jsonResponse as json,
  fireEvent,
  renderWithProviders as render,
  screen,
  waitFor,
  within,
  testQueryClient,
} from "../../../../../tests/test-utils";
import AppsManager from "./AppsManager";
import type { App } from "./appForms";
import type { BudgetGroups } from "../../customers/_components/customerBudgetGroupsPlan";

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => ({ accessToken: "test-session", userRole: "Admin", isLoading: false }),
}));
vi.mock("@/app/(dashboard)/hooks/models/useModels", () => ({
  useAllProxyModels: () => ({ data: { data: [] }, isLoading: false, isError: false }),
}));
const fetchMock = vi.fn<typeof fetch>();
const resource = { resource_type: "team", resource_id: "team-chat", alias: "chat", app_id: null };
const app = {
  ...resource,
  app_id: "app-chat",
  spend: 0,
  created_at: "",
  updated_at: "",
  created_by: "",
  updated_by: "",
};
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
const paginatedResources = Array.from({ length: 30 }, (_, index) => ({
  resource_type: "team",
  resource_id: `team-${index}`,
  alias: `app-${String(index).padStart(2, "0")}`,
  app_id: null,
}));
let definition: { resource_type: string | null };
let configuredApps: unknown[];
let overallPresets: BudgetGroups["default_budgets"];
let resources: unknown[];
let requests: Request[];

beforeEach(() => {
  definition = { resource_type: "team" };
  configuredApps = [];
  overallPresets = [overallPreset];
  resources = [resource];
  requests = [];
  testQueryClient.clear();
  registerBaseUrlGetter(() => "http://localhost");
  fetchMock.mockReset();
  fetchMock.mockImplementation(async (input) => {
    const request = input as Request;
    requests.push(request.clone());
    const path = new URL(request.url).pathname;
    if (path.endsWith("/apps/definition")) {
      if (request.method === "PUT") definition = await request.clone().json();
      return json(definition);
    }
    if (path.endsWith("/apps/resources")) return json(resources);
    if (path.endsWith("/apps/app-chat")) return json(configuredApps[0]);
    if (path.endsWith("/app_budget_groups"))
      return json({ default_budgets: overallPresets, budget_groups_revision: "a".repeat(64) });
    if (path.endsWith("/apps") && request.method === "POST") return json(app);
    if (path.endsWith("/apps")) return json(configuredApps);
    if (path.endsWith("/models")) return json({ data: [] });
    throw new Error(`Unexpected request ${path}`);
  });
  vi.stubGlobal("fetch", fetchMock);
});
afterEach(() => vi.unstubAllGlobals());

async function openAppSettings(settings?: Partial<App>) {
  if (settings) {
    resources = [{ ...resource, app_id: app.app_id }];
    configuredApps = [{ ...app, ...settings }];
  }
  const user = userEvent.setup();
  render(<AppsManager />);
  await user.click(await screen.findByRole("button", { name: "chat" }));
  return user;
}

async function saveAppSettings(user: UserEvent, method: "PATCH" | "POST" = "PATCH", index = 0) {
  await user.click(screen.getByRole("button", { name: "Save app settings" }));
  await waitFor(() => expect(requests.filter((request) => request.method === method)).toHaveLength(index + 1));
  return requests
    .filter((request) => request.method === method)
    [index].clone()
    .json();
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
    "saves app settings via %s without changing the relocated organization default",
    async (method) => {
      const user = await openAppSettings(method === "PATCH" ? { default_tier_budget_id: "standard" } : undefined);
      const form = within(await screen.findByRole("form", { name: "Defaults for chat" }));
      expect(form.queryByRole("combobox", { name: "Budget group" })).not.toBeInTheDocument();
      const body = await saveAppSettings(user, method);
      expect(body).not.toHaveProperty("default_tier_budget_id");
      expect(body).toMatchObject({
        ...(method === "POST" ? { resource_id: "team-chat", resource_type: "team" } : {}),
        default_customer_member_budget: null,
        auto_create_customers: true,
      });
      expect(requests.some((request) => new URL(request.url).pathname.includes("/tiers"))).toBe(false);
    },
  );

  it("changes app and member-default rates without persisting unchanged displayed spend caps", async () => {
    const user = await openAppSettings({ budget: overallPreset, default_customer_member_budget: overallPreset });
    const overall = within(await screen.findByRole("region", { name: "Overall app budget" }));
    const member = within(screen.getByRole("region", { name: "Default per-member budget" }));
    expect(overall.getByRole("spinbutton", { name: "Spend limit (USD)" })).toHaveValue(150);
    fireEvent.change(overall.getByRole("spinbutton", { name: "Requests per minute" }), { target: { value: "0" } });
    fireEvent.change(member.getByRole("spinbutton", { name: "Requests per minute" }), { target: { value: "30" } });
    const body = await saveAppSettings(user);
    expect(body.budget).toEqual({ rpm_limit: 0 });
    expect(body.default_customer_member_budget).toEqual({ rpm_limit: 30 });
  });

  it("keeps overall presets linked until a custom edit and excludes presets from member defaults", async () => {
    const user = await openAppSettings({
      budget_id: overallPreset.budget_id,
      budget: { ...overallPreset, max_budget: 250 },
    });
    const overall = within(await screen.findByRole("region", { name: "Overall app budget" }));
    await user.click(screen.getByRole("combobox", { name: "Default per-member budget" }));
    expect(await screen.findAllByRole("option")).toHaveLength(2);
    expect(screen.getByRole("option", { name: "Enter custom limits instead" })).toBeInTheDocument();
    await user.keyboard("{Escape}");
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
    await chooseAppOption(user, source, "Enter custom limits instead");
    expect(overall.getByRole("spinbutton", { name: "Spend limit (USD)" })).toHaveValue(150);
    fireEvent.change(overall.getByRole("spinbutton", { name: "Requests per minute" }), { target: { value: "0" } });
    const custom = await saveAppSettings(user, "PATCH", 1);
    expect(custom.budget).toEqual({ max_budget: 150, budget_duration: "1mo", rpm_limit: 0 });
    expect(custom).not.toHaveProperty("budget_id");
    expect(custom.default_customer_member_budget).toBeNull();
  });

  it("clears an overall preset with No limit without changing member defaults", async () => {
    const user = await openAppSettings({ budget_id: overallPreset.budget_id, budget: overallPreset });
    await chooseAppOption(user, screen.getByRole("combobox", { name: "Overall app budget" }), /^No limit/);
    const body = await saveAppSettings(user);
    expect(body.budget_id).toBeNull();
    expect(body).not.toHaveProperty("budget");
    expect(body.default_customer_member_budget).toBeNull();
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
    const expected = {
      budget: {
        ...copiedPolicy,
        budget_duration: "1mo",
        soft_budget: null,
        rpm_limit: 0,
        model_rpm_limit: null,
      },
      default_customer_member_budget: null,
      auto_create_customers: true,
      customer_margin: null,
    };
    expect(await saveAppSettings(user)).toEqual(expected);
  });

  it.each([
    ["Overall app budget", "budget"],
    ["Default per-member budget", "default_customer_member_budget"],
  ])("recalculates the reset when changing a stored spend period in %s", async (title, key) => {
    const budget = { rpm_limit: 20, budget_limits: [yearly] };
    const user = await openAppSettings({ budget, default_customer_member_budget: budget });
    const editor = within(await screen.findByRole("region", { name: title }));
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
    await user.click(screen.getByRole("option", { name: "/ month", exact: true }));
    const body = await saveAppSettings(user);
    expect(body[key]).toEqual({ budget_limits: [{ ...yearly, budget_duration: "1mo", reset_at: null }] });
    expect(body[key === "budget" ? "default_customer_member_budget" : "budget"]).toEqual({});
  });

  it.each([
    ["Overall app budget", "budget", "clear"],
    ["Overall app budget", "budget", "remove"],
    ["Default per-member budget", "default_customer_member_budget", "clear"],
    ["Default per-member budget", "default_customer_member_budget", "remove"],
  ])("removes only the selected window in %s (%s) using %s", async (title, key, action) => {
    const budget = {
      tpm_limit: 1000,
      rpm_limit: 20,
      budget_limits: [{ max_budget: 100, budget_duration: "1mo" }, yearly],
    };
    const user = await openAppSettings({ budget, default_customer_member_budget: budget });
    const editor = within(await screen.findByRole("region", { name: title }));
    await user.click(editor.getAllByRole("combobox", { name: "Reset period" })[1]);
    expect(await screen.findByRole("option", { name: "/ year", exact: true })).toHaveAttribute("aria-disabled", "true");
    await user.keyboard("{Escape}");
    const amount = editor.getAllByRole("spinbutton", { name: "Additional spend limit (USD)" })[0];
    expect(amount).toHaveAttribute("placeholder", "No limit");
    expect(amount).toHaveAccessibleDescription(
      "Leave an amount blank to remove that spend limit. Other spend limits and TPM/RPM limits stay unchanged.",
    );
    if (action === "clear") {
      fireEvent.change(amount, { target: { value: "" } });
      await user.click(editor.getAllByRole("combobox", { name: "Reset period" })[2]);
      expect(await screen.findByRole("option", { name: "/ month", exact: true })).toHaveAttribute(
        "aria-disabled",
        "true",
      );
      await user.keyboard("{Escape}");
    } else await user.click(editor.getAllByRole("button", { name: "Remove limit" })[0]);
    const body = await saveAppSettings(user);
    expect(body[key]).toEqual({ budget_limits: [yearly] });
    expect(body[key === "budget" ? "default_customer_member_budget" : "budget"]).toEqual({});
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
    fireEvent.change(field, { target: { value: "" } });
    expect(await saveAppSettings(user, "POST", 1)).toMatchObject({
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

  it("should list every application with its surcharge, customer creation and configured state", async () => {
    resources = [
      { ...resource, app_id: "app-chat" },
      { resource_type: "team", resource_id: "team-search", alias: "search", app_id: null },
    ];
    configuredApps = [{ ...app, spend: 12.5, customer_margin: 0.05, auto_create_customers: false }];
    render(<AppsManager />);
    await screen.findByRole("button", { name: "chat" });
    const table = screen.getByRole("table", { name: "Applications" });
    const [chat, search] = within(table).getAllByRole("row").slice(1);
    expect(chat).toHaveTextContent("chat");
    expect(chat).toHaveTextContent("$12.50");
    expect(chat).toHaveTextContent("5%");
    expect(chat).toHaveTextContent("Blocked");
    expect(chat).toHaveTextContent("Configured");
    expect(search).toHaveTextContent("search");
    expect(search).toHaveTextContent("Gateway default");
    expect(search).toHaveTextContent("Allowed");
    expect(search).toHaveTextContent("Not configured");
  });

  it("should show the current budget window's spend and cap instead of lifetime spend", async () => {
    resources = [{ ...resource, app_id: "app-chat" }];
    configuredApps = [
      {
        ...app,
        spend: 900,
        budget: { budget_limits: [{ budget_duration: "1mo", max_budget: 100 }] },
        budget_usage: [{ budget_duration: "1mo", max_budget: 100, spend: 40, reset_at: null }],
      },
    ];
    render(<AppsManager />);
    await screen.findByRole("button", { name: "chat" });
    const usage = within(screen.getByRole("table", { name: "Applications" })).getByRole("progressbar");
    expect(usage).toHaveAttribute("aria-valuenow", "40");
    expect(usage).toHaveAttribute("aria-valuetext", expect.stringContaining("$100"));
  });

  it("should show one page of applications at a time and move to the next page", async () => {
    const user = userEvent.setup();
    resources = paginatedResources;
    render(<AppsManager />);
    await screen.findByRole("button", { name: "app-00" });
    expect(screen.queryByRole("button", { name: "app-25" })).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Go to next page" }));
    expect(await screen.findByRole("button", { name: "app-25" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "app-00" })).not.toBeInTheDocument();
  });

  it("should open the second page of applications from a link that says page 2", async () => {
    resources = paginatedResources;
    render(<AppsManager />, { searchParams: "?page=2" });
    expect(await screen.findByRole("button", { name: "app-25" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "app-00" })).not.toBeInTheDocument();
  });

  it("should open a linked application without showing the list first", async () => {
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
      expect(await screen.findByRole("button", { name: "chat" })).toBeInTheDocument();
    },
  );

  it("should narrow the list with the search box and say when nothing matches", async () => {
    render(<AppsManager />);
    await screen.findByRole("button", { name: "chat" });
    fireEvent.change(screen.getByRole("textbox", { name: "Search applications" }), { target: { value: "billing" } });
    expect(screen.queryByRole("button", { name: "chat" })).not.toBeInTheDocument();
    expect(screen.getByText("No applications match your search.")).toBeInTheDocument();
  });

  it("should return from an application's settings to the list", async () => {
    const user = await openAppSettings();
    expect(await screen.findByRole("form", { name: "Defaults for chat" })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Back to applications" }));
    expect(await screen.findByRole("table", { name: "Applications" })).toBeInTheDocument();
  });

  it("should show definition failures without presenting an empty resource list", async () => {
    definition = { resource_type: null };
    fetchMock.mockResolvedValue(json({ detail: "Definition unavailable" }, 503));
    render(<AppsManager />);
    expect(await screen.findByRole("alert")).toHaveTextContent("Definition unavailable");
    expect(screen.queryByRole("combobox")).not.toBeInTheDocument();
  });
});
