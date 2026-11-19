import { chooseAppOption } from "./appTestUtils";
import userEvent from "@testing-library/user-event";
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
const paginatedResources = Array.from({ length: 30 }, (_, index) => ({
  resource_type: "team",
  resource_id: `team-${index}`,
  alias: `app-${String(index).padStart(2, "0")}`,
  app_id: null,
}));
let definition: { resource_type: string | null };
let configuredApps: unknown[];
let resources: unknown[];
let requests: Request[];

beforeEach(() => {
  definition = { resource_type: "team" };
  configuredApps = [];
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
    if (path.endsWith("/customer_organization_budget_groups"))
      return json({ default_budgets: [{ budget_id: "org-standard", name: "Standard", max_budget: 1000 }] });
    if (path.endsWith("/app_customer_organization_budget_groups"))
      return json({
        default_budgets: [{ budget_id: "standard", name: "Tier 1", max_budget: 100 }],
        revision: "a".repeat(64),
      });
    if (path.endsWith("/apps") && request.method === "POST") return json(app);
    if (path.endsWith("/apps")) return json(configuredApps);
    if (path.endsWith("/models")) return json({ data: [] });
    throw new Error(`Unexpected request ${path}`);
  });
  vi.stubGlobal("fetch", fetchMock);
});
afterEach(() => vi.unstubAllGlobals());

async function openAppSettings() {
  const user = userEvent.setup();
  render(<AppsManager />);
  await user.click(await screen.findByRole("button", { name: "chat" }));
  return user;
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

  it("should reuse a shared customer budget group when configuring an existing resource", async () => {
    const user = await openAppSettings();
    await chooseAppOption(user, await screen.findByRole("combobox", { name: "Budget group" }), "Tier 1");
    expect(screen.getByText("$100.00")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Save app settings" }));
    await waitFor(() => expect(requests.some((request) => request.method === "POST")).toBe(true));
    const expectedSettings = {
      resource_id: "team-chat",
      resource_type: "team",
      default_tier_budget_id: "standard",
      default_customer_member_budget: null,
      auto_create_customers: true,
    };
    expect(await requests.find((request) => request.method === "POST")!.json()).toMatchObject(expectedSettings);
    expect(requests.some((request) => new URL(request.url).pathname.includes("/tiers"))).toBe(false);
  });

  it("changes app and member-default rates without persisting unchanged displayed spend caps", async () => {
    resources = [{ ...resource, app_id: app.app_id }];
    const budget = {
      max_budget: 150,
      budget_duration: "1mo",
      rpm_limit: 20,
      budget_limits: [{ max_budget: 1200, budget_duration: "12mo" }],
    };
    configuredApps = [{ ...app, budget, default_customer_member_budget: budget }];
    const user = await openAppSettings();
    const overall = within(await screen.findByRole("group", { name: "Overall app budget" }));
    const member = within(screen.getByRole("group", { name: "Default per-member budget" }));
    expect(overall.getByRole("spinbutton", { name: "Spend limit (USD)" })).toHaveValue(150);
    fireEvent.change(overall.getByRole("spinbutton", { name: "Requests per minute" }), { target: { value: "0" } });
    fireEvent.change(member.getByRole("spinbutton", { name: "Requests per minute" }), { target: { value: "30" } });
    await user.click(screen.getByRole("button", { name: "Save app settings" }));
    await waitFor(() => expect(requests.some((request) => request.method === "PATCH")).toBe(true));
    const body = await requests.find((request) => request.method === "PATCH")!.json();
    expect(body.budget).toEqual({ rpm_limit: 0 });
    expect(body.default_customer_member_budget).toEqual({ rpm_limit: 30 });
  });

  it.each([
    ["Overall app budget", "budget", "clear"],
    ["Overall app budget", "budget", "remove"],
    ["Default per-member budget", "default_customer_member_budget", "clear"],
    ["Default per-member budget", "default_customer_member_budget", "remove"],
  ])("removes only the selected window in %s (%s) using %s", async (title, key, action) => {
    resources = [{ ...resource, app_id: app.app_id }];
    const yearly = { max_budget: 1200, budget_duration: "12mo", reset_at: "2027-01-01T00:00:00Z" };
    const budget = {
      tpm_limit: 1000,
      rpm_limit: 20,
      budget_limits: [{ max_budget: 100, budget_duration: "1mo" }, yearly],
    };
    configuredApps = [{ ...app, budget, default_customer_member_budget: budget }];
    const user = await openAppSettings();
    const editor = within(await screen.findByRole("group", { name: title }));
    const amount = editor.getByRole("spinbutton", { name: "1mo spend limit (USD)" });
    expect(amount).toHaveAttribute("placeholder", "No limit");
    expect(amount).toHaveAccessibleDescription(
      "Leave an amount blank to remove that spend limit. Other spend limits and TPM/RPM limits stay unchanged.",
    );
    if (action === "clear") fireEvent.change(amount, { target: { value: "" } });
    else await user.click(editor.getAllByRole("button", { name: "Remove limit" })[0]);
    await user.click(screen.getByRole("button", { name: "Save app settings" }));
    await waitFor(() => expect(requests.some((request) => request.method === "PATCH")).toBe(true));
    const body = await requests.find((request) => request.method === "PATCH")!.json();
    expect(body[key]).toEqual({ budget_limits: [yearly] });
    expect(body[key === "budget" ? "default_customer_member_budget" : "budget"]).toEqual({});
  });

  it("should send the app's customer creation switch turned off when the admin unchecks it", async () => {
    const user = await openAppSettings();
    await user.click(await screen.findByRole("checkbox", { name: "This app may create new customers" }));
    await user.click(screen.getByRole("button", { name: "Save app settings" }));
    await waitFor(() => expect(requests.some((request) => request.method === "POST")).toBe(true));
    expect(await requests.find((request) => request.method === "POST")!.json()).toMatchObject({
      resource_id: "team-chat",
      auto_create_customers: false,
    });
  });

  it("should send the app's surcharge as a fraction and clear it when the field is emptied", async () => {
    const user = await openAppSettings();
    const field = await screen.findByRole("spinbutton", { name: "Surcharge on customer requests (%)" });
    fireEvent.change(field, { target: { value: "2.5" } });
    await user.click(screen.getByRole("button", { name: "Save app settings" }));
    await waitFor(() => expect(requests.filter((request) => request.method === "POST")).toHaveLength(1));
    expect(await requests.find((request) => request.method === "POST")!.json()).toMatchObject({
      customer_margin: 0.025,
    });
    fireEvent.change(field, { target: { value: "" } });
    await user.click(screen.getByRole("button", { name: "Save app settings" }));
    await waitFor(() => expect(requests.filter((request) => request.method === "POST")).toHaveLength(2));
    expect(await requests.filter((request) => request.method === "POST")[1].json()).toMatchObject({
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
