import { chooseAppOption } from "./appTestUtils";
import userEvent from "@testing-library/user-event";
import { beforeEach, afterEach, describe, expect, it, vi } from "vitest";
import { registerBaseUrlGetter } from "@/lib/http/runtime";
import { uiHref } from "@/utils/uiHref";
import {
  renderWithProviders as render,
  screen,
  fireEvent,
  act,
  waitFor,
  within,
  testQueryClient,
} from "../../../../../tests/test-utils";
import CustomerOrganizationApps from "./CustomerOrganizationApps";

vi.mock("@/app/(dashboard)/hooks/models/useModels", () => ({
  useAllProxyModels: () => ({ data: { data: [] }, isLoading: false, isError: false }),
}));
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
const json = (value: unknown, status = 200) =>
  new Response(JSON.stringify(value), { status, headers: { "Content-Type": "application/json" } });
let configured: boolean;
let bound: boolean;
let assigned: boolean;
let requests: Request[];
let associationStatus: number;

beforeEach(() => {
  configured = true;
  bound = false;
  assigned = false;
  associationStatus = 404;
  requests = [];
  testQueryClient.clear();
  registerBaseUrlGetter(() => "http://localhost");
  fetchMock.mockReset();
  fetchMock.mockImplementation(async (input) => {
    const request = input as Request;
    requests.push(request.clone());
    const path = new URL(request.url).pathname;
    if (path.endsWith("/apps/definition")) return json({ resource_type: configured ? "team" : null });
    if (path.endsWith("/apps/resources"))
      return json([
        { ...resource, app_id: bound ? app.app_id : null },
        { ...resource, resource_id: "team-search", alias: "search", app_id: null },
      ]);
    if (path.endsWith("/apps")) {
      if (request.method === "POST") {
        bound = true;
        return json(app);
      }
      return json(bound ? [app] : []);
    }
    if (path.endsWith("/customer_organization_budget_groups"))
      return json({
        default_budgets: [{ budget_id: "standard", name: "Standard", max_budget: 100 }],
        revision: "a".repeat(64),
      });
    if (request.method === "PUT") {
      assigned = true;
      return json({ app_id: app.app_id, customer_org_id: "org-1", spend: 0 });
    }
    if (path.endsWith("/customer_organizations/org-1"))
      return assigned ? json(assignedBudget) : json({ detail: "Association unavailable" }, associationStatus);
    if (path.endsWith("/models")) return json({ data: [] });
    throw new Error(`Unexpected request ${path}`);
  });
  vi.stubGlobal("fetch", fetchMock);
});
afterEach(() => vi.unstubAllGlobals());

describe("customer app budgets", () => {
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
    expect(table.getByText("$100.00")).toBeInTheDocument();
    expect(table.queryByText("$1,000.00")).not.toBeInTheDocument();
    expect(table.getByRole("button", { name: "Budget details" })).toBeInTheDocument();
    expect(table.queryByText("No spend limit")).not.toBeInTheDocument();
  });
  it("should route Add app budget to Gateway setup when the app definition is unset", async () => {
    configured = false;
    render(<CustomerOrganizationApps organizationId="org-1" />);
    expect(await screen.findByText(/Define what an app is in your Gateway/)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Add app budget" })).toHaveAttribute("href", uiHref("apps"));
    expect(requests.every((request) => request.method === "GET")).toBe(true);
    expect(screen.queryByRole("combobox")).not.toBeInTheDocument();
  });

  it("should select an existing resource and assign a shared profile with a customer-specific member default", async () => {
    const user = userEvent.setup();
    render(<CustomerOrganizationApps organizationId="org-1" organizationAlias="Customer Y" />);
    await user.click(await screen.findByRole("button", { name: "Add app budget" }));
    const dialog = within(await screen.findByRole("dialog", { name: "Add app budget" }));
    await chooseAppOption(user, dialog.getByRole("combobox", { name: "Select an app" }), "chat");
    expect(dialog.getByRole("textbox", { name: "Customer organization" })).toHaveValue("Customer Y");
    await chooseAppOption(user, screen.getByRole("combobox", { name: "Budget source" }), "Budget group");
    await chooseAppOption(user, await screen.findByRole("combobox", { name: "Budget group" }), "Standard");
    const member = within(screen.getByRole("group", { name: "Per-member default for this customer" }));
    act(() => {
      fireEvent.click(member.getByRole("checkbox", { name: "Set custom limits" }));
    });
    act(() => {
      fireEvent.change(member.getByRole("spinbutton", { name: "Requests per minute" }), { target: { value: "60" } });
    });
    await user.click(screen.getByRole("button", { name: "Save app budget" }));
    await waitFor(() => expect(requests.some((request) => request.method === "PUT")).toBe(true));
    const mutations = requests.filter((request) => request.method !== "GET");
    expect(mutations.map((request) => request.method)).toEqual(["POST", "PUT"]);
    expect(await mutations[0].json()).toEqual({ resource_id: "team-chat", resource_type: "team" });
    expect(new URL(mutations[1].url).pathname).toBe("/management/v1/apps/app-chat/customer_organizations/org-1");
    expect(await mutations[1].json()).toMatchObject({
      tier_budget_id: "standard",
      budget_override: null,
      customer_member_budget: { rpm_limit: 60 },
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
    const dialog = within(await screen.findByRole("dialog", { name: "Budget for chat" }));
    expect(dialog.getByRole("textbox", { name: "Customer organization" })).toHaveValue("Customer Y");
    expect(dialog.getByRole("textbox", { name: "App", exact: true })).toHaveValue("chat");
    expect(dialog.getByRole("combobox", { name: "Budget source" })).toHaveValue("Budget group");
    await user.click(dialog.getByRole("button", { name: "Cancel" }));
    await user.click(screen.getByRole("button", { name: "Add app budget" }));
    const fresh = within(await screen.findByRole("dialog", { name: "Add app budget" }));
    expect(fresh.getByRole("combobox", { name: "Select an app" })).toHaveValue("");
    expect(fresh.queryByRole("textbox", { name: "Customer organization" })).not.toBeInTheDocument();
    expect(requests.every((request) => request.method === "GET")).toBe(true);
  });

  it("should save a custom budget without creating a tier or changing the global customer budget", async () => {
    const user = userEvent.setup();
    render(<CustomerOrganizationApps organizationId="org-1" />);
    await user.click(await screen.findByRole("button", { name: "Add app budget" }));
    await chooseAppOption(user, await screen.findByRole("combobox", { name: "Select an app" }), "search");
    await chooseAppOption(user, screen.getByRole("combobox", { name: "Budget source" }), "Custom budget");
    act(() => {
      fireEvent.change(screen.getByRole("spinbutton", { name: "Spend limit (USD)" }), { target: { value: "100" } });
    });
    act(() => {
      fireEvent.change(screen.getByRole("textbox", { name: "Reset period" }), { target: { value: "1mo" } });
    });
    await user.click(screen.getByRole("button", { name: "Save app budget" }));
    await waitFor(() => expect(requests.some((request) => request.method === "PUT")).toBe(true));
    expect(await requests.find((request) => request.method === "PUT")!.json()).toMatchObject({
      tier_budget_id: null,
      budget_override: { max_budget: 100, budget_duration: "1mo" },
    });
    expect(requests.some((request) => request.method === "PATCH")).toBe(false);
  });

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
});
