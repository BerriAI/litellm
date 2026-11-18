import userEvent from "@testing-library/user-event";
import { beforeEach, afterEach, describe, expect, it, vi } from "vitest";
import { registerBaseUrlGetter } from "@/lib/http/runtime";
import {
  renderWithProviders as render,
  screen,
  fireEvent,
  waitFor,
  within,
  chooseSelectOption,
  testQueryClient,
} from "../../../../../tests/test-utils";
import AppsManager from "./AppsManager";
import AppTiers from "./AppTiers";
import type { App } from "./appForms";

vi.mock("@/app/(dashboard)/hooks/models/useModels", () => ({
  useAllProxyModels: () => ({ data: { data: [] }, isLoading: false, isError: false }),
}));
const app: App = {
  app_id: "app-1",
  alias: "Assistant",
  spend: 2,
  tier_budget_ids: ["tier-1"],
  default_tier_budget_id: null,
  created_at: "",
  updated_at: "",
  created_by: "",
  updated_by: "",
  tiers: [{ budget_id: "tier-1", name: "Standard", max_budget: 10, rpm_limit: 2 }],
};
const fetchMock = vi.fn<typeof fetch>();
const json = (value: unknown, status = 200) =>
  new Response(JSON.stringify(value), {
    status,
    headers: { "Content-Type": "application/json" },
  });

beforeEach(() => {
  testQueryClient.clear();
  registerBaseUrlGetter(() => "http://localhost");
  fetchMock.mockReset();
  vi.stubGlobal("fetch", fetchMock);
});
afterEach(() => vi.unstubAllGlobals());

describe("Apps dashboard", () => {
  it("creates an app with an alias and app budget", async () => {
    const requests: Request[] = [];
    fetchMock.mockImplementation(async (input) => {
      const request = input as Request;
      requests.push(request.clone());
      return json(request.method === "GET" ? [] : app);
    });
    render(<AppsManager />);
    expect(await screen.findByText("No apps yet. Create an app to track its customer budgets.")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Create app" }));
    fireEvent.change(screen.getByRole("textbox", { name: "App alias" }), { target: { value: "New app" } });
    const budget = within(screen.getByRole("group", { name: "App budget" }));
    fireEvent.click(budget.getByRole("checkbox", { name: "Set a budget and limits" }));
    fireEvent.change(budget.getByRole("spinbutton", { name: "Spend limit (USD)" }), { target: { value: "25" } });
    fireEvent.click(screen.getByRole("button", { name: "Save app settings" }));
    await waitFor(() => expect(requests.some((request) => request.method === "POST")).toBe(true));
    const request = requests.find((row) => row.method === "POST");
    expect(await request?.json()).toMatchObject({
      alias: "New app",
      budget: { max_budget: 25 },
      default_customer_member_budget: null,
    });
  });

  it("shows load failures instead of an empty list", async () => {
    fetchMock.mockResolvedValue(json({ detail: "Apps unavailable" }, 503));
    render(<AppsManager />);
    expect(await screen.findByRole("alert")).toHaveTextContent("Apps unavailable");
    expect(screen.queryByText("No apps yet. Create an app to track its customer budgets.")).not.toBeInTheDocument();
  });

  it("updates an app's alias and default tier", async () => {
    const requests: Request[] = [];
    fetchMock.mockImplementation(async (input) => {
      const request = input as Request;
      requests.push(request.clone());
      return json(request.method === "GET" ? [app] : app);
    });
    const user = userEvent.setup();
    render(<AppsManager />);
    await user.click(await screen.findByRole("button", { name: "Edit Assistant" }));
    fireEvent.change(screen.getByRole("textbox", { name: "App alias" }), { target: { value: "Renamed" } });
    await chooseSelectOption(user, screen.getByRole("combobox", { name: "Default tier" }), "Standard");
    await user.click(screen.getByRole("button", { name: "Save app settings" }));
    await waitFor(() => expect(requests.some((request) => request.method === "PATCH")).toBe(true));
    const request = requests.find((row) => row.method === "PATCH");
    expect(new URL(request!.url).pathname).toBe("/management/v1/apps/app-1");
    expect(await request?.json()).toMatchObject({ alias: "Renamed", default_tier_budget_id: "tier-1" });
  });

  it("creates an unlimited tier and confirms successful deletion", async () => {
    const requests: Request[] = [];
    fetchMock.mockImplementation(async (input) => {
      const request = input as Request;
      requests.push(request.clone());
      return request.method === "DELETE" ? new Response(null, { status: 204 }) : json(app.tiers?.[0]);
    });
    const onSaved = vi.fn(async () => {});
    const user = userEvent.setup();
    render(<AppTiers app={app} onSaved={onSaved} />);
    await user.click(screen.getByRole("button", { name: "Add tier" }));
    fireEvent.change(screen.getByRole("textbox", { name: "Name" }), { target: { value: "Unlimited" } });
    await user.click(screen.getByRole("button", { name: "Save tier" }));
    await waitFor(() => expect(onSaved).toHaveBeenCalledOnce());
    expect(await requests[0].json()).toMatchObject({ name: "Unlimited", max_budget: null, rpm_limit: null });
    await user.click(screen.getByRole("button", { name: "Delete" }));
    await user.click(screen.getByRole("button", { name: "Confirm delete" }));
    await waitFor(() => expect(onSaved).toHaveBeenCalledTimes(2));
    expect(requests[1].method).toBe("DELETE");
    expect(await requests[1].json()).toEqual({ budget_id: "tier-1" });
  });

  it("edits tier limits and keeps deletion conflicts visible", async () => {
    const requests: Request[] = [];
    fetchMock.mockImplementation(async (input) => {
      const request = input as Request;
      requests.push(request.clone());
      return request.method === "DELETE" ? json({ detail: "Tier is in use" }, 409) : json(app.tiers?.[0]);
    });
    const onSaved = vi.fn(async () => {});
    const user = userEvent.setup();
    render(<AppTiers app={app} onSaved={onSaved} />);
    await user.click(screen.getByRole("button", { name: "Edit" }));
    fireEvent.change(screen.getByRole("spinbutton", { name: "Spend limit (USD)" }), { target: { value: "50" } });
    await user.click(screen.getByRole("button", { name: "Save tier" }));
    await waitFor(() => expect(onSaved).toHaveBeenCalledOnce());
    expect(await requests[0].json()).toMatchObject({
      budget_id: "tier-1",
      name: "Standard",
      max_budget: 50,
      rpm_limit: 2,
    });
    await user.click(screen.getByRole("button", { name: "Delete" }));
    await user.click(screen.getByRole("button", { name: "Confirm delete" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Tier is in use");
    expect(screen.getByText("Standard", { selector: "td" })).toBeInTheDocument();
  });
});
