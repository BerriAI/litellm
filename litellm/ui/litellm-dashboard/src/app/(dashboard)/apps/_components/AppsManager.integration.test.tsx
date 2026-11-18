import { chooseAppOption } from "./appTestUtils";
import userEvent from "@testing-library/user-event";
import { beforeEach, afterEach, describe, expect, it, vi } from "vitest";
import { registerBaseUrlGetter } from "@/lib/http/runtime";
import {
  fireEvent,
  renderWithProviders as render,
  screen,
  waitFor,
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
const json = (value: unknown, status = 200) =>
  new Response(JSON.stringify(value), { status, headers: { "Content-Type": "application/json" } });
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
let definition: { resource_type: string | null };
let requests: Request[];

beforeEach(() => {
  definition = { resource_type: null };
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
    if (path.endsWith("/apps/resources")) return json([resource]);
    if (path.endsWith("/customer_organization_budget_groups"))
      return json({
        default_budgets: [{ budget_id: "standard", name: "Standard", max_budget: 100 }],
        revision: "a".repeat(64),
      });
    if (path.endsWith("/apps") && request.method === "POST") return json(app);
    if (path.endsWith("/models")) return json({ data: [] });
    throw new Error(`Unexpected request ${path}`);
  });
  vi.stubGlobal("fetch", fetchMock);
});
afterEach(() => vi.unstubAllGlobals());

describe("Gateway app definition", () => {
  it("should render an unset choice without creating apps or selecting a resource layer", async () => {
    render(<AppsManager />);
    expect(await screen.findByRole("combobox", { name: "Apps are represented by" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Save app definition" })).toBeDisabled();
    expect(screen.queryByRole("button", { name: "Create app" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Add tier" })).not.toBeInTheDocument();
    expect(requests.every((request) => request.method === "GET")).toBe(true);
    expect(requests.some((request) => new URL(request.url).pathname.endsWith("/apps/resources"))).toBe(false);
  });

  it("should preview existing resources and save only the chosen Gateway resource type", async () => {
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
    definition = { resource_type: "team" };
    const user = userEvent.setup();
    render(<AppsManager />);
    await chooseAppOption(user, await screen.findByRole("combobox", { name: "Select an app" }), "chat");
    await chooseAppOption(user, await screen.findByRole("combobox", { name: "Budget group" }), "Standard");
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

  it("should send the app's customer creation switch turned off when the admin unchecks it", async () => {
    const user = userEvent.setup();
    definition = { resource_type: "team" };
    render(<AppsManager />);
    await chooseAppOption(user, await screen.findByRole("combobox", { name: "Select an app" }), "chat");
    await user.click(await screen.findByRole("checkbox", { name: "This app may create new customers" }));
    await user.click(screen.getByRole("button", { name: "Save app settings" }));
    await waitFor(() => expect(requests.some((request) => request.method === "POST")).toBe(true));
    expect(await requests.find((request) => request.method === "POST")!.json()).toMatchObject({
      resource_id: "team-chat",
      auto_create_customers: false,
    });
  });

  it("should send the app's surcharge as a fraction and clear it when the field is emptied", async () => {
    const user = userEvent.setup();
    definition = { resource_type: "team" };
    render(<AppsManager />);
    await chooseAppOption(user, await screen.findByRole("combobox", { name: "Select an app" }), "chat");
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
    const user = userEvent.setup();
    definition = { resource_type: "team" };
    render(<AppsManager />);
    await chooseAppOption(user, await screen.findByRole("combobox", { name: "Select an app" }), "chat");
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
    definition = { resource_type: "team" };
    render(<AppsManager />);
    await user.click(await screen.findByRole("checkbox", { name: "Requests with no app may create new customers" }));
    await waitFor(() => expect(requests.some((request) => request.method === "PUT")).toBe(true));
    expect(await requests.find((request) => request.method === "PUT")!.json()).toEqual({
      resource_type: "team",
      auto_create_customers_without_app: false,
    });
  });

  it("should show definition failures without presenting an empty resource list", async () => {
    fetchMock.mockResolvedValue(json({ detail: "Definition unavailable" }, 503));
    render(<AppsManager />);
    expect(await screen.findByRole("alert")).toHaveTextContent("Definition unavailable");
    expect(screen.queryByRole("combobox")).not.toBeInTheDocument();
  });
});
