import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { toast } from "sonner";
import { registerBaseUrlGetter } from "@/lib/http/runtime";
import type { components } from "@/lib/http/schema";
import {
  act,
  fireEvent,
  chooseSelectOption,
  jsonResponse as json,
  renderWithProviders as render,
  screen,
  testQueryClient,
  waitFor,
  within,
} from "../../../../../tests/test-utils";
import AppBudgetDefaults from "./AppBudgetDefaults";
import AppBudgetGroupsPage from "../budget-groups/page";
import type { App, Budget } from "./appForms";

vi.mock("sonner", () => ({ toast: { success: vi.fn() } }));
const auth = vi.hoisted(() => ({ accessToken: "test-session", userId: "user-1", userRole: "Admin" }));
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({ default: () => auth }));
const resources = ["Chat", "Support", "Search"].map((alias) => ({
  resource_type: "team",
  resource_id: alias.toLowerCase(),
  alias,
}));
type AppFixture = (typeof resources)[number] &
  Pick<
    App,
    "app_id" | "default_tier_budget_id" | "default_customer_member_budget_id" | "default_customer_member_budget"
  >;
const writes: Request[] = [];
const catalog = (groups: Record<string, string>) =>
  json({ default_budgets: Object.entries(groups).map(([budget_id, name]) => ({ budget_id, name })) });
let apps: AppFixture[];
let staleApps: AppFixture[] | null;
let rejectedIds: Set<string>;

beforeEach(() => {
  auth.userRole = "Admin";
  apps = resources.slice(0, 2).map((resource, index) => ({
    ...resource,
    app_id: `app-${resource.resource_id}`,
    default_tier_budget_id: `tier-${index + 1}`,
    default_customer_member_budget_id: `member-${index + 1}`,
  }));
  staleApps = null;
  rejectedIds = new Set(["chat"]);
  writes.length = 0;
  vi.mocked(toast.success).mockClear();
  testQueryClient.clear();
  registerBaseUrlGetter(() => "http://localhost");
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input, init) => {
      const request = new Request(input, init);
      const path = new URL(request.url).pathname;
      if (request.method !== "GET") {
        writes.push(request.clone());
        const body: components["schemas"]["AppPatch"] & { resource_id?: string } = await request.json();
        const id = body.resource_id ?? apps.find((app) => path.endsWith(`/${app.app_id}`))?.resource_id;
        if (!id) throw new Error(`Unknown app ${path}`);
        if (rejectedIds.has(id)) return json({ detail: "Could not save the default group." }, 503);
        const previous = apps.find((app) => app.resource_id === id);
        const app = {
          ...previous,
          ...resources.find((resource) => resource.resource_id === id)!,
          app_id: `app-${id}`,
          ...body,
          ...(body.default_customer_member_budget && {
            default_customer_member_budget_id: `private-${id}`,
            default_customer_member_budget: {
              ...previous?.default_customer_member_budget,
              ...body.default_customer_member_budget,
            },
          }),
          ...(body.default_customer_member_budget_id === null && { default_customer_member_budget: null }),
        };
        apps = [...apps.filter((existing) => existing.app_id !== app.app_id), app];
        return json(app);
      }
      const listed = staleApps ?? apps;
      if (path.endsWith("/apps/definition")) return json({ resource_type: "team" });
      if (path.endsWith("/apps/resources"))
        return json(
          resources.map((resource) => ({
            ...resource,
            app_id: listed.find((app) => app.resource_id === resource.resource_id)?.app_id ?? null,
          })),
        );
      if (path.endsWith("/apps")) return json(listed);
      if (path.endsWith("/apps/app-chat")) return json(listed.find((app) => app.app_id === "app-chat"));
      if (path === "/models") return json({ data: [{ id: "model-a" }] });
      if (path.endsWith("/models")) return json(["model-a"]);
      if (path.endsWith("/app_customer_member_budget_groups"))
        return catalog({ "member-1": "Member 1", "member-2": "Member 2" });
      if (path.endsWith("/app_customer_organization_budget_groups"))
        return catalog({ "tier-1": "Tier 1", "tier-2": "Tier 2" });
      if (path.endsWith("/customer_organization_budget_groups"))
        return catalog({ "org-standard": "Standard", "org-enterprise": "Enterprise" });
      throw new Error(`Unexpected request ${request.method} ${path}`);
    }),
  );
});
afterEach(() => vi.unstubAllGlobals());

it("changes only the current app's member default from its Budget groups page without loading other apps", async () => {
  rejectedIds.clear();
  const user = userEvent.setup();
  render(<AppBudgetGroupsPage />, { searchParams: "?app_id=app-chat" });
  const chat = within(await screen.findByRole("group", { name: "Defaults for Chat" }));
  const select = chat.getByRole("combobox", { name: "Budget group" });
  await waitFor(() => expect(select).toBeEnabled());
  await chooseSelectOption(user, select, /^Member 2/);
  const defaults = within(screen.getByRole("region", { name: "Default per-member budget" }));
  await user.click(defaults.getByRole("button", { name: "Save changes" }));
  await waitFor(() => expect(toast.success).toHaveBeenCalledWith("App settings saved."));
  await expectWrites([["PATCH", "chat", "member-2"]], "default_customer_member_budget_id");
  expect(screen.queryByRole("group", { name: "Defaults for Support" })).not.toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Add app default" })).not.toBeInTheDocument();
  const paths = vi
    .mocked(fetch)
    .mock.calls.map(([input]) => new URL(input instanceof Request ? input.url : input).pathname);
  expect(paths).toEqual(
    expect.arrayContaining(["/management/v1/apps/app-chat", "/management/v1/app_customer_member_budget_groups"]),
  );
  expect(paths.every((path) => !path.endsWith("/apps/resources"))).toBe(true);
});

it("edits shared app-member templates from the app page without changing the app's assignment", async () => {
  const fallback = vi.mocked(fetch).getMockImplementation()!;
  vi.mocked(fetch).mockImplementation(async (input, init) => {
    const request = new Request(input, init);
    if (request.method !== "PATCH") return fallback(request);
    writes.push(request.clone());
    return catalog({ "member-1": "Member 1", "member-2": "Member 2" });
  });
  const user = userEvent.setup();
  render(<AppBudgetGroupsPage />, { searchParams: "?app_id=app-chat" });
  const row = within(await screen.findByRole("row", { name: "Member 1" }));
  expect(screen.getByText(/These templates are shared across apps/)).toBeInTheDocument();
  fireEvent.change(row.getByRole("spinbutton", { name: "Tokens per minute" }), { target: { value: "500" } });
  const templates = within(screen.getByRole("form", { name: "Budget groups" }));
  await user.click(templates.getByRole("button", { name: "Save changes" }));
  await waitFor(() => expect(writes).toHaveLength(1));
  expect(new URL(writes[0].url).pathname).toBe("/management/v1/app_customer_member_budget_groups");
  expect(await writes[0].json()).toMatchObject({
    default_budgets: expect.arrayContaining([expect.objectContaining({ budget_id: "member-1", tpm_limit: 500 })]),
  });
});

it.each(["Internal User", "Admin Viewer"])("keeps app budget groups inaccessible to %s", async (role) => {
  auth.userRole = role;
  render(<AppBudgetGroupsPage />, { searchParams: "?app_id=app-chat" });
  expect(await screen.findByRole("alert")).toHaveTextContent("Only proxy admins");
  expect(fetch).not.toHaveBeenCalled();
});

async function expectWrites(
  expected: ["POST" | "PATCH", string, string | null][],
  field: "default_tier_budget_id" | "default_customer_member_budget_id" = "default_tier_budget_id",
) {
  expect(writes).toHaveLength(expected.length);
  for (const [index, [method, resource, default_tier_budget_id]] of expected.entries()) {
    const request = writes[index];
    expect(request.method).toBe(method);
    expect(new URL(request.url).pathname).toBe(`/management/v1/apps${method === "PATCH" ? `/app-${resource}` : ""}`);
    expect(await request.json()).toEqual({
      [field]: default_tier_budget_id,
      ...(method === "POST" ? { resource_id: resource, resource_type: "team" } : {}),
    });
  }
}

it.each([
  ["organization", "PATCH", "Tier 2", "tier-2", "tier-1"],
  ["organization", "PATCH", "Tier 2", "tier-2", "org-standard"],
  ["organization", "PATCH", "No limit", null, "org-standard"],
  ["organization", "POST", "Tier 2", "tier-2", null],
  ["member", "PATCH", "Member 2", "member-2", "member-1"],
  ["member", "PATCH", "No limit", null, "member-1"],
  ["member", "POST", "Member 2", "member-2", null],
] as const)(
  "saves %s %s default %s from %s/%s without changing another app",
  async (...[scope, method, label, id, initial]) => {
    const field = scope === "member" ? "default_customer_member_budget_id" : "default_tier_budget_id";
    apps = apps.flatMap((app) => {
      if (app.resource_id !== "chat") return [app];
      return method === "POST" ? [] : [{ ...app, [field]: initial }];
    });
    const user = userEvent.setup();
    render(<AppBudgetDefaults scope={scope} />);
    const support = within(await screen.findByRole("group", { name: "Defaults for Support" }));
    await waitFor(() =>
      expect(support.getByRole("combobox", { name: "Budget group" })).toHaveTextContent(
        scope === "member" ? "Member 2" : "Tier 2",
      ),
    );
    if (method === "POST") {
      await user.click(screen.getByRole("button", { name: "Add app default" }));
      await chooseSelectOption(user, screen.getByRole("combobox", { name: "Select an app" }), "Chat");
    }
    const chat = within(await screen.findByRole("group", { name: "Defaults for Chat" }));
    const group = chat.getByRole("combobox", { name: "Budget group" });
    if (scope === "member") {
      await user.click(group);
      expect(await screen.findByRole("option", { name: /^Member 1/ })).toBeInTheDocument();
      expect(screen.queryByRole("option", { name: /^Tier 1|^Standard/ })).not.toBeInTheDocument();
      await user.keyboard("{Escape}");
    }
    if (initial === "org-standard") {
      expect(group).toHaveTextContent("Standard (overall organization group)");
      await user.click(group);
      expect(await screen.findByRole("option", { name: /^Standard/ })).toHaveAttribute("aria-disabled", "true");
      expect(screen.queryByRole("option", { name: /^Enterprise/ })).not.toBeInTheDocument();
      await user.keyboard("{Escape}");
    }
    expect(writes).toHaveLength(0);
    await chooseSelectOption(user, group, new RegExp(`^${label}`));
    apps = apps.map((app) => (app.resource_id === "support" ? { ...app, [field]: null } : app));
    await act(async () => {
      await testQueryClient.invalidateQueries({ queryKey: ["apps"] });
    });
    expect(group).toHaveTextContent(label);
    await waitFor(() => expect(support.getByRole("combobox", { name: "Budget group" })).toHaveTextContent("No limit"));
    const save = screen.getByRole("button", { name: "Save changes" });
    await user.click(save);
    expect(await chat.findByRole("alert")).toHaveTextContent("Could not save the default group.");
    expect(group).toHaveTextContent(label);
    rejectedIds.clear();
    await waitFor(() => expect(save).toBeEnabled());
    await user.click(save);
    await waitFor(() => expect(toast.success).toHaveBeenCalledWith("App settings saved."));
    expect(save).toBeDisabled();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(group).toHaveTextContent(label);
    expect(support.getByRole("combobox", { name: "Budget group" })).toHaveTextContent("No limit");
    await expectWrites(
      [
        [method, "chat", id],
        [method, "chat", id],
      ],
      field,
    );
  },
);

it.each(["private-chat", "member-1", null])(
  "applies custom member drafts from %s without changing untouched limits, resets, or other defaults",
  async (initial) => {
    const budget: Budget = {
      max_budget: 30,
      budget_duration: "1mo",
      budget_limits: [{ max_budget: 10, budget_duration: "7d", reset_at: "2027-12-25T00:00:00Z" }],
      model_rpm_limit: { "model-a": 4 },
    };
    apps = apps.map((app) =>
      app.resource_id === "chat"
        ? {
            ...app,
            default_customer_member_budget_id: initial,
            default_customer_member_budget: initial ? budget : null,
          }
        : app,
    );
    const user = userEvent.setup();
    render(<AppBudgetDefaults scope="member" />);
    const chat = within(await screen.findByRole("group", { name: "Defaults for Chat" }));
    const group = chat.getByRole("combobox", { name: "Budget group" });
    await waitFor(() => expect(group).toBeEnabled());
    await chooseSelectOption(user, group, "Enter custom limits instead");
    if (initial === "private-chat") {
      await user.click(within(await screen.findByRole("dialog")).getByRole("button", { name: "Apply" }));
      expect(screen.getByRole("button", { name: "Save changes" })).toBeDisabled();
      await user.click(chat.getByRole("button", { name: "Edit custom allowance" }));
    }
    const dialog = within(await screen.findByRole("dialog", { name: "Per-member default for Chat" }));
    const spend = dialog.getByRole("spinbutton", { name: "Spend limit (USD)" });
    expect(spend).toHaveValue(initial ? 30 : null);
    fireEvent.change(spend, { target: { value: "" } });
    fireEvent.change(dialog.getByRole("spinbutton", { name: "Tokens per minute" }), { target: { value: "-1" } });
    await user.click(dialog.getByRole("button", { name: "Apply" }));
    expect(dialog.getByRole("spinbutton", { name: "Tokens per minute" })).toBeInvalid();
    fireEvent.change(dialog.getByRole("spinbutton", { name: "Tokens per minute" }), { target: { value: "0" } });
    await user.click(dialog.getByRole("button", { name: "Apply" }));
    expect(writes).toHaveLength(0);
    expect(group).toHaveTextContent("Custom");
    await user.click(chat.getByRole("button", { name: "Edit custom allowance" }));
    const reopened = within(await screen.findByRole("dialog", { name: "Per-member default for Chat" }));
    expect(reopened.getByRole("spinbutton", { name: "Tokens per minute" })).toHaveValue(0);
    fireEvent.change(reopened.getByRole("spinbutton", { name: "Tokens per minute" }), { target: { value: "99" } });
    await user.click(reopened.getByRole("button", { name: "Cancel" }));
    const save = screen.getByRole("button", { name: "Save changes" });
    await user.click(save);
    expect(await chat.findByRole("alert")).toHaveTextContent("Could not save the default group.");
    rejectedIds.clear();
    await waitFor(() => expect(save).toBeEnabled());
    await user.click(save);
    await waitFor(() => expect(save).toBeDisabled());
    for (const request of writes) {
      expect(await request.json()).toEqual({
        default_customer_member_budget: {
          tpm_limit: 0,
          ...(initial ? { max_budget: null } : { budget_duration: "1mo" }),
        },
      });
    }
    expect(apps.find((app) => app.resource_id === "chat")).toMatchObject({
      default_tier_budget_id: "tier-1",
      default_customer_member_budget: initial ? { ...budget, max_budget: null, tpm_limit: 0 } : { tpm_limit: 0 },
    });
    await chooseSelectOption(user, group, /^No limit/);
    await user.click(save);
    await waitFor(() => expect(writes).toHaveLength(3));
    expect(await writes[2].json()).toEqual({ default_customer_member_budget_id: null });
  },
);

it.each(["support", "search"])(
  "retries only failed %s after bulk save, retaining created IDs through stale refreshes",
  async (rejected) => {
    apps = apps.filter((app) => app.resource_id !== "chat");
    staleApps = [...apps];
    rejectedIds = new Set([rejected]);
    const user = userEvent.setup();
    render(<AppBudgetDefaults />);
    const support = within(await screen.findByRole("group", { name: "Defaults for Support" }));
    const supportGroup = support.getByRole("combobox", { name: "Budget group" });
    await waitFor(() => expect(supportGroup).toBeEnabled());
    await chooseSelectOption(user, supportGroup, /^Tier 1/);
    for (const alias of ["Chat", "Search"]) {
      await user.click(screen.getByRole("button", { name: "Add app default" }));
      await chooseSelectOption(user, screen.getByRole("combobox", { name: "Select an app" }), alias);
    }
    const chat = within(screen.getByRole("group", { name: "Defaults for Chat" }));
    const chatGroup = chat.getByRole("combobox", { name: "Budget group" });
    await chooseSelectOption(user, chatGroup, /^Tier 2/);
    const save = screen.getByRole("button", { name: "Save changes" });
    await user.click(save);
    const failed = within(
      screen.getByRole("group", { name: rejected === "support" ? "Defaults for Support" : "Defaults for Search" }),
    );
    expect(await failed.findByRole("alert")).toHaveTextContent("Could not save the default group.");
    expect(chat.queryByRole("alert")).not.toBeInTheDocument();
    await waitFor(() => expect(save).toBeEnabled());
    expect(supportGroup).toHaveTextContent("Tier 1");
    expect(chatGroup).toHaveTextContent("Tier 2");
    expect(toast.success).not.toHaveBeenCalled();
    rejectedIds.clear();
    await user.click(save);
    await waitFor(() => expect(toast.success).toHaveBeenCalledOnce());
    expect(save).toBeDisabled();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    await chooseSelectOption(user, chatGroup, /^No limit/);
    await user.click(save);
    await waitFor(() => expect(toast.success).toHaveBeenCalledTimes(2));
    await expectWrites([
      ["POST", "chat", "tier-2"],
      ["POST", "search", null],
      ["PATCH", "support", "tier-1"],
      rejected === "support" ? ["PATCH", "support", "tier-1"] : ["POST", "search", null],
      ["PATCH", "chat", null],
    ]);
  },
);
