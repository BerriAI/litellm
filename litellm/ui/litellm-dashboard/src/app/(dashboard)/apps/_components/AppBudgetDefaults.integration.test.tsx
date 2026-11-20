import userEvent, { type UserEvent } from "@testing-library/user-event";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { toast } from "sonner";
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
import { appFixture, mockAppRequests } from "./appTestUtils";

vi.mock("sonner", () => ({ toast: { success: vi.fn() } }));
const auth = vi.hoisted(() => ({ accessToken: "test-session", userId: "user-1", userRole: "Admin" }));
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({ default: () => auth }));
const resources = ["Chat", "Support", "Search"].map((alias) => ({
  resource_type: "team" as const,
  resource_id: alias.toLowerCase(),
  alias,
}));
const catalog = (groups: Record<string, string>) => ({
  default_budgets: Object.entries(groups).map(([budget_id, name]) => ({ budget_id, name })),
});
let apps: App[];
let staleApps: App[] | null;
let rejectedIds: Set<string>;
let network: ReturnType<typeof mockAppRequests>;
let writes: Request[];

async function saveDefault(request: Request) {
  const body: Pick<
    App,
    "default_tier_budget_id" | "default_customer_member_budget_id" | "default_customer_member_budget"
  > & { resource_id?: string } = await request.json();
  const id = body.resource_id ?? apps.find((app) => request.url.endsWith(`/${app.app_id}`))?.resource_id;
  if (!id) throw new Error(`Unknown app ${request.url}`);
  if (rejectedIds.has(id)) return json({ detail: "Could not save the default group." }, 503);
  const previous = apps.find((app) => app.resource_id === id);
  const app = {
    ...appFixture(previous),
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
  return app;
}

beforeEach(() => {
  auth.userRole = "Admin";
  apps = resources.slice(0, 2).map((resource, index) => ({
    ...appFixture(resource),
    app_id: `app-${resource.resource_id}`,
    default_tier_budget_id: `tier-${index + 1}`,
    default_customer_member_budget_id: `member-${index + 1}`,
  }));
  staleApps = null;
  rejectedIds = new Set(["chat"]);
  vi.mocked(toast.success).mockClear();
  const routes: Parameters<typeof mockAppRequests>[0] = {
    "GET /management/v1/apps/definition": () => ({ resource_type: "team" }),
    "GET /management/v1/apps/resources": () =>
      resources.map((resource) => ({
        ...resource,
        app_id: (staleApps ?? apps).find((app) => app.resource_id === resource.resource_id)?.app_id ?? null,
      })),
    "GET /management/v1/apps": () => staleApps ?? apps,
    "GET /management/v1/apps/app-chat": () => (staleApps ?? apps).find((app) => app.app_id === "app-chat"),
    "POST /management/v1/apps": saveDefault,
    ...Object.fromEntries(
      resources.map(({ resource_id }) => [`PATCH /management/v1/apps/app-${resource_id}`, saveDefault]),
    ),
    "GET /models": () => ({ data: [{ id: "model-a" }] }),
    "GET /management/v1/apps/resources/team/chat/models": () => ["model-a"],
    "GET /management/v1/app_customer_member_budget_groups": () =>
      catalog({ "member-1": "Member 1", "member-2": "Member 2" }),
    "GET /management/v1/app_customer_organization_budget_groups": () =>
      catalog({ "tier-1": "Tier 1", "tier-2": "Tier 2" }),
    "GET /management/v1/customer_organization_budget_groups": () =>
      catalog({ "org-standard": "Standard", "org-enterprise": "Enterprise" }),
  };
  network = mockAppRequests(routes);
  writes = network.writes;
});
afterEach(() => vi.unstubAllGlobals());

async function defaultGroup(alias: string) {
  const row = await screen.findByRole("group", { name: `Defaults for ${alias}` });
  const group = within(row).getByRole("combobox", { name: "Budget group" });
  await waitFor(() => expect(group).toBeEnabled());
  return { row, group };
}

async function retryDefaults(user: UserEvent, element: HTMLElement, label: string) {
  const row = within(element);
  const save = screen.getByRole("button", { name: "Save changes" });
  await user.click(save);
  expect(await row.findByRole("alert")).toHaveTextContent("Could not save the default group.");
  expect(row.getByRole("combobox", { name: "Budget group" })).toHaveTextContent(label);
  rejectedIds.clear();
  await waitFor(() => expect(save).toBeEnabled());
  await user.click(save);
  await waitFor(() => expect(toast.success).toHaveBeenCalledWith("App settings saved."));
  expect(save).toBeDisabled();
  expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  return save;
}

it("changes only the current app's member default from its Budget groups page without loading other apps", async () => {
  rejectedIds.clear();
  const user = userEvent.setup();
  render(<AppBudgetGroupsPage />, { searchParams: "?app_id=app-chat" });
  const { group } = await defaultGroup("Chat");
  await chooseSelectOption(user, group, /^Member 2/);
  const defaults = within(screen.getByRole("region", { name: "Default per-member budget" }));
  await user.click(defaults.getByRole("button", { name: "Save changes" }));
  await waitFor(() => expect(toast.success).toHaveBeenCalledWith("App settings saved."));
  await expectWrites([["PATCH", "chat", "member-2"]], "default_customer_member_budget_id");
  expect(screen.queryByRole("group", { name: "Defaults for Support" })).not.toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Add app default" })).not.toBeInTheDocument();
  const paths = network.requests.map((request) => new URL(request.url).pathname);
  expect(paths).toEqual(
    expect.arrayContaining(["/management/v1/apps/app-chat", "/management/v1/app_customer_member_budget_groups"]),
  );
  expect(paths.every((path) => !path.endsWith("/apps/resources"))).toBe(true);
});

it("edits shared app-member templates from the app page without changing the app's assignment", async () => {
  network.routes["PATCH /management/v1/app_customer_member_budget_groups"] =
    network.routes["GET /management/v1/app_customer_member_budget_groups"];
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
    expect(await request.clone().json()).toEqual({
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
    const { group: supportGroup } = await defaultGroup("Support");
    await waitFor(() => expect(supportGroup).toHaveTextContent(scope === "member" ? "Member 2" : "Tier 2"));
    if (method === "POST") {
      await user.click(screen.getByRole("button", { name: "Add app default" }));
      await chooseSelectOption(user, screen.getByRole("combobox", { name: "Select an app" }), "Chat");
    }
    const { row, group } = await defaultGroup("Chat");
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
    await waitFor(() => expect(supportGroup).toHaveTextContent("No limit"));
    await retryDefaults(user, row, label);
    expect(group).toHaveTextContent(label);
    expect(supportGroup).toHaveTextContent("No limit");
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
    const { row, group } = await defaultGroup("Chat");
    const chat = within(row);
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
    const save = await retryDefaults(user, row, "Custom");
    expect(writes).toHaveLength(2);
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
    const { group: supportGroup } = await defaultGroup("Support");
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
