import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { toast } from "sonner";
import { registerBaseUrlGetter } from "@/lib/http/runtime";
import {
  act,
  chooseSelectOption,
  jsonResponse as json,
  renderWithProviders as render,
  screen,
  testQueryClient,
  waitFor,
  within,
} from "../../../../../tests/test-utils";
import AppOrganizationDefaults from "./AppOrganizationDefaults";

vi.mock("sonner", () => ({ toast: { success: vi.fn() } }));
const resources = ["Chat", "Support", "Search"].map((alias) => ({
  resource_type: "team",
  resource_id: alias.toLowerCase(),
  alias,
}));
type AppFixture = (typeof resources)[number] & { app_id: string; default_tier_budget_id: string | null };
const writes: Request[] = [];
let apps: AppFixture[];
let staleApps: AppFixture[] | null;
let rejectedIds: Set<string>;

beforeEach(() => {
  apps = resources.slice(0, 2).map((resource, index) => ({
    ...resource,
    app_id: `app-${resource.resource_id}`,
    default_tier_budget_id: `tier-${index + 1}`,
  }));
  staleApps = null;
  rejectedIds = new Set(["chat"]);
  writes.length = 0;
  vi.mocked(toast.success).mockClear();
  testQueryClient.clear();
  registerBaseUrlGetter(() => "http://localhost");
  vi.stubGlobal(
    "fetch",
    vi.fn(async (request: Request) => {
      const path = new URL(request.url).pathname;
      if (request.method !== "GET") {
        writes.push(request.clone());
        const body: { default_tier_budget_id: string | null; resource_id?: string } = await request.json();
        const id = body.resource_id ?? apps.find((app) => path.endsWith(`/${app.app_id}`))?.resource_id;
        if (!id) throw new Error(`Unknown app ${path}`);
        if (rejectedIds.has(id)) return json({ detail: "Could not save the default group." }, 503);
        const app = { ...resources.find((resource) => resource.resource_id === id)!, app_id: `app-${id}`, ...body };
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
      if (path.endsWith("/app_customer_organization_budget_groups"))
        return json({
          default_budgets: [
            { budget_id: "tier-1", name: "Tier 1" },
            { budget_id: "tier-2", name: "Tier 2" },
          ],
        });
      if (path.endsWith("/customer_organization_budget_groups"))
        return json({
          default_budgets: [
            { budget_id: "org-standard", name: "Standard" },
            { budget_id: "org-enterprise", name: "Enterprise" },
          ],
        });
      throw new Error(`Unexpected request ${request.method} ${path}`);
    }),
  );
});
afterEach(() => vi.unstubAllGlobals());

it.each([
  ["PATCH", "Tier 2", "tier-2", "tier-1"],
  ["PATCH", "Tier 2", "tier-2", "org-standard"],
  ["PATCH", "No limit", null, "org-standard"],
  ["POST", "Tier 2", "tier-2", null],
] as const)("saves %s default %s from %s/%s without changing another app", async (method, label, id, initial) => {
  apps = apps.flatMap((app) => {
    if (app.resource_id !== "chat") return [app];
    return method === "POST" ? [] : [{ ...app, default_tier_budget_id: initial }];
  });
  const user = userEvent.setup();
  render(<AppOrganizationDefaults />);
  const support = within(await screen.findByRole("group", { name: "Defaults for Support" }));
  await waitFor(() => expect(support.getByRole("combobox", { name: "Budget group" })).toHaveTextContent("Tier 2"));
  if (method === "POST") {
    await user.click(screen.getByRole("button", { name: "Add app default" }));
    await chooseSelectOption(user, screen.getByRole("combobox", { name: "Select an app" }), "Chat");
  }
  const chat = within(await screen.findByRole("group", { name: "Defaults for Chat" }));
  const group = chat.getByRole("combobox", { name: "Budget group" });
  await waitFor(() => expect(group).toBeEnabled());
  if (initial === "org-standard") {
    expect(group).toHaveTextContent("Standard (overall organization group)");
    await user.click(group);
    expect(await screen.findByRole("option", { name: /^Standard/ })).toHaveAttribute("aria-disabled", "true");
    expect(screen.queryByRole("option", { name: /^Enterprise/ })).not.toBeInTheDocument();
    await user.keyboard("{Escape}");
  }
  expect(writes).toHaveLength(0);
  await chooseSelectOption(user, group, new RegExp(`^${label}`));
  apps = apps.map((app) => (app.resource_id === "support" ? { ...app, default_tier_budget_id: null } : app));
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
  expect(writes).toHaveLength(2);
  for (const request of writes) {
    expect(request.method).toBe(method);
    expect(new URL(request.url).pathname).toBe(`/management/v1/apps${method === "PATCH" ? "/app-chat" : ""}`);
    expect(await request.json()).toEqual({
      default_tier_budget_id: id,
      ...(method === "POST" ? { resource_id: "chat", resource_type: "team" } : {}),
    });
  }
});

it.each(["support", "search"])(
  "retries only failed %s after bulk save, retaining created IDs through stale refreshes",
  async (rejected) => {
    apps = apps.filter((app) => app.resource_id !== "chat");
    staleApps = [...apps];
    rejectedIds = new Set([rejected]);
    const user = userEvent.setup();
    render(<AppOrganizationDefaults />);
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
    expect(writes).toHaveLength(5);
    expect(writes.map((request) => [request.method, new URL(request.url).pathname])).toEqual([
      ["POST", "/management/v1/apps"],
      ["POST", "/management/v1/apps"],
      ["PATCH", "/management/v1/apps/app-support"],
      rejected === "support" ? ["PATCH", "/management/v1/apps/app-support"] : ["POST", "/management/v1/apps"],
      ["PATCH", "/management/v1/apps/app-chat"],
    ]);
    expect(await Promise.all(writes.map((request) => request.json()))).toEqual([
      { resource_id: "chat", resource_type: "team", default_tier_budget_id: "tier-2" },
      { resource_id: "search", resource_type: "team", default_tier_budget_id: null },
      { default_tier_budget_id: "tier-1" },
      rejected === "support"
        ? { default_tier_budget_id: "tier-1" }
        : { resource_id: "search", resource_type: "team", default_tier_budget_id: null },
      { default_tier_budget_id: null },
    ]);
  },
);
