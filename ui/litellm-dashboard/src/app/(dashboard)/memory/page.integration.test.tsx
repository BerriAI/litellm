import { fireEvent, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders, testQueryClient } from "../../../../tests/test-utils";
import type { components } from "@/lib/http/schema";
import Memory from "./page";

vi.unmock("@/app/(dashboard)/hooks/useAuthorized");
vi.unmock("@/lib/toast");

const fetchMock = vi.fn<typeof fetch>();
const calls: { path: string; method: string; body: unknown; params: string }[] = [];
let settings: components["schemas"]["MemorySettings"];
let paginated = false;
let failure = "";
let canEdit = true;
const entry = {
  memory_id: "entry-1",
  key: "demo",
  title: "Demo port",
  content: "Use port 8123",
  evidence: "Fixture recommendation",
  when_to_use: "Configuring the demo",
  scope: "demo",
  kind: "decision",
  certainty: "inferred",
  source: "fixture.md",
  created_at: "2026-09-12T00:00:00Z",
  updated_at: "2026-09-12T00:00:00Z",
  actor: "u1",
  actor_name: "Alex Rivera",
  user_id: "u1",
  team_id: "engineering",
  team_name: "Engineering",
};
const session = (user_role: string) => {
  const payload = { key: "sk-test", user_id: "u1", user_role, exp: Math.floor(Date.now() / 1000) + 3600 };
  document.cookie = `token=${btoa("{}")}\.${btoa(JSON.stringify(payload))}.signature; path=/`;
};

beforeEach(async () => {
  await testQueryClient.cancelQueries();
  testQueryClient.clear();
  calls.length = 0;
  settings = { enabled: false, everyone: true, user_ids: [] };
  paginated = false;
  failure = "";
  canEdit = true;
  vi.clearAllMocks();
  fetchMock.mockImplementation(async (input, init) => {
    const request =
      input instanceof Request ? input : new Request(new URL(String(input), window.location.origin), init);
    const url = new URL(request.url);
    const path = url.pathname;
    const text = request.method === "GET" ? "" : await request.text();
    const call = { path, method: request.method, body: text ? JSON.parse(text) : undefined, params: url.search };
    calls.push(call);
    if (failure === path && (path !== "/memory/v2/settings" || request.method === "PUT")) {
      return new Response(JSON.stringify({ detail: "Memory service unavailable" }), {
        status: 503,
        headers: { "Content-Type": "application/json" },
      });
    }
    const response = () => {
      if (path === "/memory/v2/settings") {
        if (request.method === "PUT") settings = JSON.parse(text);
        return { ...settings, user_names: { u1: "Alex Rivera" } };
      }
      if (path === "/memory/v2/status")
        return {
          active: settings.enabled && (settings.everyone || settings.user_ids?.includes("u1")),
          enabled: settings.enabled,
          user_id: "u1",
          user_name: "Alex Rivera",
          team_ids: ["engineering"],
          admin_view: false,
        };
      if (path === "/memory/v2/entries") {
        if (paginated) {
          const offset = url.searchParams.get("before_memory_id") === "entry-19" ? 20 : 0;
          return Array.from({ length: offset ? 1 : 20 }, (_, i) => ({
            ...entry,
            can_edit: canEdit,
            memory_id: `entry-${offset + i}`,
            title: `Memory ${offset + i}`,
          }));
        }
        return [{ ...entry, can_edit: canEdit }];
      }
      if (path === "/memory/v2/entries/entry-1") return { ...entry, can_edit: canEdit };
      if (path === "/v1/memory") return { memories: [], total: 0 };
      if (path === "/v2/team/list")
        return { teams: [{ team_id: "engineering", team_alias: "Engineering" }], page: 1, total_pages: 1, total: 1 };
      if (path === "/user/list")
        return { users: [{ user_id: "u1", user_email: "alex@example.test" }], page: 1, total_pages: 1, total: 1 };
      if (path.includes("/team/list") || path.includes("/organization")) return [];
      return {};
    };
    return new Response(JSON.stringify(response()), { status: 200, headers: { "Content-Type": "application/json" } });
  });
  vi.stubGlobal("fetch", fetchMock);
});

describe("Memory dashboard", () => {
  it("shows recent content and attribution while automatic memory is off", async () => {
    session("internal_user");
    renderWithProviders(<Memory />);
    expect(await screen.findByText("Off · Managed by your admin")).toBeVisible();
    expect(await screen.findByText("Use port 8123")).toBeVisible();
    expect(screen.getByText("Alex Rivera")).toBeVisible();
    expect(screen.getByText("Engineering")).toBeVisible();
    expect(screen.queryByRole("switch")).not.toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: "Administration" })).not.toBeInTheDocument();
    expect(calls.some(({ path }) => path === "/memory/v2/settings")).toBe(false);
  });

  it("lets a proxy admin enable everyone through one configuration", async () => {
    session("proxy_admin");
    const user = userEvent.setup();
    renderWithProviders(<Memory />);
    await user.click(screen.getByRole("tab", { name: "Administration" }));
    const toggle = await screen.findByRole("switch", { name: "Gateway memory" });
    expect(toggle).not.toBeChecked();
    await user.click(toggle);
    await user.click(screen.getByRole("button", { name: "Save changes" }));
    await waitFor(() => expect(settings.enabled).toBe(true));
    expect(settings.everyone).toBe(true);
    await waitFor(() => expect(screen.queryByText("Unsaved changes")).not.toBeInTheDocument());
    expect(screen.getByRole("link", { name: "Manage team permissions" })).toHaveAttribute("href", "/ui/teams");
    await user.click(screen.getByRole("tab", { name: "Memories" }));
    expect(await screen.findByText("On · Managed by your admin")).toBeVisible();
    expect(calls.some(({ path }) => path.includes("policies") || path.includes("preference"))).toBe(false);
  });

  it("enrolls selected users without a key or sharing selector", async () => {
    session("proxy_admin");
    const user = userEvent.setup();
    renderWithProviders(<Memory />);
    await user.click(screen.getByRole("tab", { name: "Administration" }));
    await user.click(await screen.findByRole("switch", { name: "Gateway memory" }));
    await user.click(screen.getByRole("combobox", { name: "Enable for" }));
    await user.click(screen.getByRole("option", { name: "Selected users" }));
    await user.click(screen.getByLabelText("Add a user"));
    await user.click(await screen.findByRole("option", { name: "alex@example.test" }));
    expect(screen.getByRole("button", { name: "Remove alex@example.test" })).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Save changes" }));
    await waitFor(() => expect(settings).toEqual({ enabled: true, everyone: false, user_ids: ["u1"] }));
  });

  it("shows saved user names after loading and sends only editable settings", async () => {
    session("proxy_admin");
    settings = { enabled: true, everyone: false, user_ids: ["u1"] };
    const user = userEvent.setup();
    renderWithProviders(<Memory />);
    await user.click(screen.getByRole("tab", { name: "Administration" }));
    expect(await screen.findByRole("button", { name: "Remove Alex Rivera" })).toBeVisible();
    await user.click(screen.getByRole("switch", { name: "Gateway memory" }));
    await user.click(screen.getByRole("button", { name: "Save changes" }));
    await waitFor(() => expect(settings).toEqual({ enabled: false, everyone: false, user_ids: ["u1"] }));
  });

  it("keeps an unsuccessful activation unsaved and shows the error", async () => {
    session("proxy_admin");
    failure = "/memory/v2/settings";
    const user = userEvent.setup();
    renderWithProviders(<Memory />);
    await user.click(screen.getByRole("tab", { name: "Administration" }));
    await user.click(await screen.findByRole("switch", { name: "Gateway memory" }));
    await user.click(screen.getByRole("button", { name: "Save changes" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Memory service unavailable");
    expect(settings.enabled).toBe(false);
    expect(screen.getByText("Unsaved changes")).toBeVisible();
  });

  it("lets viewers inspect but disables administration changes", async () => {
    session("proxy_admin_viewer");
    const user = userEvent.setup();
    renderWithProviders(<Memory />);
    await user.click(await screen.findByRole("button", { name: "Details for Demo port" }));
    expect(screen.queryByRole("button", { name: "Edit memory" })).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Close" }));
    await user.click(screen.getByRole("tab", { name: "Administration" }));
    expect(await screen.findByRole("switch", { name: "Gateway memory" })).toHaveAttribute("aria-disabled", "true");
    expect(screen.getByRole("button", { name: "Save changes" })).toBeDisabled();
  });

  it("does not show edit controls for a readable teammate record", async () => {
    session("internal_user");
    canEdit = false;
    const user = userEvent.setup();
    renderWithProviders(<Memory />);
    await user.click(await screen.findByRole("button", { name: "Details for Demo port" }));
    expect(screen.queryByRole("button", { name: "Edit memory" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Delete memory" })).not.toBeInTheDocument();
    expect(screen.getAllByText("Use port 8123")[0]).toBeVisible();
  });

  it("preserves attribution and revision when an owner corrects content while off", async () => {
    session("internal_user");
    const user = userEvent.setup();
    renderWithProviders(<Memory />);
    await user.click(await screen.findByRole("button", { name: "Details for Demo port" }));
    await user.click(await screen.findByRole("button", { name: "Edit memory" }));
    fireEvent.change(screen.getByLabelText("Correct this memory"), { target: { value: "Use port 8124" } });
    await user.click(screen.getByRole("button", { name: "Save correction" }));
    const expectedCall = {
      path: "/memory/v2/entries/entry-1",
      method: "PUT",
      params: "",
      body: {
        key: entry.key,
        title: entry.title,
        content: "Use port 8124",
        evidence: entry.evidence,
        when_to_use: entry.when_to_use,
        scope: entry.scope,
        kind: entry.kind,
        certainty: entry.certainty,
        source: entry.source,
        expected_revision: entry.updated_at,
      },
    };
    await waitFor(() => expect(calls).toContainEqual(expectedCall));
  });

  it("clears the search and team filter together", async () => {
    session("internal_user");
    const user = userEvent.setup();
    renderWithProviders(<Memory />);
    await screen.findByText("Off · Managed by your admin");
    fireEvent.change(screen.getByRole("textbox", { name: "Search memories" }), { target: { value: "demo" } });
    await user.click(screen.getByLabelText("Team"));
    await user.click(await screen.findByRole("option", { name: "Engineering" }));
    await waitFor(() => expect(calls.some(({ params }) => params.includes("team_id=engineering"))).toBe(true));
    await user.click(screen.getByRole("button", { name: "Clear filters" }));
    expect(screen.getByRole("textbox", { name: "Search memories" })).toHaveValue("");
    expect(screen.queryByRole("button", { name: "Clear filters" })).not.toBeInTheDocument();
    expect(calls.some(({ params }) => params.includes("key_id"))).toBe(false);
  });

  it("labels the contributor picker and sends its selected user filter", async () => {
    session("proxy_admin");
    const user = userEvent.setup();
    renderWithProviders(<Memory />);
    await screen.findByText("Off · Managed by your admin");
    await user.click(screen.getByRole("combobox", { name: "Contributor" }));
    await user.click(await screen.findByRole("option", { name: "alex@example.test" }));
    await waitFor(() => expect(calls.some(({ params }) => params.includes("user_id=u1"))).toBe(true));
  });

  it("searches and paginates without replacing the visible first page", async () => {
    session("internal_user");
    paginated = true;
    const user = userEvent.setup();
    renderWithProviders(<Memory />);
    await user.click(await screen.findByRole("button", { name: "Load more memories" }));
    expect(await screen.findByText("Memory 20")).toBeVisible();
    expect(screen.getByText("Memory 0")).toBeVisible();
    expect(screen.getByText("21 memories shown")).toBeVisible();
    await user.type(screen.getByRole("textbox", { name: "Search memories" }), "demo");
    await waitFor(() => expect(calls.some(({ params }) => params.includes("query=demo"))).toBe(true));
  });

  it("shows a loading failure instead of claiming an empty collection", async () => {
    session("internal_user");
    failure = "/memory/v2/entries";
    renderWithProviders(<Memory />);
    expect(await screen.findByRole("alert")).toHaveTextContent("Memory service unavailable");
    expect(screen.queryByText("No memories yet")).not.toBeInTheDocument();
  });

  it("keeps V1 separate and loads its records only when selected", async () => {
    session("proxy_admin");
    const user = userEvent.setup();
    renderWithProviders(<Memory />);
    expect(calls.some(({ path }) => path === "/v1/memory")).toBe(false);
    await user.click(screen.getByRole("tab", { name: "Memory API (V1)" }));
    await waitFor(() => expect(calls.some(({ path }) => path === "/v1/memory")).toBe(true));
    expect(screen.queryByText(/draft deprecation list/i)).not.toBeInTheDocument();
  });
});
