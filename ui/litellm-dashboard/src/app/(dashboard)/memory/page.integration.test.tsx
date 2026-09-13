import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { renderWithProviders, testQueryClient } from "../../../../tests/test-utils";
import Memory from "./page";
import { Toaster } from "@/components/ui/sonner";

vi.unmock("@/app/(dashboard)/hooks/useAuthorized");
vi.unmock("@/lib/toast");

const fetchMock = vi.fn<typeof fetch>();
const calls: { path: string; method: string; body: unknown; keyId: string | null }[] = [];
let enabled = false;
let activation = "opt_in";
let available = true;
let paginated = false;
let failPreference = false;
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
};
const session = (user_role: string) => {
  const payload = { key: "sk-test", user_id: "u1", user_role, exp: Math.floor(Date.now() / 1000) + 3600 };
  document.cookie = `token=${btoa("{}")}\.${btoa(JSON.stringify(payload))}.signature; path=/`;
};

beforeEach(async () => {
  await testQueryClient.cancelQueries();
  testQueryClient.clear();
  calls.length = 0;
  enabled = false;
  activation = "opt_in";
  available = true;
  paginated = false;
  failPreference = false;
  vi.clearAllMocks();
  fetchMock.mockImplementation(async (input, init) => {
    const request =
      input instanceof Request ? input : new Request(new URL(String(input), window.location.origin), init);
    const path = new URL(request.url).pathname;
    const text = request.method === "GET" ? "" : await request.text();
    const url = new URL(request.url);
    const keyId = url.searchParams.get("key_id");
    if (url.pathname === "/key/list" && url.searchParams.get("user_id")) {
      expect(url.searchParams.get("include_team_keys")).toBe("false");
      expect(url.searchParams.get("include_created_by_keys")).toBe("false");
    }
    const call = { path, method: request.method, body: text ? JSON.parse(text) : undefined, keyId };
    calls.push(call);
    if (path === "/v2/memory/preference" && request.method === "PUT" && failPreference) {
      return new Response(JSON.stringify({ detail: "Preference unavailable" }), {
        status: 503,
        headers: { "Content-Type": "application/json" },
      });
    }
    const response = () => {
      if (path === "/v2/memory/preference") {
        if (request.method === "PUT") enabled = JSON.parse(text).enabled;
        return { enabled };
      }
      if (path === "/v2/memory/policies") return [];
      if (path === "/v1/memory") return { memories: [], total: 0 };
      if (path === "/v2/memory/status")
        return {
          active: available && (activation === "automatic" || enabled),
          opted_in: enabled,
          activation,
          scope: available ? "key" : null,
        };
      if (path === "/v2/memory/entries") {
        if (request.method === "POST") return entry;
        if (keyId === "b".repeat(64))
          return [{ ...entry, memory_id: "other", title: "Other key memory", content: "Another project" }];
        if (paginated) {
          const offset = new URL(request.url).searchParams.get("before_memory_id") === "entry-19" ? 20 : 0;
          return Array.from({ length: offset ? 1 : 20 }, (_, i) => ({
            ...entry,
            memory_id: `entry-${offset + i}`,
            title: `Memory ${offset + i}`,
          }));
        }
        return [entry];
      }
      if (path.includes("/key/list"))
        return {
          keys: [
            { token: "a".repeat(64), key_alias: "QA key" },
            { token: "b".repeat(64), key_alias: "Other key" },
          ],
          total_count: 2,
          current_page: 1,
          total_pages: 1,
        };
      if (path.includes("/team/list") || path.includes("/organization")) return [];
      return {};
    };
    const data = response();
    return new Response(JSON.stringify(data), { status: 200, headers: { "Content-Type": "application/json" } });
  });
  vi.stubGlobal("fetch", fetchMock);
});

describe("Memory dashboard", () => {
  it("preserves observation attribution when a person corrects saved content", async () => {
    session("internal_user");
    enabled = true;
    const user = userEvent.setup();
    renderWithProviders(<Memory />);
    await user.click(await screen.findByRole("button", { name: "Details for Demo port" }));
    await user.click(await screen.findByRole("button", { name: "Edit memory" }));
    fireEvent.change(screen.getByLabelText("Correct this memory"), { target: { value: "Use port 8124" } });
    await user.click(screen.getByRole("button", { name: "Save correction" }));
    const expectedCall = {
      path: "/v2/memory/entries",
      keyId: "a".repeat(64),
      method: "POST",
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

  it("lets an administrator choose automatic activation and a sharing scope", async () => {
    session("proxy_admin");
    const user = userEvent.setup();
    renderWithProviders(<Memory />);
    expect(screen.queryByLabelText("Activation")).not.toBeInTheDocument();
    expect(calls.filter(({ path }) => path === "/v1/memory" || path === "/v2/memory/policies")).toEqual([]);
    await user.click(screen.getByRole("button", { name: "Advanced settings" }));
    expect(await screen.findByLabelText("Activation")).toHaveValue("opt_in");
    await user.selectOptions(screen.getByLabelText("Activation"), "automatic");
    await user.selectOptions(screen.getByLabelText("Who shares the memories"), "team");
    await user.click(screen.getByRole("button", { name: "Save memory policy" }));
    const expectedCall = {
      path: "/v2/memory/policies",
      keyId: null,
      method: "PUT",
      body: { target_type: "gateway", target_id: "*", activation: "automatic", scope: "team" },
    };
    await waitFor(() => expect(calls).toContainEqual(expectedCall));
    await waitFor(() => expect(screen.getByRole("button", { name: "Save memory policy" })).toBeEnabled());
    expect(screen.queryByText(/draft deprecation list/i)).not.toBeInTheDocument();
  });

  it("offers members their own preference without fetching administrator memory rows", async () => {
    session("internal_user");
    const user = userEvent.setup();
    renderWithProviders(<Memory />);
    const preference = await screen.findByRole("switch", { name: "Memory" });
    await waitFor(() => expect(preference).toBeEnabled());
    expect(preference).not.toBeChecked();
    expect(await screen.findByText("Use port 8123")).toBeVisible();
    expect(screen.queryByText("Fixture recommendation")).not.toBeInTheDocument();
    await user.click(preference);
    const expectedCall = {
      path: "/v2/memory/preference",
      method: "PUT",
      body: { enabled: true },
      keyId: "a".repeat(64),
    };
    await waitFor(() => expect(calls).toContainEqual(expectedCall));
    await waitFor(() => expect(preference).toBeChecked());
    await waitFor(() => expect(preference).not.toHaveAttribute("aria-disabled", "true"));
    await user.click(preference);
    await waitFor(() => expect(preference).not.toBeChecked());
    expect(screen.getByText("Use port 8123")).toBeVisible();
    expect(calls.filter(({ path }) => path === "/v1/memory" || path === "/v2/memory/policies")).toEqual([]);
    expect(screen.queryByRole("button", { name: "Save memory policy" })).not.toBeInTheDocument();
  });

  it("allows viewers to inspect memory while disabling preference writes", async () => {
    session("internal_user_viewer");
    renderWithProviders(<Memory />);
    expect(await screen.findByRole("switch", { name: "Memory" })).toHaveAttribute("aria-disabled", "true");
    expect(await screen.findByText("Use port 8123")).toBeVisible();
  });

  it("appends older memories and resets the feed when switching keys", async () => {
    session("proxy_admin");
    paginated = true;
    const user = userEvent.setup();
    renderWithProviders(<Memory />);
    const feed = await screen.findByRole("list", { name: "Saved memories" });
    await waitFor(() => expect(within(feed).getAllByRole("listitem")).toHaveLength(20));
    await user.click(screen.getByRole("button", { name: "Load more memories" }));
    await waitFor(() => expect(within(feed).getAllByRole("listitem")).toHaveLength(21));
    expect(screen.getByRole("heading", { name: "Memory 0" })).toBeVisible();
    await user.click(screen.getByLabelText("Virtual key"));
    await user.click(await screen.findByRole("option", { name: "Other key" }));
    expect(await screen.findByText("Another project")).toBeVisible();
    expect(screen.queryByRole("heading", { name: "Memory 0" })).not.toBeInTheDocument();
  });

  it("shows administrator-managed memory as on even when the preference is off", async () => {
    session("internal_user");
    activation = "automatic";
    renderWithProviders(<Memory />);
    const toggle = await screen.findByRole("switch", { name: "Memory" });
    expect(toggle).toBeChecked();
    expect(toggle).toHaveAttribute("aria-disabled", "true");
    expect(screen.getByText(/Your administrator manages this setting/)).toBeVisible();
  });

  it("shows memory as unavailable when an automatic policy cannot resolve a sharing scope", async () => {
    session("internal_user");
    activation = "automatic";
    available = false;
    renderWithProviders(<Memory />);
    const toggle = await screen.findByRole("switch", { name: "Memory" });
    expect(toggle).not.toBeChecked();
    expect(toggle).toHaveAttribute("aria-disabled", "true");
    expect(screen.getByText("Memory is off. Your administrator can make it available for this key.")).toBeVisible();
  });

  it("keeps the actual state off when saving a preference fails", async () => {
    session("internal_user");
    failPreference = true;
    const user = userEvent.setup();
    renderWithProviders(
      <>
        <Memory />
        <Toaster />
      </>,
    );
    const toggle = await screen.findByRole("switch", { name: "Memory" });
    await user.click(toggle);
    expect(await screen.findByText("Preference unavailable")).toBeVisible();
    await waitFor(() => expect(toggle).not.toHaveAttribute("aria-disabled", "true"));
    expect(toggle).not.toBeChecked();
  });
});
