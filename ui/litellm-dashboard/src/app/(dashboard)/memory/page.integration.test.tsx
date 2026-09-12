import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { renderWithProviders, testQueryClient } from "../../../../tests/test-utils";
import Memory from "./page";

vi.unmock("@/app/(dashboard)/hooks/useAuthorized");
vi.unmock("@/lib/toast");

const fetchMock = vi.fn<typeof fetch>();
const calls: { path: string; method: string; body: unknown }[] = [];
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
  vi.clearAllMocks();
  fetchMock.mockImplementation(async (input, init) => {
    const request =
      input instanceof Request ? input : new Request(new URL(String(input), window.location.origin), init);
    const path = new URL(request.url).pathname;
    const text = request.method === "GET" ? "" : await request.text();
    calls.push({ path, method: request.method, body: text ? JSON.parse(text) : undefined });
    const response = () => {
      if (path === "/v2/memory/preference") return { enabled: request.method === "PUT" };
      if (path === "/v2/memory/policies") return [];
      if (path === "/v1/memory") return { memories: [], total: 0 };
      if (path === "/v2/memory/status") return { active: true, scope: "key" };
      if (path === "/v2/memory/entries") return request.method === "POST" ? entry : [entry];
      if (path.includes("/key/list"))
        return {
          keys: [{ token: "a".repeat(64), key_alias: "QA key" }],
          total_count: 1,
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

describe("Gateway memory settings", () => {
  it("preserves observation attribution when a person corrects saved content", async () => {
    session("internal_user");
    const user = userEvent.setup();
    renderWithProviders(<Memory />);
    await user.click(await screen.findByLabelText("Virtual key"));
    await user.click(await screen.findByRole("option", { name: "QA key" }));
    await user.click(await screen.findByRole("button", { name: "Edit memory" }));
    await user.clear(screen.getByLabelText("Correct this memory"));
    await user.type(screen.getByLabelText("Correct this memory"), "Use port 8124");
    await user.click(screen.getByRole("button", { name: "Save correction" }));
    await waitFor(() =>
      expect(calls).toContainEqual({
        path: "/v2/memory/entries",
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
      }),
    );
  });

  it("lets an administrator choose automatic activation and a sharing scope", async () => {
    session("proxy_admin");
    const user = userEvent.setup();
    renderWithProviders(<Memory />);
    await user.selectOptions(await screen.findByLabelText("Activation"), "automatic");
    await user.selectOptions(screen.getByLabelText("Who shares the memories"), "team");
    await user.click(screen.getByRole("button", { name: "Save memory policy" }));
    await waitFor(() =>
      expect(calls).toContainEqual({
        path: "/v2/memory/policies",
        method: "PUT",
        body: { target_type: "gateway", target_id: "*", activation: "automatic", scope: "team" },
      }),
    );
    await waitFor(() => expect(screen.getByRole("button", { name: "Save memory policy" })).toBeEnabled());
    expect(screen.queryByText(/draft deprecation list/i)).not.toBeInTheDocument();
  });

  it("offers members their own preference without fetching administrator memory rows", async () => {
    session("internal_user");
    const user = userEvent.setup();
    renderWithProviders(<Memory />);
    const preference = await screen.findByRole("switch", { name: "Use memory when offered" });
    await waitFor(() => expect(preference).toBeEnabled());
    await user.click(preference);
    await waitFor(() =>
      expect(calls).toContainEqual({ path: "/v2/memory/preference", method: "PUT", body: { enabled: true } }),
    );
    expect(calls.filter(({ path }) => path === "/v1/memory" || path === "/v2/memory/policies")).toEqual([]);
    expect(screen.queryByRole("button", { name: "Save memory policy" })).not.toBeInTheDocument();
  });

  it("allows viewers to inspect memory while disabling preference writes", async () => {
    session("internal_user_viewer");
    renderWithProviders(<Memory />);
    expect(await screen.findByRole("switch", { name: "Use memory when offered" })).toHaveAttribute(
      "aria-disabled",
      "true",
    );
    expect(screen.getByRole("heading", { name: "Saved gateway memories" })).toBeInTheDocument();
  });
});
