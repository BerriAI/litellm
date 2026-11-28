import { beforeEach, afterEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, renderWithProviders, screen, testQueryClient } from "../../../../tests/test-utils";
import ExternalCloudAgents from "./ExternalCloudAgents";
import type { Source, SpendReport } from "./hooks";

const root = "/v1/external_cloud_agents";
const connection = {
  source_id: "source-1",
  hook_path: "/grok_bot/inference_hook/source-1",
  ingest_key: "test-ingest-key",
};
const source: Source = {
  source_id: "source-1",
  vendor: "grok_bot",
  display_name: "GrokBot",
  tenant_id: "tenant",
  bot_count: 1,
  connected: true,
  has_admin_api_key: false,
  last_event_at: "2026-09-18T12:00:00Z",
  last_spend_sync_at: null,
  last_spend_sync_error: null,
  created_at: "2026-09-18T12:00:00Z",
  created_by: "admin",
};
const report: SpendReport = {
  source_id: "source-1",
  days: 30,
  start_day: "2026-08-20",
  end_day: "2026-09-18",
  members: [
    {
      vendor_user_id: "vendor-user",
      email: "member@example.test",
      display_name: "Member One",
      litellm_user_id: "user",
      grok_cost_usd: "3.25",
      litellm_gateway_spend_usd: "4.75",
      tokens: 100,
      requests: 8,
    },
  ],
  teams: [
    {
      team_id: null,
      team_name: "Multiple teams",
      grok_cost_usd: "3.25",
      litellm_gateway_spend_usd: "4.75",
      tokens: 100,
      requests: 8,
      teams: [
        { team_id: "platform", team_name: "Platform" },
        { team_id: "security", team_name: "Security" },
      ],
    },
  ],
  totals: { grok_cost_usd: "3.25", litellm_gateway_spend_usd: "4.75", tokens: 100, requests: 8 },
};

class Server {
  source: Source | null = null;
  invalidKey = false;
  agentCount = 1;
  requests: { path: string; method: string; body?: string }[] = [];

  fetch = async (input: string | URL | Request, init?: RequestInit) => {
    const path = new URL(String(input), "http://localhost").pathname;
    const method = init?.method ?? "GET";
    this.requests.push({ path, method, body: typeof init?.body === "string" ? init.body : undefined });
    if (path === root + "/sources" && method === "GET") return Response.json(this.source ? [this.source] : []);
    if (path === root + "/sources" && method === "POST") {
      this.source = { ...source, connected: false, bot_count: 0, last_event_at: null };
      return Response.json(connection);
    }
    if (path === root + "/sources/source-1/admin_api_key" && method === "PUT") {
      if (this.invalidKey) return Response.json({ detail: "Invalid GrokBot Admin API key" }, { status: 400 });
      this.source = { ...source, has_admin_api_key: true, last_spend_sync_at: source.last_event_at };
      return Response.json({ status: "ok", rows: 1 });
    }
    if (path === root + "/sources/source-1/admin_api_key" && method === "DELETE") {
      this.source = source;
      return Response.json({ status: "ok" });
    }
    if (path === root + "/sources/source-1" && method === "DELETE") {
      this.source = null;
      return Response.json({ status: "ok" });
    }
    if (path === root + "/sources/source-1/connection") return Response.json(connection);
    if (path === root + "/spend") return Response.json(report);
    if (path === root + "/agents") {
      const agents = {
        data: Array.from({ length: this.agentCount }, (_, index) => ({
          source_id: "source-1",
          external_id: `bot-${String(index + 1).padStart(3, "0")}`,
          first_seen_at: source.created_at,
          last_seen_at: source.last_event_at,
          request_count: 7,
          last_actor_id: "actor-1",
          last_model: "external-model",
          tools: ["Shell", "ReadFile"],
        })),
        page: 1,
        page_size: 20,
        total: this.agentCount,
      };
      return Response.json(agents);
    }
    throw new Error(`Unexpected request ${method} ${path}`);
  };
}

describe("External Cloud Agents", () => {
  let server: Server;
  beforeEach(() => {
    testQueryClient.clear();
    server = new Server();
    vi.stubGlobal("fetch", server.fetch);
  });
  afterEach(() => {
    vi.unstubAllGlobals();
    testQueryClient.clear();
  });

  it("creates without a form, shows setup credentials, then discovers the connected inventory", async () => {
    renderWithProviders(<ExternalCloudAgents accessToken="admin-token" />);
    fireEvent.click(await screen.findByRole("button", { name: "Add GrokBot" }));
    expect(await screen.findByText("Bearer test-ingest-key")).toBeVisible();
    expect(screen.getByText(/http.*\/grok_bot\/inference_hook\/source-1/)).toBeVisible();
    expect(screen.getByText("Authorization", { exact: true })).toBeVisible();
    expect(screen.getByRole("status")).toHaveTextContent("Waiting for first request…");
    expect(server.requests.find((request) => request.method === "POST")).toEqual({
      path: root + "/sources",
      method: "POST",
      body: undefined,
    });
    server.source = source;
    await act(() => testQueryClient.refetchQueries());
    expect(await screen.findByText("Connected · 1 bot")).toBeVisible();
    expect(await screen.findByText("bot-001")).toBeVisible();
    expect(screen.getByText("Page 1 · 1 bot")).toBeVisible();
    expect(screen.queryByText("Bearer test-ingest-key")).not.toBeInTheDocument();
  });

  it.each([0, 2])("uses plural bot counts for %i bots in the header and pagination", async (count) => {
    server.source = { ...source, bot_count: count };
    server.agentCount = count;
    renderWithProviders(<ExternalCloudAgents accessToken="admin-token" />);
    expect(await screen.findByText(`Connected · ${count} bots`)).toBeVisible();
    expect(await screen.findByText(`Page 1 · ${count} bots`)).toBeVisible();
  });

  it("connects spend, renders member and team costs, then removes the key", async () => {
    server.source = source;
    renderWithProviders(<ExternalCloudAgents accessToken="admin-token" />, { searchParams: { cloud_tab: "spend" } });
    expect(await screen.findByRole("heading", { name: "Add GrokBot spend" })).toBeVisible();
    fireEvent.change(screen.getByLabelText("GrokBot Admin API key"), { target: { value: "vendor-key" } });
    fireEvent.click(screen.getByRole("button", { name: "Connect spend" }));
    expect(await screen.findByText("Member One")).toBeVisible();
    expect(screen.getByText("USD 8.00")).toBeVisible();
    expect(screen.getByText("Period 2026-08-20 to 2026-09-18 (UTC)")).toBeVisible();
    expect(screen.getByText(/Spend covers members and teams only, not individual bots/)).toBeVisible();
    expect(server.requests.find((request) => request.method === "PUT")?.body).toBe(
      JSON.stringify({ api_key: "vendor-key" }),
    );
    fireEvent.click(screen.getByRole("button", { name: "By team" }));
    expect(await screen.findByText("Multiple teams")).toBeVisible();
    expect(screen.getByText("Platform, Security")).toBeVisible();
    fireEvent.click(screen.getByRole("button", { name: "Remove key" }));
    expect(await screen.findByRole("heading", { name: "Add GrokBot spend" })).toBeVisible();
    expect(screen.getByLabelText("GrokBot Admin API key")).toHaveValue("");
  });

  it("shows a vendor 400 inline without pretending that the key was saved", async () => {
    server.source = source;
    server.invalidKey = true;
    renderWithProviders(<ExternalCloudAgents accessToken="admin-token" />, { searchParams: { cloud_tab: "spend" } });
    fireEvent.change(await screen.findByLabelText("GrokBot Admin API key"), { target: { value: "invalid-key" } });
    fireEvent.click(screen.getByRole("button", { name: "Connect spend" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Invalid GrokBot Admin API key");
    expect(screen.queryByRole("button", { name: "Refresh" })).not.toBeInTheDocument();
    expect(server.source.has_admin_api_key).toBe(false);
  });

  it("only removes the connection after confirmation", async () => {
    server.source = source;
    renderWithProviders(<ExternalCloudAgents accessToken="admin-token" />);
    fireEvent.click(await screen.findByRole("button", { name: "Remove", exact: true }));
    expect(await screen.findByRole("alertdialog")).toBeVisible();
    expect(server.requests.some((request) => request.method === "DELETE")).toBe(false);
    fireEvent.click(screen.getByRole("button", { name: "Remove", exact: true }));
    expect(await screen.findByRole("button", { name: "Add GrokBot" })).toBeVisible();
  });
});
