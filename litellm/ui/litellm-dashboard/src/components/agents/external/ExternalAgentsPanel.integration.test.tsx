import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders, testQueryClient } from "../../../../tests/test-utils";
import { apiClient } from "@/components/networking";
import ExternalAgentsPanel from "./ExternalAgentsPanel";
import type { Bot, Decision, Source } from "./useExternalAgents";

vi.mock("@/components/networking", () => ({
  apiClient: { get: vi.fn(), request: vi.fn() },
  getProxyBaseUrl: () => "https://proxy.example.test",
  getRuntimeBackendBaseUrl: () => null,
}));

const source: Source = {
  source_id: "source-1",
  display_name: "Company bots",
  vendor: "grok_bot",
  mode: "shadow",
  last_event_at: "2026-01-01T00:00:00Z",
};
const bots: Bot[] = ["Support bot", "Research bot"].map((name, index) => ({
  id: `bot-${index}`,
  source_id: source.source_id,
  external_agent_id: name,
  display_name: name,
  owner_inferred: false,
  status: "active",
  first_seen_at: "2026-01-01T00:00:00Z",
  last_seen_at: "2026-01-01T00:00:00Z",
  request_count: 5,
  issues: ["no team", "no owner", "unmapped member"],
}));
const summary = {
  discovered: 2,
  covered_by_team_policy: 0,
  needs_owner: 2,
  unmapped_actors: 2,
  denied_24h: 0,
  last_event_at: null,
  needs_attention: 2,
};

beforeEach(() => {
  testQueryClient.clear();
  vi.resetAllMocks();
  vi.mocked(apiClient.get).mockImplementation(async (path) => {
    if (path.endsWith("/sources")) return [source];
    if (path.endsWith("/summary")) return summary;
    if (path.endsWith("/connection"))
      return { ...source, ingest_key: "litellm_ingest_test", hook_path: "/grok_bot/inference_hook/source-1" };
    if (path.endsWith("/policies") || path.endsWith("/decisions")) return [];
    return bots;
  });
  vi.mocked(apiClient.request).mockResolvedValue({ updated: 1 });
});

describe("external agent administration", () => {
  it("saves the optional admin key and displays the automatic member sync result", async () => {
    vi.mocked(apiClient.request).mockResolvedValue({
      ...source,
      sync_result: { matched: 2, unmatched: 1, already_mapped: 1 },
    });
    renderWithProviders(<ExternalAgentsPanel token="test-token" teams={[]} />);
    fireEvent.click(await screen.findByRole("button", { name: "Show setup" }));
    fireEvent.click(screen.getByText("Auto-match members (recommended)"));
    const form = screen.getByRole("form", { name: "Auto-match members (recommended)" });
    fireEvent.change(within(form).getByLabelText("GrokBot Admin API key"), { target: { value: "admin-key" } });
    fireEvent.click(within(form).getByRole("button", { name: "Save", exact: true }));
    expect(await screen.findByText("Matched 3 of 4 members")).toBeInTheDocument();
    expect(apiClient.request).toHaveBeenCalledWith(
      "PATCH",
      "/v1/external_agents/sources/source-1",
      expect.objectContaining({ body: { admin_api_key: "admin-key" } }),
    );
  });
  it("shows owner labels with user IDs as fallback", async () => {
    vi.mocked(apiClient.get).mockImplementation(async (path) => {
      if (path.endsWith("/sources")) return [source];
      if (path.endsWith("/summary")) return summary;
      return [
        { ...bots[0], owner_user_id: "owner-email-id", owner_display_name: "owner@example.test" },
        { ...bots[1], owner_user_id: "owner-without-label" },
      ];
    });
    renderWithProviders(<ExternalAgentsPanel token="test-token" teams={[]} />);
    expect(await screen.findByText("owner@example.test")).toBeInTheDocument();
    expect(screen.getByText("owner-without-label")).toBeInTheDocument();
    expect(screen.queryByText("owner-email-id")).not.toBeInTheDocument();
  });

  it("shows decision bot and member labels, falls back to IDs, and renders empty tools as a dash", async () => {
    const decision: Decision = {
      request_id: "request-1",
      source_id: source.source_id,
      external_agent_id: "external-bot-1",
      actor_id: "external-member-1",
      user_id: "user-1",
      user_display_name: "Member alias",
      agent_display_name: "Friendly bot",
      frame_type: "prompt",
      tool_names: [],
      verdict: "allow",
      would_verdict: "allow",
      enforced: false,
      created_at: "2026-01-01T00:00:00Z",
    };
    vi.mocked(apiClient.get).mockImplementation(async (path) => {
      if (path.endsWith("/sources")) return [source];
      if (path.endsWith("/summary")) return summary;
      return [
        decision,
        {
          ...decision,
          request_id: "request-2",
          external_agent_id: "bot-fallback",
          agent_display_name: null,
          user_display_name: null,
          user_id: "user-fallback",
          tool_names: ["Shell"],
        },
        { ...decision, request_id: "request-3", user_display_name: null, user_id: null, actor_id: "actor-fallback" },
      ];
    });
    renderWithProviders(<ExternalAgentsPanel token="test-token" teams={[]} />, {
      searchParams: { external_tab: "decisions" },
    });
    expect(await screen.findByText("Member alias")).toBeInTheDocument();
    expect(screen.getAllByText("Friendly bot")).toHaveLength(2);
    expect(screen.getByText("bot-fallback")).toBeInTheDocument();
    expect(screen.getByText("user-fallback")).toBeInTheDocument();
    expect(screen.getByText("actor-fallback")).toBeInTheDocument();
    expect(screen.getAllByRole("cell", { name: "-", exact: true })).toHaveLength(2);
    expect(screen.getByText("Shell")).toBeInTheDocument();
  });

  it("creates without input and shows endpoint and custom header setup", async () => {
    let connected = false;
    vi.mocked(apiClient.get).mockImplementation(async (path) => {
      if (path.endsWith("/sources")) return connected ? [{ ...source, last_event_at: null }] : [];
      if (path.endsWith("/summary")) return summary;
      if (path.endsWith("/connection")) return { ...source, ingest_key: "litellm_ingest_test" };
      return bots;
    });
    vi.mocked(apiClient.request).mockImplementation(async () => {
      connected = true;
      return source;
    });
    renderWithProviders(<ExternalAgentsPanel token="test-token" teams={[]} />);
    expect(await screen.findByText("Govern every bot, without manual registration")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Add GrokBot" }));
    expect(await screen.findByLabelText("Endpoint URL")).toHaveValue(
      "https://proxy.example.test/grok_bot/inference_hook/source-1",
    );
    expect(screen.getByLabelText("Custom Header name")).toHaveValue("Authorization");
    expect(screen.getByLabelText("Custom Header value")).toHaveValue("Bearer litellm_ingest_test");
    expect(apiClient.request).toHaveBeenCalledWith(
      "POST",
      "/v1/external_agents/sources",
      expect.objectContaining({
        body: {},
      }),
    );
  });

  it("defaults to exceptions and sends exceptions_only when toggled off", async () => {
    renderWithProviders(<ExternalAgentsPanel token="test-token" teams={[]} />);
    const toggle = await screen.findByRole("switch", { name: "Needs attention" });
    expect(toggle).toBeChecked();
    expect(apiClient.get).toHaveBeenCalledWith(
      "/v1/external_agents",
      expect.objectContaining({ query: expect.objectContaining({ exceptions_only: true }) }),
    );
    fireEvent.click(toggle);
    await waitFor(() =>
      expect(apiClient.get).toHaveBeenCalledWith(
        "/v1/external_agents",
        expect.objectContaining({ query: expect.objectContaining({ exceptions_only: false }) }),
      ),
    );
  });

  it("assigns a team only to the selected bot IDs", async () => {
    renderWithProviders(
      <ExternalAgentsPanel token="test-token" teams={[{ team_id: "team-1", team_alias: "Support" }]} />,
    );
    fireEvent.click(await screen.findByRole("checkbox", { name: "Select Support bot" }));
    const bar = screen.getByRole("region", { name: "Bulk actions" });
    expect(bar).toHaveTextContent("1 selected");
    fireEvent.click(within(bar).getByRole("button", { name: "Assign team" }));
    const dialog = screen.getByRole("dialog", { name: "Assign team" });
    fireEvent.change(within(dialog).getByLabelText("Team"), { target: { value: "team-1" } });
    fireEvent.click(within(dialog).getByRole("button", { name: "Apply assignment" }));
    await waitFor(() =>
      expect(apiClient.request).toHaveBeenCalledWith(
        "PATCH",
        "/v1/external_agents/bulk",
        expect.objectContaining({
          body: { agent_ids: ["bot-0"], team_id: "team-1" },
        }),
      ),
    );
  });

  it("maps CSV members to the selected source and rejects malformed rows before posting", async () => {
    renderWithProviders(<ExternalAgentsPanel token="test-token" teams={[]} />, {
      searchParams: { external_tab: "policies" },
    });
    fireEvent.click(await screen.findByRole("button", { name: "Map members" }));
    fireEvent.change(screen.getByLabelText("Member mappings"), { target: { value: "member-only" } });
    fireEvent.click(screen.getByRole("button", { name: "Save mappings" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Line 1 must contain");
    expect(apiClient.request).not.toHaveBeenCalled();
    fireEvent.change(screen.getByLabelText("Member mappings"), { target: { value: "member-one,user-one" } });
    fireEvent.click(screen.getByRole("button", { name: "Save mappings" }));
    await waitFor(() =>
      expect(apiClient.request).toHaveBeenCalledWith(
        "POST",
        "/v1/external_agents/actors/bulk",
        expect.objectContaining({
          body: { source_id: source.source_id, mappings: [{ external_actor_id: "member-one", user_id: "user-one" }] },
        }),
      ),
    );
  });

  it("displays a source creation failure without asking for credentials", async () => {
    vi.mocked(apiClient.get).mockResolvedValue([]);
    vi.mocked(apiClient.request).mockRejectedValue(new Error("Connection unavailable"));
    renderWithProviders(<ExternalAgentsPanel token="test-token" teams={[]} />);
    fireEvent.click(await screen.findByRole("button", { name: "Add GrokBot" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Connection unavailable");
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });
});
