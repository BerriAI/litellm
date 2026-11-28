import React from "react";
import { render as renderWithoutNuqs, waitFor, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, it, expect, vi, beforeEach } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { NuqsTestingAdapter, type OnUrlUpdateFunction } from "nuqs/adapters/testing";
import { chooseSelectOption, fireEvent } from "@/../tests/test-utils";
import MCPServers from "./mcp_servers";
import type { MCPServer } from "@/components/mcp_tools/types";
import * as networking from "@/components/networking";
import { TOOLS_OAUTH_UI_STATE_KEY } from "@/hooks/mcpOAuthUtils";
import { setSecureItem } from "@/utils/secureStorage";
import { EDIT_OAUTH_UI_STATE_KEY } from "./mcp_server_edit";

vi.mock("@/components/networking", () => ({
  fetchMCPServers: vi.fn(),
  fetchMCPServerHealth: vi.fn(),
  deleteMCPServer: vi.fn(),
  getProxyBaseUrl: vi.fn().mockReturnValue("http://localhost:4000"),
  getGatewayBaseUrl: vi.fn().mockReturnValue("http://localhost:4000"),
  fetchMCPClientIp: vi.fn().mockResolvedValue(null),
  getGeneralSettingsCall: vi.fn().mockResolvedValue([]),
  updateConfigFieldSetting: vi.fn().mockResolvedValue(undefined),
  deleteConfigFieldSetting: vi.fn().mockResolvedValue(undefined),
  modelHubCall: vi.fn().mockResolvedValue({ data: [] }),
  getMCPUserEnvVars: vi.fn((_accessToken: string, serverId: string) =>
    Promise.resolve({
      server_id: serverId,
      required: [{ name: "API_KEY", description: "API key", is_set: false }],
      missing_count: 1,
    }),
  ),
  storeMCPUserEnvVars: vi.fn(),
  clearMCPUserEnvVars: vi.fn(),
  listMCPUserEnvVarStatus: vi.fn().mockResolvedValue([]),
  listMCPTools: vi.fn().mockResolvedValue({ tools: [] }),
  callMCPTool: vi.fn(),
  getMCPOAuthUserCredentialStatus: vi.fn().mockResolvedValue(null),
  fetchMCPGatewaySessions: vi.fn(),
  terminateMCPGatewaySessions: vi.fn(),
  getUiConfig: vi.fn().mockResolvedValue({}),
  getMCPSemanticFilterSettings: vi.fn().mockResolvedValue({}),
}));

const stubUiConfig = (config: object) =>
  vi.spyOn(globalThis, "fetch").mockImplementation(async () => new Response(JSON.stringify(config), { status: 200 }));

const createQueryClient = () =>
  new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        gcTime: 0,
      },
    },
  });

const URL_BASE_SERVER = {
  url: "https://example.com/mcp",
  transport: "http",
  auth_type: "none",
  created_at: "2024-01-01T00:00:00Z",
  created_by: "user-1",
  updated_at: "2024-01-01T00:00:00Z",
  updated_by: "user-1",
  teams: [],
  mcp_access_groups: [],
};

const urlServers: MCPServer[] = [
  {
    ...URL_BASE_SERVER,
    server_id: "alpha",
    alias: "alpha-alias",
    server_name: "Alpha",
    teams: [{ team_id: "team-a", team_alias: "Team A" }],
    mcp_access_groups: ["group-x"],
  },
  {
    ...URL_BASE_SERVER,
    server_id: "bravo",
    alias: "bravo-alias",
    server_name: "Bravo",
    created_at: "2024-01-02T00:00:00Z",
    teams: [{ team_id: "team-b", team_alias: "Team B" }],
  },
  {
    ...URL_BASE_SERVER,
    server_id: "charlie",
    alias: "charlie-alias",
    server_name: "Charlie",
    created_at: "2024-01-03T00:00:00Z",
    teams: [{ team_id: "team-a", team_alias: "Team A" }],
  },
];

const urlProps = { accessToken: "123", userRole: "Admin", userID: "admin-user-id" };

const lastUrl = (onUrlUpdate: ReturnType<typeof vi.fn<OnUrlUpdateFunction>>) => onUrlUpdate.mock.calls.at(-1)?.[0];

const shownServerNames = () =>
  within(screen.getByTestId("mcp-servers-grid"))
    .getAllByText(/^(Alpha|Bravo|Charlie)$/)
    .map((node) => node.textContent);

const cardFor = (name: string): HTMLElement => within(screen.getByTestId("mcp-servers-grid")).getByText(name);

const openedServer = (name: string) => screen.findByRole("heading", { level: 1, name });

const editForm = () => screen.queryByRole("form", { name: "Edit MCP server" });

const renderPage = (
  searchParams: string,
  props: Partial<typeof urlProps> = {},
  { keepMountWrites = false }: { keepMountWrites?: boolean } = {},
) => {
  const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
  renderWithoutNuqs(
    <NuqsTestingAdapter
      searchParams={searchParams}
      onUrlUpdate={onUrlUpdate}
      hasMemory
      resetUrlUpdateQueueOnMount={!keepMountWrites}
    >
      <QueryClientProvider client={createQueryClient()}>
        <MCPServers {...urlProps} {...props} />
      </QueryClientProvider>
    </NuqsTestingAdapter>,
  );
  return onUrlUpdate;
};

describe("MCPServers URL state", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    stubUiConfig({});
    window.sessionStorage.clear();
    vi.mocked(networking.fetchMCPServers).mockResolvedValue(urlServers);
    vi.mocked(networking.fetchMCPServerHealth).mockResolvedValue([]);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it.each([
    ["toolsets", "Toolsets"],
    ["connect", "Connect"],
    ["semantic-filter", "Semantic Filter"],
    ["tool-search", "Tool Search"],
    ["network-settings", "Network Settings"],
    ["submitted", "Submitted MCPs"],
    ["connections", "Live Connections"],
  ])("opens the %s tab from the URL for an admin", (value, name) => {
    renderPage(`?tab=${value}`);

    expect(screen.getByRole("tab", { name })).toHaveAttribute("aria-selected", "true");
  });

  it("opens the tab named in the URL and writes the tab the user picks", async () => {
    const onUrlUpdate = renderPage("?tab=toolsets");

    expect(screen.getByRole("tab", { name: "Toolsets" })).toHaveAttribute("aria-selected", "true");

    await userEvent.click(screen.getByRole("tab", { name: "Network Settings" }));

    expect(lastUrl(onUrlUpdate)?.searchParams.get("tab")).toBe("network-settings");
    expect(screen.getByRole("tab", { name: "Network Settings" })).toHaveAttribute("aria-selected", "true");
  });

  it("shows All Servers for an admin-only tab link opened by a non-admin and keeps the link", async () => {
    const onUrlUpdate = renderPage("?tab=submitted", { userRole: "Internal User" }, { keepMountWrites: true });

    expect(await screen.findByText("Alpha")).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "All Servers" })).toHaveAttribute("aria-selected", "true");
    expect(screen.queryByRole("tab", { name: "Submitted MCPs" })).not.toBeInTheDocument();
    expect(onUrlUpdate.mock.calls.some(([update]) => !update.searchParams.has("tab"))).toBe(false);
  });

  it("shows All Servers to an org admin who opens a live connections link and keeps the link", async () => {
    const onUrlUpdate = renderPage("?tab=connections", { userRole: "org_admin" }, { keepMountWrites: true });

    expect(await screen.findByText("Alpha")).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "All Servers" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("tab", { name: "Submitted MCPs" })).toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: "Live Connections" })).not.toBeInTheDocument();
    expect(networking.fetchMCPGatewaySessions).not.toHaveBeenCalled();
    expect(onUrlUpdate.mock.calls.some(([update]) => !update.searchParams.has("tab"))).toBe(false);
  });

  it("orders the cards by the sort in the URL", async () => {
    renderPage("?sort=name_asc");

    await screen.findByText("Alpha");
    expect(shownServerNames()).toEqual(["Alpha", "Bravo", "Charlie"]);
  });

  it("falls back to newest first for an unknown sort", async () => {
    renderPage("?sort=bogus");

    await screen.findByText("Alpha");
    expect(shownServerNames()).toEqual(["Charlie", "Bravo", "Alpha"]);
  });

  it("writes the sort the user picks", async () => {
    const onUrlUpdate = renderPage("");
    await screen.findByText("Alpha");
    expect(shownServerNames()).toEqual(["Charlie", "Bravo", "Alpha"]);

    await chooseSelectOption(userEvent, screen.getByText("Recently created"), "Name (A→Z)");

    expect(lastUrl(onUrlUpdate)?.searchParams.get("sort")).toBe("name_asc");
    expect(shownServerNames()).toEqual(["Alpha", "Bravo", "Charlie"]);
  });

  it("filters by the team and access group in the URL", async () => {
    renderPage("?team=team-a&access_group=group-x");

    await screen.findByText("Alpha");
    expect(shownServerNames()).toEqual(["Alpha"]);
    expect(screen.getByRole("combobox", { name: "Team" })).toHaveTextContent("Team A");
    expect(screen.getByRole("combobox", { name: "Access Group" })).toHaveTextContent("group-x");
  });

  it("writes the team and access group the user picks", async () => {
    const onUrlUpdate = renderPage("");
    await screen.findByText("Alpha");

    await chooseSelectOption(userEvent, screen.getByRole("combobox", { name: "Team" }), "Team A");
    expect(lastUrl(onUrlUpdate)?.searchParams.get("team")).toBe("team-a");
    expect(shownServerNames()).toEqual(["Charlie", "Alpha"]);

    await chooseSelectOption(userEvent, screen.getByRole("combobox", { name: "Access Group" }), "group-x");
    expect(lastUrl(onUrlUpdate)?.searchParams.get("access_group")).toBe("group-x");
    expect(lastUrl(onUrlUpdate)?.searchParams.get("team")).toBe("team-a");
    expect(shownServerNames()).toEqual(["Alpha"]);

    await chooseSelectOption(userEvent, screen.getByRole("combobox", { name: "Team" }), "All Servers");
    expect(lastUrl(onUrlUpdate)?.searchParams.has("team")).toBe(false);
  });

  it("searches by the query in the URL and writes what the user types", async () => {
    const onUrlUpdate = renderPage("?search=brav");

    await screen.findByText("Bravo");
    expect(shownServerNames()).toEqual(["Bravo"]);
    const searchBox = screen.getByPlaceholderText("Search by name, alias, URL, or ID");
    expect(searchBox).toHaveValue("brav");

    fireEvent.change(searchBox, { target: { value: "charlie" } });

    await waitFor(() => expect(lastUrl(onUrlUpdate)?.searchParams.get("search")).toBe("charlie"));
    expect(shownServerNames()).toEqual(["Charlie"]);
  });

  it("opens the server in the URL even when the filters hide it", async () => {
    renderPage("?server=bravo&team=team-a");

    expect(await openedServer("Bravo")).toBeInTheDocument();
  });

  it("shows the list instead of a placeholder detail for an unknown server and keeps the link", async () => {
    const onUrlUpdate = renderPage("?server=missing", {}, { keepMountWrites: true });

    await screen.findByText("Alpha");
    expect(screen.queryByRole("button", { name: /Back to All Servers/ })).not.toBeInTheDocument();
    expect(onUrlUpdate.mock.calls.some(([update]) => !update.searchParams.has("server"))).toBe(false);
  });

  it("ignores a server tab link without a server", async () => {
    renderPage("?server_tab=settings");

    expect(await screen.findByTestId("mcp-servers-grid")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Back to All Servers/ })).not.toBeInTheDocument();
  });

  it("leaves a bare URL untouched until the user opens a server", async () => {
    const onUrlUpdate = renderPage("", {}, { keepMountWrites: true });

    await screen.findByText("Alpha");
    await userEvent.click(cardFor("Alpha"));
    await openedServer("Alpha");
    expect(onUrlUpdate).toHaveBeenCalledTimes(1);
  });

  it("shows markup in the search link as literal text", async () => {
    const payload = "<img src=x onerror=alert(1)>";
    renderPage(`?search=${encodeURIComponent(payload)}`);

    expect(await screen.findByText("No servers match the current filters or search.")).toBeInTheDocument();
    expect(screen.getByPlaceholderText("Search by name, alias, URL, or ID")).toHaveValue(payload);
    expect(screen.queryByRole("img", { name: "" })).not.toBeInTheDocument();
  });

  it("opens an admin tab link once the role finishes loading", async () => {
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    const page = (userRole: string) => (
      <NuqsTestingAdapter
        searchParams="?tab=semantic-filter"
        onUrlUpdate={onUrlUpdate}
        hasMemory
        resetUrlUpdateQueueOnMount={false}
      >
        <QueryClientProvider client={createQueryClient()}>
          <MCPServers {...urlProps} userRole={userRole} />
        </QueryClientProvider>
      </NuqsTestingAdapter>
    );
    const { rerender } = renderWithoutNuqs(page("Undefined Role"));

    expect(await screen.findByText("Alpha")).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "All Servers" })).toHaveAttribute("aria-selected", "true");
    expect(networking.getMCPSemanticFilterSettings).not.toHaveBeenCalled();

    rerender(page("Admin"));

    expect(await screen.findByRole("tab", { name: "Semantic Filter" })).toHaveAttribute("aria-selected", "true");
    expect(onUrlUpdate.mock.calls.some(([update]) => !update.searchParams.has("tab"))).toBe(false);
  });

  it("never loads admin-only tab data for an internal user who opens an admin tab link", async () => {
    renderPage("?tab=semantic-filter", { userRole: "Internal User" });

    expect(await screen.findByText("Alpha")).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "All Servers" })).toHaveAttribute("aria-selected", "true");
    expect(networking.getMCPSemanticFilterSettings).not.toHaveBeenCalled();
  });

  it("pushes the clicked server and clears it on Back", async () => {
    const onUrlUpdate = renderPage("?team=team-a&tool=stale&tool_search=old");

    await screen.findByText("Charlie");
    await userEvent.click(cardFor("Charlie"));

    expect(await openedServer("Charlie")).toBeInTheDocument();
    const opened = lastUrl(onUrlUpdate);
    expect(opened?.searchParams.get("server")).toBe("charlie");
    expect(opened?.searchParams.has("tool")).toBe(false);
    expect(opened?.searchParams.has("tool_search")).toBe(false);
    expect(opened?.searchParams.get("team")).toBe("team-a");
    expect(opened?.options.history).toBe("push");

    await userEvent.click(screen.getByRole("button", { name: /Back to All Servers/ }));

    expect(await screen.findByTestId("mcp-servers-grid")).toBeInTheDocument();
    const closed = lastUrl(onUrlUpdate);
    expect(closed?.searchParams.has("server")).toBe(false);
    expect(closed?.searchParams.get("team")).toBe("team-a");
    expect(closed?.options.history).toBe("push");
  });

  it("clears the server tab, tool and tool search on Back but keeps the filters", async () => {
    const onUrlUpdate = renderPage("?server=charlie&server_tab=tools&tool=search_docs&tool_search=se&team=team-a");

    await openedServer("Charlie");
    await userEvent.click(screen.getByRole("button", { name: /Back to All Servers/ }));

    expect(await screen.findByTestId("mcp-servers-grid")).toBeInTheDocument();
    const closed = lastUrl(onUrlUpdate);
    for (const key of ["server", "server_tab", "tool", "tool_search"]) {
      expect(closed?.searchParams.has(key)).toBe(false);
    }
    expect(closed?.searchParams.get("team")).toBe("team-a");
  });

  it("opens a clicked server on its overview tab", async () => {
    const onUrlUpdate = renderPage("?server=missing&server_tab=settings");

    await screen.findByText("Alpha");
    await userEvent.click(cardFor("Alpha"));

    expect(await openedServer("Alpha")).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Overview" })).toHaveAttribute("aria-selected", "true");
    expect(lastUrl(onUrlUpdate)?.searchParams.get("server")).toBe("alpha");
    expect(lastUrl(onUrlUpdate)?.searchParams.has("server_tab")).toBe(false);
  });

  it("returns to the tools tab of the server after a tools OAuth redirect", async () => {
    setSecureItem(TOOLS_OAUTH_UI_STATE_KEY, JSON.stringify({ serverId: "alpha" }));

    const onUrlUpdate = renderPage("", {}, { keepMountWrites: true });

    expect(await openedServer("Alpha")).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "MCP Tools" })).toHaveAttribute("aria-selected", "true");
    expect(lastUrl(onUrlUpdate)?.searchParams.get("server")).toBe("alpha");
    expect(lastUrl(onUrlUpdate)?.searchParams.get("server_tab")).toBe("tools");
    expect(window.sessionStorage.getItem(TOOLS_OAUTH_UI_STATE_KEY)).toBeNull();

    await userEvent.click(screen.getByRole("button", { name: /Back to All Servers/ }));

    expect(await screen.findByTestId("mcp-servers-grid")).toBeInTheDocument();
    expect(lastUrl(onUrlUpdate)?.searchParams.has("server")).toBe(false);
  });

  it("returns to the editable settings of the server after an edit OAuth redirect", async () => {
    setSecureItem(EDIT_OAUTH_UI_STATE_KEY, JSON.stringify({ serverId: "charlie" }));

    const onUrlUpdate = renderPage("", {}, { keepMountWrites: true });

    expect(await openedServer("Charlie")).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Settings" })).toHaveAttribute("aria-selected", "true");
    expect(editForm()).toBeInTheDocument();
    expect(lastUrl(onUrlUpdate)?.searchParams.get("server")).toBe("charlie");
    expect(lastUrl(onUrlUpdate)?.searchParams.get("server_tab")).toBe("settings");
  });

  it("keeps the server in the URL over a stale edit OAuth state and drops that state", async () => {
    setSecureItem(EDIT_OAUTH_UI_STATE_KEY, JSON.stringify({ serverId: "charlie" }));

    const onUrlUpdate = renderPage("?server=bravo", {}, { keepMountWrites: true });

    expect(await openedServer("Bravo")).toBeInTheDocument();
    expect(onUrlUpdate.mock.calls.some(([update]) => update.searchParams.get("server") === "charlie")).toBe(false);
    expect(window.sessionStorage.getItem(EDIT_OAUTH_UI_STATE_KEY)).toBeNull();
  });

  it("drops an edit OAuth state whose server no longer exists", async () => {
    setSecureItem(EDIT_OAUTH_UI_STATE_KEY, JSON.stringify({ serverId: "deleted" }));

    renderPage("", {}, { keepMountWrites: true });

    await screen.findByText("Alpha");
    await waitFor(() => expect(window.sessionStorage.getItem(EDIT_OAUTH_UI_STATE_KEY)).toBeNull());
    expect(screen.queryByRole("button", { name: /Back to All Servers/ })).not.toBeInTheDocument();
  });

  it.each([["not json"], [JSON.stringify({ serverId: 7 })], [JSON.stringify({})], [JSON.stringify({ serverId: "" })]])(
    "ignores the malformed stored OAuth state %s",
    async (stored) => {
      setSecureItem(TOOLS_OAUTH_UI_STATE_KEY, stored);

      const onUrlUpdate = renderPage("", {}, { keepMountWrites: true });

      await screen.findByText("Alpha");
      expect(screen.queryByRole("button", { name: /Back to All Servers/ })).not.toBeInTheDocument();
      expect(onUrlUpdate.mock.calls.some(([update]) => update.searchParams.has("server"))).toBe(false);
    },
  );
});
