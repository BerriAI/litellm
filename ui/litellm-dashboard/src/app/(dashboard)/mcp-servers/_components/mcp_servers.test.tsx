import React from "react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { QueryClientProvider } from "@tanstack/react-query";
import { NuqsTestingAdapter, type OnUrlUpdateFunction } from "nuqs/adapters/testing";
import MCPServers from "./mcp_servers";
import * as networking from "@/components/networking";
import type { MCPServer } from "@/components/mcp_tools/types";
import { TOOLS_OAUTH_UI_STATE_KEY } from "@/hooks/mcpOAuthUtils";
import { setSecureItem } from "@/utils/secureStorage";
import { EDIT_OAUTH_UI_STATE_KEY } from "./mcp_server_edit";
import {
  act,
  chooseSelectOption,
  fireEvent,
  render,
  renderWithProviders,
  screen,
  testQueryClient,
  waitFor,
  within,
} from "@/../tests/test-utils";

vi.mock("@/components/networking", () => ({
  fetchMCPServers: vi.fn(),
  fetchMCPServerHealth: vi.fn(),
  deleteMCPServer: vi.fn(),
  getProxyBaseUrl: vi.fn().mockReturnValue("http://localhost:4000"),
  fetchMCPClientIp: vi.fn().mockResolvedValue(null),
  getGeneralSettingsCall: vi.fn().mockResolvedValue([]),
  updateConfigFieldSetting: vi.fn().mockResolvedValue(undefined),
  deleteConfigFieldSetting: vi.fn().mockResolvedValue(undefined),
  listMCPUserEnvVarStatus: vi.fn().mockResolvedValue([]),
}));

interface ServerViewStubProps {
  mcpServer: MCPServer;
  isEditing: boolean;
  onBack: () => void;
}

vi.mock("./mcp_server_view", () => ({
  MCPServerView: ({ mcpServer, isEditing, onBack }: ServerViewStubProps) => (
    <section aria-label="server detail">
      <p>viewing {mcpServer.server_name}</p>
      <p>{isEditing ? "settings editable" : "settings read only"}</p>
      <button onClick={onBack}>Back to All Servers</button>
    </section>
  ),
}));

interface EnvVarsModalStubProps {
  server: MCPServer | null;
  open: boolean;
  onClose: () => void;
}

vi.mock("./UserEnvVarsModal", () => ({
  default: ({ server, open, onClose }: EnvVarsModalStubProps) =>
    open && server ? (
      <section aria-label="fill env vars">
        <p>fill fields for {server.server_name}</p>
        <button onClick={onClose}>Close env vars</button>
      </section>
    ) : null,
}));

const BASE_SERVER = {
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

const makeServer = (overrides: Partial<MCPServer> & Pick<MCPServer, "server_id">): MCPServer => ({
  ...BASE_SERVER,
  server_name: overrides.server_id,
  alias: `${overrides.server_id}-alias`,
  ...overrides,
});

const TEAM_A = [{ team_id: "team-a", team_alias: "Team A" }];
const TEAM_B = [{ team_id: "team-b", team_alias: "Team B" }];

const urlServers: MCPServer[] = [
  {
    ...BASE_SERVER,
    server_id: "alpha",
    alias: "alpha-alias",
    server_name: "Alpha",
    teams: TEAM_A,
    mcp_access_groups: ["group-x"],
  },
  {
    ...BASE_SERVER,
    server_id: "bravo",
    alias: "bravo-alias",
    server_name: "Bravo",
    created_at: "2024-01-02T00:00:00Z",
    teams: TEAM_B,
  },
  {
    ...BASE_SERVER,
    server_id: "charlie",
    alias: "charlie-alias",
    server_name: "Charlie",
    created_at: "2024-01-03T00:00:00Z",
    teams: TEAM_A,
  },
];

const defaultProps = {
  accessToken: "123",
  userRole: "Admin",
  userID: "admin-user-id",
};

const lastUrl = (onUrlUpdate: ReturnType<typeof vi.fn<OnUrlUpdateFunction>>) => onUrlUpdate.mock.calls.at(-1)?.[0];

const shownServerNames = () =>
  within(screen.getByTestId("mcp-servers-grid"))
    .getAllByText(/^(Alpha|Bravo|Charlie)$/)
    .map((node) => node.textContent);

const cardFor = (name: string): HTMLElement => {
  const card = screen.getByText(name).closest<HTMLElement>('[role="button"]');
  if (!card) throw new Error(`no card for ${name}`);
  return card;
};

const selectNextTo = (label: string) => within(screen.getByText(label).parentElement!).getByRole("combobox");

const renderPage = (searchParams: string, props: Partial<typeof defaultProps> = {}) => {
  const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
  renderWithProviders(<MCPServers {...defaultProps} {...props} />, { searchParams, onUrlUpdate });
  return onUrlUpdate;
};

const renderPageKeepingMountWrites = (searchParams: string, props: Partial<typeof defaultProps> = {}) => {
  const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
  render(
    <NuqsTestingAdapter
      searchParams={searchParams}
      onUrlUpdate={onUrlUpdate}
      hasMemory
      resetUrlUpdateQueueOnMount={false}
    >
      <QueryClientProvider client={testQueryClient}>
        <MCPServers {...defaultProps} {...props} />
      </QueryClientProvider>
    </NuqsTestingAdapter>,
  );
  return onUrlUpdate;
};

describe("MCPServers", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    testQueryClient.clear();
    window.sessionStorage.clear();
    vi.mocked(networking.fetchMCPServerHealth).mockResolvedValue([]);
  });

  it("should render the MCPServers component with title", async () => {
    vi.mocked(networking.fetchMCPServers).mockResolvedValue([]);

    renderWithProviders(<MCPServers {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText("MCP Servers")).toBeInTheDocument();
    });
  });

  it("should render mocked MCP servers data in the table", async () => {
    const mockServers: MCPServer[] = [
      { ...BASE_SERVER, server_id: "server-1", server_name: "Test Server 1", alias: "test-server-1" },
      {
        ...BASE_SERVER,
        server_id: "server-2",
        server_name: "Test Server 2",
        alias: "test-server-2",
        transport: "sse",
        auth_type: "api_key",
        mcp_access_groups: ["group-1"],
      },
    ];
    vi.mocked(networking.fetchMCPServers).mockResolvedValue(mockServers);

    renderWithProviders(<MCPServers {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText("Test Server 1")).toBeInTheDocument();
    });
    expect(screen.getByText("Test Server 2")).toBeInTheDocument();
    expect(screen.getAllByText("test-server-1").length).toBeGreaterThan(0);
    expect(screen.getAllByText("test-server-2").length).toBeGreaterThan(0);
    // useMCPServers reads the token from the global useAuthorized mock
    expect(networking.fetchMCPServers).toHaveBeenCalledWith("123", undefined);
  });

  it("should fetch and merge health status for servers", async () => {
    vi.mocked(networking.fetchMCPServers).mockResolvedValue([
      makeServer({ server_id: "server-1" }),
      makeServer({ server_id: "server-2" }),
    ]);
    vi.mocked(networking.fetchMCPServerHealth).mockResolvedValue([
      { server_id: "server-1", status: "healthy" },
      { server_id: "server-2", status: "unhealthy" },
    ]);

    renderWithProviders(<MCPServers {...defaultProps} />);

    await waitFor(() => {
      expect(networking.fetchMCPServerHealth).toHaveBeenCalledWith("123");
    });
  });

  it("should display loading state while health check is in progress", async () => {
    vi.mocked(networking.fetchMCPServers).mockResolvedValue([makeServer({ server_id: "server-1" })]);
    vi.mocked(networking.fetchMCPServerHealth).mockImplementation(() => new Promise(() => {}));

    renderWithProviders(<MCPServers {...defaultProps} />);

    await waitFor(() => {
      expect(networking.fetchMCPServerHealth).toHaveBeenCalled();
    });
  });

  it("should filter servers by team when a team is selected", async () => {
    vi.mocked(networking.fetchMCPServers).mockResolvedValue([
      makeServer({
        server_id: "server-1",
        server_name: "Team A Server",
        teams: [{ team_id: "team-a", team_alias: "Team A" }],
      }),
      makeServer({
        server_id: "server-2",
        server_name: "Team B Server",
        teams: [{ team_id: "team-b", team_alias: "Team B" }],
      }),
      makeServer({
        server_id: "server-3",
        server_name: "Team A Server 2",
        teams: [{ team_id: "team-a", team_alias: "Team A" }],
      }),
    ]);

    renderWithProviders(<MCPServers {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText("Team A Server")).toBeInTheDocument();
    });
    expect(screen.getByText("Team B Server")).toBeInTheDocument();
    expect(screen.getByText("Team A Server 2")).toBeInTheDocument();

    await chooseSelectOption(userEvent, selectNextTo("Team"), "Team A");

    await waitFor(() => {
      expect(screen.queryByText("Team B Server")).not.toBeInTheDocument();
    });
    expect(screen.getByText("Team A Server")).toBeInTheDocument();
    expect(screen.getByText("Team A Server 2")).toBeInTheDocument();
  });

  it("should not trigger an extra health check when the server list changes after deletion", async () => {
    // Regression: the health query key must not depend on the server list, or a shorter
    // list after a deletion would trigger a second health fetch.
    const twoServers = [makeServer({ server_id: "server-1" }), makeServer({ server_id: "server-2" })];
    vi.mocked(networking.fetchMCPServers)
      .mockResolvedValueOnce(twoServers)
      .mockResolvedValueOnce(twoServers.slice(0, 1));
    vi.mocked(networking.fetchMCPServerHealth).mockResolvedValue([
      { server_id: "server-1", status: "healthy" },
      { server_id: "server-2", status: "healthy" },
    ]);

    const { rerender } = renderWithProviders(<MCPServers {...defaultProps} />);

    await waitFor(() => {
      expect(networking.fetchMCPServerHealth).toHaveBeenCalledTimes(1);
    });

    await act(async () => {
      await testQueryClient.invalidateQueries({ queryKey: ["mcpServers"] });
    });
    rerender(<MCPServers {...defaultProps} />);

    expect(networking.fetchMCPServerHealth).toHaveBeenCalledTimes(1);
  });

  describe("URL state", () => {
    beforeEach(() => {
      vi.mocked(networking.fetchMCPServers).mockResolvedValue(urlServers);
    });

    it.each([
      ["toolsets", "Toolsets"],
      ["connect", "Connect"],
      ["semantic-filter", "Semantic Filter"],
      ["tool-search", "Tool Search"],
      ["network", "Network Settings"],
      ["submitted", "Submitted MCPs"],
    ])("opens the %s tab from the URL for an admin", (value, name) => {
      renderPage(`?tab=${value}`);

      expect(screen.getByRole("tab", { name })).toHaveAttribute("aria-selected", "true");
    });

    it("opens the tab named in the URL and writes the tab the user picks", async () => {
      const onUrlUpdate = renderPage("?tab=toolsets");

      expect(screen.getByRole("tab", { name: "Toolsets" })).toHaveAttribute("aria-selected", "true");

      await userEvent.click(screen.getByRole("tab", { name: "Network Settings" }));

      expect(lastUrl(onUrlUpdate)?.searchParams.get("tab")).toBe("network");
      expect(screen.getByRole("tab", { name: "Network Settings" })).toHaveAttribute("aria-selected", "true");
    });

    it("drops an admin-only tab from the URL for a non-admin", async () => {
      const onUrlUpdate = renderPageKeepingMountWrites("?tab=submitted", { userRole: "Internal User" });

      expect(screen.getByRole("tab", { name: "All Servers" })).toHaveAttribute("aria-selected", "true");
      expect(screen.queryByRole("tab", { name: "Submitted MCPs" })).not.toBeInTheDocument();
      await waitFor(() => expect(onUrlUpdate).toHaveBeenCalled());
      expect(lastUrl(onUrlUpdate)?.searchParams.has("tab")).toBe(false);
    });

    it("orders the cards by the sort in the URL", async () => {
      renderPage("?sort=name_asc");

      await screen.findByText("Alpha");
      expect(shownServerNames()).toEqual(["Alpha", "Bravo", "Charlie"]);
      expect(selectNextTo("Sort")).toHaveTextContent("Name (A→Z)");
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

      await chooseSelectOption(userEvent, selectNextTo("Sort"), "Name (A→Z)");

      expect(lastUrl(onUrlUpdate)?.searchParams.get("sort")).toBe("name_asc");
      expect(shownServerNames()).toEqual(["Alpha", "Bravo", "Charlie"]);
    });

    it("filters by the team and access group in the URL", async () => {
      renderPage("?team=team-a&access_group=group-x");

      await screen.findByText("Alpha");
      expect(shownServerNames()).toEqual(["Alpha"]);
      expect(selectNextTo("Team")).toHaveTextContent("Team A");
      expect(selectNextTo("Access Group")).toHaveTextContent("group-x");
    });

    it("writes the team and access group the user picks", async () => {
      const onUrlUpdate = renderPage("");
      await screen.findByText("Alpha");

      await chooseSelectOption(userEvent, selectNextTo("Team"), "Team A");
      expect(lastUrl(onUrlUpdate)?.searchParams.get("team")).toBe("team-a");
      expect(shownServerNames()).toEqual(["Charlie", "Alpha"]);

      await chooseSelectOption(userEvent, selectNextTo("Access Group"), "group-x");
      expect(lastUrl(onUrlUpdate)?.searchParams.get("access_group")).toBe("group-x");
      expect(lastUrl(onUrlUpdate)?.searchParams.get("team")).toBe("team-a");
      expect(shownServerNames()).toEqual(["Alpha"]);

      await chooseSelectOption(userEvent, selectNextTo("Team"), "All Servers");
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

    it("says no server matches when a URL filter excludes every server", async () => {
      renderPage("?team=unknown-team");

      expect(await screen.findByText("No servers match the current filters or search.")).toBeInTheDocument();
      expect(screen.queryByText(/No MCP servers configured/)).not.toBeInTheDocument();
    });

    it("says no servers are configured when the proxy has none", async () => {
      vi.mocked(networking.fetchMCPServers).mockResolvedValue([]);
      renderPage("?team=team-a");

      expect(await screen.findByText(/No MCP servers configured/)).toBeInTheDocument();
    });

    it("opens the server in the URL even when the filters hide it", async () => {
      renderPage("?server=bravo&team=team-a");

      expect(await screen.findByText("viewing Bravo")).toBeInTheDocument();
      expect(screen.getByText("settings read only")).toBeInTheDocument();
    });

    it("shows the list instead of a placeholder detail for an unknown server", async () => {
      renderPage("?server=missing");

      await screen.findByText("Alpha");
      expect(screen.queryByRole("region", { name: "server detail" })).not.toBeInTheDocument();
    });

    it("pushes the clicked server and clears it on Back", async () => {
      const onUrlUpdate = renderPage("?team=team-a&tool=stale&tool_search=old");

      await screen.findByText("Charlie");
      await userEvent.click(cardFor("Charlie"));

      expect(await screen.findByText("viewing Charlie")).toBeInTheDocument();
      expect(screen.getByText("settings editable")).toBeInTheDocument();
      const opened = lastUrl(onUrlUpdate);
      expect(opened?.searchParams.get("server")).toBe("charlie");
      expect(opened?.searchParams.has("tool")).toBe(false);
      expect(opened?.searchParams.has("tool_search")).toBe(false);
      expect(opened?.searchParams.get("team")).toBe("team-a");
      expect(opened?.options.history).toBe("push");

      await userEvent.click(screen.getByRole("button", { name: "Back to All Servers" }));

      expect(await screen.findByText("Alpha")).toBeInTheDocument();
      const closed = lastUrl(onUrlUpdate);
      expect(closed?.searchParams.has("server")).toBe(false);
      expect(closed?.searchParams.get("team")).toBe("team-a");
      expect(closed?.options.history).toBe("push");
    });

    it("clears the server tab, tool and tool search on Back but keeps the filters", async () => {
      const onUrlUpdate = renderPage("?server=charlie&server_tab=tools&tool=search_docs&tool_search=se&team=team-a");

      await userEvent.click(await screen.findByRole("button", { name: "Back to All Servers" }));

      expect(await screen.findByText("Alpha")).toBeInTheDocument();
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

      expect(await screen.findByText("viewing Alpha")).toBeInTheDocument();
      expect(lastUrl(onUrlUpdate)?.searchParams.get("server")).toBe("alpha");
      expect(lastUrl(onUrlUpdate)?.searchParams.has("server_tab")).toBe(false);
    });

    it("opens the env vars modal for fill_env_vars once and drops the key", async () => {
      const onUrlUpdate = renderPageKeepingMountWrites("?fill_env_vars=bravo&team=team-a");

      expect(await screen.findByText("fill fields for Bravo")).toBeInTheDocument();
      await waitFor(() => expect(lastUrl(onUrlUpdate)?.searchParams.has("fill_env_vars")).toBe(false));
      expect(lastUrl(onUrlUpdate)?.searchParams.get("team")).toBe("team-a");

      await userEvent.click(screen.getByRole("button", { name: "Close env vars" }));

      expect(screen.queryByRole("region", { name: "fill env vars" })).not.toBeInTheDocument();
    });

    it("returns to the tools tab of the server after a tools OAuth redirect", async () => {
      setSecureItem(TOOLS_OAUTH_UI_STATE_KEY, JSON.stringify({ serverId: "alpha" }));

      const onUrlUpdate = renderPageKeepingMountWrites("");

      expect(await screen.findByText("viewing Alpha")).toBeInTheDocument();
      expect(screen.getByText("settings read only")).toBeInTheDocument();
      expect(lastUrl(onUrlUpdate)?.searchParams.get("server")).toBe("alpha");
      expect(lastUrl(onUrlUpdate)?.searchParams.get("server_tab")).toBe("tools");
      expect(window.sessionStorage.getItem(TOOLS_OAUTH_UI_STATE_KEY)).toBeNull();

      await userEvent.click(screen.getByRole("button", { name: "Back to All Servers" }));

      expect(await screen.findByText("Alpha")).toBeInTheDocument();
      expect(screen.queryByRole("region", { name: "server detail" })).not.toBeInTheDocument();
      expect(lastUrl(onUrlUpdate)?.searchParams.has("server")).toBe(false);
    });

    it("returns to the editable settings of the server after an edit OAuth redirect", async () => {
      setSecureItem(EDIT_OAUTH_UI_STATE_KEY, JSON.stringify({ serverId: "charlie" }));

      const onUrlUpdate = renderPageKeepingMountWrites("");

      expect(await screen.findByText("viewing Charlie")).toBeInTheDocument();
      expect(screen.getByText("settings editable")).toBeInTheDocument();
      expect(lastUrl(onUrlUpdate)?.searchParams.get("server")).toBe("charlie");
      expect(lastUrl(onUrlUpdate)?.searchParams.get("server_tab")).toBe("settings");
      expect(window.sessionStorage.getItem(EDIT_OAUTH_UI_STATE_KEY)).not.toBeNull();
    });

    it("keeps the server in the URL over a stale edit OAuth state and drops that state", async () => {
      setSecureItem(EDIT_OAUTH_UI_STATE_KEY, JSON.stringify({ serverId: "charlie" }));

      const onUrlUpdate = renderPageKeepingMountWrites("?server=bravo");

      expect(await screen.findByText("viewing Bravo")).toBeInTheDocument();
      expect(screen.getByText("settings read only")).toBeInTheDocument();
      expect(onUrlUpdate.mock.calls.some(([update]) => update.searchParams.get("server") === "charlie")).toBe(false);
      expect(window.sessionStorage.getItem(EDIT_OAUTH_UI_STATE_KEY)).toBeNull();
    });

    it("drops an edit OAuth state whose server no longer exists", async () => {
      setSecureItem(EDIT_OAUTH_UI_STATE_KEY, JSON.stringify({ serverId: "deleted" }));

      renderPageKeepingMountWrites("");

      await screen.findByText("Alpha");
      await waitFor(() => expect(window.sessionStorage.getItem(EDIT_OAUTH_UI_STATE_KEY)).toBeNull());
      expect(screen.queryByRole("region", { name: "server detail" })).not.toBeInTheDocument();
    });

    it.each([
      ["not json"],
      [JSON.stringify({ serverId: 7 })],
      [JSON.stringify({})],
      [JSON.stringify({ serverId: "" })],
    ])("ignores the malformed stored OAuth state %s", async (stored) => {
      setSecureItem(TOOLS_OAUTH_UI_STATE_KEY, stored);

      const onUrlUpdate = renderPageKeepingMountWrites("");

      await screen.findByText("Alpha");
      expect(screen.queryByRole("region", { name: "server detail" })).not.toBeInTheDocument();
      expect(onUrlUpdate.mock.calls.some(([update]) => update.searchParams.has("server"))).toBe(false);
    });
  });
});
