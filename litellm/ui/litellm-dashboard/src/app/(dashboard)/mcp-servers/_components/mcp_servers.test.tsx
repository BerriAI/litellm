import React from "react";
import { render as renderWithoutNuqs, waitFor, screen, act, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, it, expect, vi, beforeEach } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { NuqsAdapter } from "nuqs/adapters/react";
import { NuqsTestingAdapter, type OnUrlUpdateFunction } from "nuqs/adapters/testing";
import { chooseSelectOption, fireEvent, renderWithProviders as render } from "@/../tests/test-utils";
import MCPServers, { compareServers } from "./mcp_servers";
import type { McpServerSortKey } from "./useMcpServersUrlState";
import type { MCPServer } from "@/components/mcp_tools/types";
import * as networking from "@/components/networking";
import { TOOLS_OAUTH_UI_STATE_KEY } from "@/hooks/mcpOAuthUtils";
import { setSecureItem } from "@/utils/secureStorage";
import { EDIT_OAUTH_UI_STATE_KEY } from "./mcp_server_edit";

// Mock the networking module
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

describe("compareServers", () => {
  const server = (server_id: string, name: string, created_at = ""): MCPServer => ({
    server_id,
    server_name: name,
    created_at,
    updated_at: created_at,
    created_by: "user",
    updated_by: "user",
  });

  const shuffled = [server("c", "github"), server("a", "slack"), server("b", "Jira")];

  it("orders servers without timestamps by name so config.yaml servers render in a stable order", () => {
    const byCreated = [...shuffled].sort((a, b) => compareServers(a, b, "created_desc")).map((s) => s.server_id);
    const byUpdated = [...shuffled].sort((a, b) => compareServers(a, b, "updated_desc")).map((s) => s.server_id);
    const byHealth = [...shuffled].sort((a, b) => compareServers(a, b, "health")).map((s) => s.server_id);

    expect(byCreated).toEqual(["c", "b", "a"]);
    expect(byUpdated).toEqual(["c", "b", "a"]);
    expect(byHealth).toEqual(["c", "b", "a"]);
  });

  it("keeps newest-first when timestamps differ", () => {
    const newest = server("new", "zzz", "2026-02-01T00:00:00Z");
    const oldest = server("old", "aaa", "2026-01-01T00:00:00Z");
    expect([oldest, newest].sort((a, b) => compareServers(a, b, "created_desc")).map((s) => s.server_id)).toEqual([
      "new",
      "old",
    ]);
  });

  it.each<McpServerSortKey>(["created_desc", "updated_desc", "name_asc", "health"])(
    "breaks equal timestamps and names by ID for %s regardless of input order",
    (sort) => {
      const servers = [
        server("b", "GitHub", "2026-01-01T00:00:00Z"),
        server("c", "Slack", "2026-01-01T00:00:00Z"),
        server("a", "github", "2026-01-01T00:00:00Z"),
      ];
      for (const input of [servers, [...servers].reverse()]) {
        expect([...input].sort((a, b) => compareServers(a, b, sort)).map((s) => s.server_id)).toEqual(["a", "b", "c"]);
      }
    },
  );

  it("uses the display name before alias, then falls back to alias and ID", () => {
    const servers: MCPServer[] = [
      { ...server("s-slack", "Slack"), alias: "aaa" },
      { ...server("s-github", ""), server_name: null, alias: "GitHub" },
      { ...server("confluence", ""), alias: "" },
    ];
    for (const input of [servers, [...servers].reverse()]) {
      expect([...input].sort((a, b) => compareServers(a, b, "name_asc")).map((s) => s.server_id)).toEqual([
        "confluence",
        "s-github",
        "s-slack",
      ]);
    }
  });

  it.each<McpServerSortKey>(["created_desc", "updated_desc", "health"])(
    "keeps timestamped servers before missing timestamps for %s",
    (sort) => {
      const servers = [
        server("config", "aaa"),
        server("older", "bbb", "2026-01-01T00:00:00Z"),
        server("newer", "zzz", "2026-02-01T00:00:00Z"),
      ];
      for (const input of [servers, [...servers].reverse()]) {
        expect([...input].sort((a, b) => compareServers(a, b, sort)).map((s) => s.server_id)).toEqual([
          "newer",
          "older",
          "config",
        ]);
      }
    },
  );

  it("sorts health before recency and display name", () => {
    const servers: MCPServer[] = [
      { ...server("healthy", "aaa", "2026-03-01T00:00:00Z"), status: "healthy" },
      { ...server("reachable", "aaa", "2026-04-01T00:00:00Z"), status: "reachable" },
      { ...server("unknown", "bbb", "2026-02-01T00:00:00Z"), status: "unknown" },
      { ...server("unhealthy", "zzz", "2026-01-01T00:00:00Z"), status: "unhealthy" },
    ];
    expect(servers.sort((a, b) => compareServers(a, b, "health")).map((s) => s.server_id)).toEqual([
      "unhealthy",
      "unknown",
      "reachable",
      "healthy",
    ]);
  });
});

describe("MCPServers", () => {
  const defaultProps = {
    accessToken: "123",
    userRole: "Admin",
    userID: "admin-user-id",
  };

  beforeEach(() => {
    vi.clearAllMocks();
    stubUiConfig({});
    window.history.replaceState(null, "", "/");
  });

  afterEach(() => {
    window.history.replaceState(null, "", "/");
  });

  it("should render the MCPServers component with title", async () => {
    // Mock empty response for simple render test
    vi.mocked(networking.fetchMCPServers).mockResolvedValue([]);

    const queryClient = createQueryClient();
    render(
      <QueryClientProvider client={queryClient}>
        <MCPServers {...defaultProps} />
      </QueryClientProvider>,
    );

    // Wait for the component to load and check if title renders
    await waitFor(() => {
      expect(screen.getByText("MCP Servers")).toBeInTheDocument();
    });

    // Verify the title is rendered
    expect(screen.getByText("MCP Servers")).toBeInTheDocument();
  });

  it("opens the user env vars modal from a deep link and removes only that query parameter", async () => {
    const server: MCPServer = {
      server_id: "deep-link-server",
      server_name: "Deep Link Server",
      alias: "deep-link-server",
      url: "https://example.com/mcp",
      created_at: "",
      updated_at: "",
      created_by: "user",
      updated_by: "user",
    };
    vi.mocked(networking.fetchMCPServers).mockResolvedValue([server]);
    vi.mocked(networking.fetchMCPServerHealth).mockResolvedValue([{ server_id: server.server_id, status: "healthy" }]);
    vi.mocked(networking.listMCPUserEnvVarStatus).mockResolvedValue([
      {
        server_id: server.server_id,
        server_name: server.server_name,
        required: [{ name: "API_KEY", description: "API key", is_set: false }],
        missing_count: 1,
      },
    ]);
    window.history.replaceState(null, "", "/?fill_env_vars=deep-link-server&other=1");
    renderWithoutNuqs(
      <NuqsAdapter>
        <QueryClientProvider client={createQueryClient()}>
          <MCPServers {...defaultProps} />
        </QueryClientProvider>
      </NuqsAdapter>,
    );

    expect(await screen.findByRole("heading", { name: "Set your credentials" })).toBeVisible();
    expect(within(screen.getByRole("dialog")).getByText("Deep Link Server")).toBeVisible();
    await waitFor(() => expect(window.location.search).toBe("?other=1"));
  });

  it.each(["Admin", "Internal User"])("links a %s to their MCP connections page", async (userRole) => {
    vi.mocked(networking.fetchMCPServers).mockResolvedValue([]);

    render(
      <QueryClientProvider client={createQueryClient()}>
        <MCPServers {...defaultProps} userRole={userRole} />
      </QueryClientProvider>,
    );

    const myConnections = await screen.findByRole("link", { name: "My Connections" });
    expect(myConnections).toBeVisible();
    expect(myConnections).toHaveAttribute("href", "/ui/connect");
    for (const name of ["Semantic Filter", "Tool Search", "Network Settings", "Submitted MCPs"]) {
      const tab = screen.queryByRole("tab", { name });
      if (userRole === "Admin") {
        expect(tab).toBeVisible();
      } else {
        expect(tab).not.toBeInTheDocument();
      }
    }
    expect(
      screen.getByRole("button", {
        name: userRole === "Admin" ? "+ Add New MCP Server" : "+ Submit MCP Server",
      }),
    ).toBeVisible();
  });

  it.each(["cancel", "success", "failure", "unnamed"])("preserves delete confirmation on %s", async (outcome) => {
    const server: MCPServer = {
      created_at: "",
      updated_at: "",
      server_id: "delete-server",
      server_name: outcome === "unnamed" ? null : "Delete fixture",
      alias: "delete-alias",
      url: outcome === "unnamed" ? null : "https://example.com/mcp",
      created_by: "user",
      updated_by: "user",
    };
    vi.mocked(networking.fetchMCPServers).mockResolvedValue([server]);
    vi.mocked(networking.fetchMCPServerHealth).mockResolvedValue([]);
    let finishDelete: () => void = () => {};
    vi.mocked(networking.deleteMCPServer).mockImplementation(
      () =>
        new Promise((resolve, reject) => {
          finishDelete = () => (outcome === "failure" ? reject(new Error("Delete failed")) : resolve(undefined));
        }),
    );
    render(
      <QueryClientProvider client={createQueryClient()}>
        <MCPServers {...defaultProps} />
      </QueryClientProvider>,
    );
    await userEvent.click(await screen.findByRole("button", { name: "Server actions" }));
    await userEvent.click(await screen.findByRole("menuitem", { name: "Delete" }));
    const dialog = await screen.findByRole("alertdialog", { name: "Delete MCP Server?" });
    expect(within(dialog).getByText("delete-server")).toBeVisible();
    if (outcome === "unnamed") {
      expect(within(dialog).queryByText("Name")).not.toBeInTheDocument();
      expect(within(dialog).queryByText("URL")).not.toBeInTheDocument();
    } else {
      expect(within(dialog).getByText("Delete fixture")).toBeVisible();
      expect(within(dialog).getByText("https://example.com/mcp")).toBeVisible();
    }
    if (outcome === "cancel") {
      await userEvent.click(within(dialog).getByRole("button", { name: "Cancel" }));
      expect(networking.deleteMCPServer).not.toHaveBeenCalled();
    } else {
      await userEvent.click(within(dialog).getByRole("button", { name: "Delete" }));
      expect(within(dialog).getByRole("button", { name: "Deleting..." })).toBeDisabled();
      expect(within(dialog).getByRole("button", { name: "Cancel" })).toBeDisabled();
      expect(networking.deleteMCPServer).toHaveBeenCalledWith("123", "delete-server");
      await act(async () => finishDelete());
    }
    await waitFor(() => expect(screen.queryByRole("alertdialog")).not.toBeInTheDocument());
  });

  it("filters servers by access group", async () => {
    const server = { created_by: "user", updated_by: "user" };
    vi.mocked(networking.fetchMCPServers).mockResolvedValue([
      {
        ...server,
        server_id: "string-group",
        server_name: "String group",
        alias: "string-alias",
        mcp_access_groups: ["shared"],
      },
      {
        ...server,
        server_id: "legacy-group",
        server_name: "Legacy group",
        alias: "legacy-alias",
        mcp_access_groups: ["shared"],
      },
      {
        ...server,
        server_id: "other-group",
        server_name: "Other group",
        alias: "other-alias",
        mcp_access_groups: ["different"],
      },
    ]);
    vi.mocked(networking.fetchMCPServerHealth).mockResolvedValue([]);
    render(
      <QueryClientProvider client={createQueryClient()}>
        <MCPServers {...defaultProps} />
      </QueryClientProvider>,
    );
    await screen.findByText("String group");
    await userEvent.click(screen.getByRole("combobox", { name: "Access Group" }));
    await userEvent.click(await screen.findByRole("option", { name: "shared" }));
    expect(screen.getByText("String group")).toBeVisible();
    expect(screen.getByText("Legacy group")).toBeVisible();
    expect(screen.queryByText("Other group")).not.toBeInTheDocument();
  });

  it.each(["server_name", "alias", "url", "server_id"] as const)("searches by %s case-insensitively", async (field) => {
    const server: MCPServer = {
      created_at: "",
      updated_at: "",
      server_id: "search-server",
      server_name: "Search fixture",
      created_by: "user",
      updated_by: "user",
      [field]: "Needle",
    };
    vi.mocked(networking.fetchMCPServers).mockResolvedValue([server]);
    vi.mocked(networking.fetchMCPServerHealth).mockResolvedValue([]);
    render(
      <QueryClientProvider client={createQueryClient()}>
        <MCPServers {...defaultProps} />
      </QueryClientProvider>,
    );
    await screen.findByTestId("mcp-servers-grid");
    const search = screen.getByPlaceholderText("Search by name, alias, URL, or ID");
    fireEvent.change(search, { target: { value: "  NEEDLE  " } });
    expect(screen.getByTestId("mcp-servers-grid")).toBeVisible();
    fireEvent.change(search, { target: { value: "no-match" } });
    expect(screen.queryByTestId("mcp-servers-grid")).not.toBeInTheDocument();
    expect(screen.getByText("No servers match the current filters or search.")).toBeVisible();
  });

  it.each([
    [false, true],
    [true, false],
  ])("marks stdio servers as disabled only when the proxy reports stdio off (enabled=%s)", async (enabled, flagged) => {
    stubUiConfig({ mcp_stdio_enabled: enabled });
    vi.mocked(networking.fetchMCPServers).mockResolvedValue([
      {
        server_id: "stdio-1",
        server_name: "local_tools",
        alias: "local_tools",
        transport: "stdio",
        command: "python",
        args: ["server.py"],
        created_by: "user",
        updated_by: "user",
      } as MCPServer,
    ]);
    vi.mocked(networking.fetchMCPServerHealth).mockResolvedValue([]);

    render(
      <QueryClientProvider client={createQueryClient()}>
        <MCPServers {...defaultProps} />
      </QueryClientProvider>,
    );

    const grid = await screen.findByTestId("mcp-servers-grid");
    await waitFor(() => expect(globalThis.fetch).toHaveBeenCalled());
    await waitFor(() => expect(within(grid).queryByText("stdio disabled") !== null).toBe(flagged));
    expect(within(grid).getByText("STDIO")).toBeInTheDocument();
  });

  const stdioServer = {
    server_id: "stdio-1",
    server_name: "local_tools",
    alias: "local_tools",
    transport: "stdio",
    command: "python",
    args: ["server.py"],
    created_by: "user",
    updated_by: "user",
  } as MCPServer;

  it("greys out stdio in the create form when the proxy reports stdio off", async () => {
    stubUiConfig({ mcp_stdio_enabled: false });
    vi.mocked(networking.fetchMCPServers).mockResolvedValue([]);
    const user = userEvent.setup();
    render(
      <QueryClientProvider client={createQueryClient()}>
        <MCPServers {...defaultProps} userRole="Internal User" />
      </QueryClientProvider>,
    );
    await waitFor(() => expect(globalThis.fetch).toHaveBeenCalled());

    await user.click(await screen.findByRole("button", { name: "+ Submit MCP Server" }));
    await user.click(await screen.findByRole("combobox", { name: /Transport Type/ }));

    expect(await screen.findByRole("option", { name: /Standard Input\/Output \(stdio\)/ })).toHaveAttribute(
      "data-disabled",
    );
    expect(screen.getByRole("option", { name: /Streamable HTTP/ })).not.toHaveAttribute("data-disabled");

    await user.keyboard("{Escape}");
    await waitFor(() => expect(screen.queryByRole("listbox")).not.toBeInTheDocument());
  });

  it("explains on the edit page why an existing stdio server cannot run when the proxy reports stdio off", async () => {
    stubUiConfig({ mcp_stdio_enabled: false });
    vi.mocked(networking.fetchMCPServers).mockResolvedValue([stdioServer]);
    vi.mocked(networking.fetchMCPServerHealth).mockResolvedValue([]);
    const user = userEvent.setup();
    render(
      <QueryClientProvider client={createQueryClient()}>
        <MCPServers {...defaultProps} />
      </QueryClientProvider>,
    );
    const grid = await screen.findByTestId("mcp-servers-grid");
    await waitFor(() => expect(within(grid).getByText("stdio disabled")).toBeInTheDocument());

    await user.click(within(grid).getAllByText("local_tools")[0]);
    await user.click(await screen.findByRole("tab", { name: "Settings" }));

    expect(await screen.findByText("stdio is disabled on this proxy")).toBeInTheDocument();
  });

  it("should render mocked MCP servers data in the table", async () => {
    // Mock MCP servers data
    const mockServers = [
      {
        server_id: "server-1",
        server_name: "Test Server 1",
        alias: "test-server-1",
        url: "https://example.com/mcp",
        transport: "http",
        auth_type: "none",
        created_at: "2024-01-01T00:00:00Z",
        created_by: "user-1",
        updated_at: "2024-01-01T00:00:00Z",
        updated_by: "user-1",
        teams: [],
        mcp_access_groups: [],
      },
      {
        server_id: "server-2",
        server_name: "Test Server 2",
        alias: "test-server-2",
        url: "https://example2.com/mcp",
        transport: "sse",
        auth_type: "api_key",
        created_at: "2024-01-02T00:00:00Z",
        created_by: "user-2",
        updated_at: "2024-01-02T00:00:00Z",
        updated_by: "user-2",
        teams: [],
        mcp_access_groups: ["group-1"],
      },
    ];

    vi.mocked(networking.fetchMCPServers).mockResolvedValue(mockServers);

    const queryClient = createQueryClient();
    render(
      <QueryClientProvider client={queryClient}>
        <MCPServers {...defaultProps} />
      </QueryClientProvider>,
    );

    // Wait for the component to load
    await waitFor(() => {
      expect(screen.getByText("MCP Servers")).toBeInTheDocument();
    });

    // Wait for the mocked data to render in the table
    await waitFor(() => {
      expect(screen.getByText("Test Server 1")).toBeInTheDocument();
    });

    // Verify the mocked server data is rendered in the table
    expect(screen.getByText("Test Server 1")).toBeInTheDocument();
    expect(screen.getByText("Test Server 2")).toBeInTheDocument();
    expect(screen.getAllByText("test-server-1").length).toBeGreaterThan(0);
    expect(screen.getAllByText("test-server-2").length).toBeGreaterThan(0);

    // Verify the API was called
    // Note: useMCPServers uses useAuthorized() internally, which returns "123" from global mock
    expect(networking.fetchMCPServers).toHaveBeenCalledWith("123", undefined, undefined, undefined);
  });

  it("should fetch and merge health status for servers", async () => {
    // Mock MCP servers data without health status
    const mockServers = [
      {
        server_id: "server-1",
        server_name: "Test Server 1",
        alias: "test-server-1",
        url: "https://example.com/mcp",
        transport: "http",
        auth_type: "none",
        created_at: "2024-01-01T00:00:00Z",
        created_by: "user-1",
        updated_at: "2024-01-01T00:00:00Z",
        updated_by: "user-1",
        teams: [],
        mcp_access_groups: [],
        status: undefined,
      },
      {
        server_id: "server-2",
        server_name: "Test Server 2",
        alias: "test-server-2",
        url: "https://example2.com/mcp",
        transport: "sse",
        auth_type: "api_key",
        created_at: "2024-01-02T00:00:00Z",
        created_by: "user-2",
        updated_at: "2024-01-02T00:00:00Z",
        updated_by: "user-2",
        teams: [],
        mcp_access_groups: ["group-1"],
        status: undefined,
      },
    ];

    // Mock health status data
    const mockHealthStatuses = [
      { server_id: "server-1", status: "healthy" },
      { server_id: "server-2", status: "unhealthy" },
    ];

    vi.mocked(networking.fetchMCPServers).mockResolvedValue(mockServers);
    vi.mocked(networking.fetchMCPServerHealth).mockResolvedValue(mockHealthStatuses);

    const queryClient = createQueryClient();
    render(
      <QueryClientProvider client={queryClient}>
        <MCPServers {...defaultProps} />
      </QueryClientProvider>,
    );

    // Wait for the component to load
    await waitFor(() => {
      expect(screen.getByText("MCP Servers")).toBeInTheDocument();
    });

    // Verify the health check API was called (without a server ID filter — the hook always
    // fetches health for all servers so the query key stays stable)
    await waitFor(() => {
      expect(networking.fetchMCPServerHealth).toHaveBeenCalledWith("123");
    });
  });

  it("should display loading state while health check is in progress", async () => {
    const mockServers = [
      {
        server_id: "server-1",
        server_name: "Test Server 1",
        alias: "test-server-1",
        url: "https://example.com/mcp",
        transport: "http",
        auth_type: "none",
        created_at: "2024-01-01T00:00:00Z",
        created_by: "user-1",
        updated_at: "2024-01-01T00:00:00Z",
        updated_by: "user-1",
        teams: [],
        mcp_access_groups: [],
      },
    ];

    vi.mocked(networking.fetchMCPServers).mockResolvedValue(mockServers);
    // Mock health check to never resolve (to test loading state)
    vi.mocked(networking.fetchMCPServerHealth).mockImplementation(
      () => new Promise(() => {}), // Never resolves
    );

    const queryClient = createQueryClient();
    render(
      <QueryClientProvider client={queryClient}>
        <MCPServers {...defaultProps} />
      </QueryClientProvider>,
    );

    // Wait for the component to load
    await waitFor(() => {
      expect(screen.getByText("MCP Servers")).toBeInTheDocument();
    });

    // Verify that health check was initiated
    await waitFor(() => {
      expect(networking.fetchMCPServerHealth).toHaveBeenCalled();
    });
  });

  it("should filter servers by team when a team is selected", async () => {
    // Mock MCP servers with different teams
    const mockServers = [
      {
        server_id: "server-1",
        server_name: "Team A Server",
        alias: "team-a-server",
        url: "https://example.com/mcp",
        transport: "http",
        auth_type: "none",
        created_at: "2024-01-01T00:00:00Z",
        created_by: "user-1",
        updated_at: "2024-01-01T00:00:00Z",
        updated_by: "user-1",
        teams: [{ team_id: "team-a", team_alias: "Team A" }],
        mcp_access_groups: [],
      },
      {
        server_id: "server-2",
        server_name: "Team B Server",
        alias: "team-b-server",
        url: "https://example2.com/mcp",
        transport: "sse",
        auth_type: "api_key",
        created_at: "2024-01-02T00:00:00Z",
        created_by: "user-2",
        updated_at: "2024-01-02T00:00:00Z",
        updated_by: "user-2",
        teams: [{ team_id: "team-b", team_alias: "Team B" }],
        mcp_access_groups: [],
      },
      {
        server_id: "server-3",
        server_name: "Team A Server 2",
        alias: "team-a-server-2",
        url: "https://example3.com/mcp",
        transport: "http",
        auth_type: "none",
        created_at: "2024-01-03T00:00:00Z",
        created_by: "user-1",
        updated_at: "2024-01-03T00:00:00Z",
        updated_by: "user-1",
        teams: [{ team_id: "team-a", team_alias: "Team A" }],
        mcp_access_groups: [],
      },
    ];

    vi.mocked(networking.fetchMCPServers).mockResolvedValue(mockServers);
    vi.mocked(networking.fetchMCPServerHealth).mockResolvedValue([]);

    const queryClient = createQueryClient();
    render(
      <QueryClientProvider client={queryClient}>
        <MCPServers {...defaultProps} />
      </QueryClientProvider>,
    );

    // Wait for the component to load
    await waitFor(() => {
      expect(screen.getByText("MCP Servers")).toBeInTheDocument();
    });

    // Wait for servers to be rendered
    await waitFor(() => {
      expect(screen.getByText("Team A Server")).toBeInTheDocument();
    });

    // Verify all servers are initially displayed
    expect(screen.getByText("Team A Server")).toBeInTheDocument();
    expect(screen.getByText("Team B Server")).toBeInTheDocument();
    expect(screen.getByText("Team A Server 2")).toBeInTheDocument();

    await chooseSelectOption(userEvent, screen.getByRole("combobox", { name: "Team" }), "Team A");

    // Wait for filtering to complete
    await waitFor(() => {
      // Team A servers should still be visible
      expect(screen.getByText("Team A Server")).toBeInTheDocument();
      expect(screen.getByText("Team A Server 2")).toBeInTheDocument();
    });

    // Team B server should not be visible
    expect(screen.queryByText("Team B Server")).not.toBeInTheDocument();
  });

  it("should not trigger an extra health check when the server list changes after deletion", async () => {
    // Regression test: previously useMCPServerHealth received serverIds derived from the
    // server list. Deleting a server changed serverIds, which changed the React Query key,
    // which caused a new health check request for every remaining server.
    //
    // Fix: useMCPServerHealth uses a stable, argument-free query key. The component
    // re-rendering with a shorter server list must NOT produce a second health fetch.
    const twoServers = [
      {
        server_id: "server-1",
        server_name: "Test Server 1",
        alias: "test-server-1",
        url: "https://example.com/mcp",
        transport: "http",
        auth_type: "none",
        created_at: "2024-01-01T00:00:00Z",
        created_by: "user-1",
        updated_at: "2024-01-01T00:00:00Z",
        updated_by: "user-1",
        teams: [],
        mcp_access_groups: [],
      },
      {
        server_id: "server-2",
        server_name: "Test Server 2",
        alias: "test-server-2",
        url: "https://example2.com/mcp",
        transport: "sse",
        auth_type: "api_key",
        created_at: "2024-01-02T00:00:00Z",
        created_by: "user-2",
        updated_at: "2024-01-02T00:00:00Z",
        updated_by: "user-2",
        teams: [],
        mcp_access_groups: [],
      },
    ];
    const oneServer = twoServers.slice(0, 1);

    // First call returns two servers; second (after deletion) returns one
    vi.mocked(networking.fetchMCPServers).mockResolvedValueOnce(twoServers).mockResolvedValueOnce(oneServer);
    vi.mocked(networking.fetchMCPServerHealth).mockResolvedValue([
      { server_id: "server-1", status: "healthy" },
      { server_id: "server-2", status: "healthy" },
    ]);

    // Use a shared queryClient with a non-zero gcTime so cached health data survives
    // the re-render triggered by the server list refresh
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false, gcTime: 60_000 } },
    });

    const { rerender } = render(
      <QueryClientProvider client={queryClient}>
        <MCPServers {...defaultProps} />
      </QueryClientProvider>,
    );

    // Wait for the initial health fetch to complete
    await waitFor(() => {
      expect(networking.fetchMCPServerHealth).toHaveBeenCalledTimes(1);
    });

    // Simulate what happens after a server is deleted: the server list query is
    // refetched (returns oneServer), causing the component to re-render with the
    // shorter list.
    await act(async () => {
      await queryClient.invalidateQueries({ queryKey: ["mcpServers"] });
    });

    rerender(
      <QueryClientProvider client={queryClient}>
        <MCPServers {...defaultProps} />
      </QueryClientProvider>,
    );

    // The server list refresh must NOT trigger a second health check
    expect(networking.fetchMCPServerHealth).toHaveBeenCalledTimes(1);
  });

  const liveSessionsReport = {
    worker_pid: 4242,
    total_sessions: 1,
    by_client: [{ label: "claude-code", count: 1 }],
    by_user: [{ label: "alice", count: 1 }],
    sessions: [
      {
        session_id_prefix: "aaaa1111",
        client_name: "claude-code",
        client_version: "1.0.0",
        user_id: "alice",
        user_email: "alice@example.com",
        key_alias: "alice-key",
        team_id: null,
        team_alias: null,
        client_ip: "10.0.0.1",
        idle_seconds: 5,
        in_flight_requests: 0,
      },
    ],
  };

  const openLiveConnections = async (props: { isViewOnly?: boolean }) => {
    vi.mocked(networking.fetchMCPServers).mockResolvedValue([]);
    vi.mocked(networking.fetchMCPGatewaySessions).mockResolvedValue(liveSessionsReport);
    render(
      <QueryClientProvider client={createQueryClient()}>
        <MCPServers {...defaultProps} {...props} />
      </QueryClientProvider>,
    );
    await userEvent.click(await screen.findByRole("tab", { name: "Live Connections" }));
    return within(await screen.findByRole("region", { name: "Live sessions" })).getByRole("row", { name: /aaaa1111/ });
  };

  it("lets a full admin disconnect a live session", async () => {
    const row = await openLiveConnections({ isViewOnly: false });
    expect(within(row).getByRole("button", { name: "Disconnect session aaaa1111" })).toBeInTheDocument();
  });

  it("shows live sessions to a view-only admin session without any disconnect control", async () => {
    const row = await openLiveConnections({ isViewOnly: true });
    expect(row).toHaveTextContent("alice@example.com");
    expect(within(row).queryByRole("button", { name: /^Disconnect/ })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /^Disconnect all/ })).not.toBeInTheDocument();
  });
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

const cardFor = (name: string): HTMLElement => {
  const card = within(screen.getByTestId("mcp-servers-grid")).getByText(name).closest<HTMLElement>('[role="button"]');
  if (!card) throw new Error(`no card for ${name}`);
  return card;
};

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
    ["network", "Network Settings"],
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

    expect(lastUrl(onUrlUpdate)?.searchParams.get("tab")).toBe("network");
    expect(screen.getByRole("tab", { name: "Network Settings" })).toHaveAttribute("aria-selected", "true");
  });

  it("drops an admin-only tab from the URL for a non-admin", async () => {
    const onUrlUpdate = renderPage("?tab=submitted", { userRole: "Internal User" }, { keepMountWrites: true });

    expect(screen.getByRole("tab", { name: "All Servers" })).toHaveAttribute("aria-selected", "true");
    expect(screen.queryByRole("tab", { name: "Submitted MCPs" })).not.toBeInTheDocument();
    await waitFor(() => expect(onUrlUpdate).toHaveBeenCalled());
    expect(lastUrl(onUrlUpdate)?.searchParams.has("tab")).toBe(false);
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

  it("shows the list instead of a placeholder detail for an unknown server", async () => {
    renderPage("?server=missing");

    await screen.findByText("Alpha");
    expect(screen.queryByRole("button", { name: /Back to All Servers/ })).not.toBeInTheDocument();
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
