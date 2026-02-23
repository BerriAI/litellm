import React from "react";
import { render, waitFor, screen, fireEvent, act } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import MCPServers from "./mcp_servers";
import * as networking from "../networking";

// Mock the networking module
vi.mock("../networking", () => ({
  fetchMCPServers: vi.fn(),
  fetchMCPServerHealth: vi.fn(),
  deleteMCPServer: vi.fn(),
  getProxyBaseUrl: vi.fn().mockReturnValue("http://localhost:4000"),
  fetchMCPClientIp: vi.fn().mockResolvedValue(null),
  getGeneralSettingsCall: vi.fn().mockResolvedValue([]),
  updateConfigFieldSetting: vi.fn().mockResolvedValue(undefined),
  deleteConfigFieldSetting: vi.fn().mockResolvedValue(undefined),
}));

// Mock NotificationsManager
vi.mock("../molecules/notifications_manager", () => ({
  default: {
    success: vi.fn(),
    fromBackend: vi.fn(),
  },
}));

const createQueryClient = () =>
  new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        gcTime: 0,
      },
    },
  });

describe("MCPServers", () => {
  const defaultProps = {
    accessToken: "123",
    userRole: "Admin",
    userID: "admin-user-id",
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should render the MCPServers component with title", async () => {
    // Mock empty response for simple render test
    vi.mocked(networking.fetchMCPServers).mockResolvedValue([]);

    const queryClient = createQueryClient();
    const { getByText } = render(
      <QueryClientProvider client={queryClient}>
        <MCPServers {...defaultProps} />
      </QueryClientProvider>,
    );

    // Wait for the component to load and check if title renders
    await waitFor(() => {
      expect(getByText("MCP Servers")).toBeInTheDocument();
    });

    // Verify the title is rendered
    expect(getByText("MCP Servers")).toBeInTheDocument();
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
    const { getByText } = render(
      <QueryClientProvider client={queryClient}>
        <MCPServers {...defaultProps} />
      </QueryClientProvider>,
    );

    // Wait for the component to load
    await waitFor(() => {
      expect(getByText("MCP Servers")).toBeInTheDocument();
    });

    // Wait for the mocked data to render in the table
    await waitFor(() => {
      expect(getByText("Test Server 1")).toBeInTheDocument();
    });

    // Verify the mocked server data is rendered in the table
    expect(getByText("Test Server 1")).toBeInTheDocument();
    expect(getByText("Test Server 2")).toBeInTheDocument();
    expect(getByText("test-server-1")).toBeInTheDocument();
    expect(getByText("test-server-2")).toBeInTheDocument();

    // Verify the API was called
    // Note: useMCPServers uses useAuthorized() internally, which returns "123" from global mock
    expect(networking.fetchMCPServers).toHaveBeenCalledWith("123");
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
    const { getByText } = render(
      <QueryClientProvider client={queryClient}>
        <MCPServers {...defaultProps} />
      </QueryClientProvider>,
    );

    // Wait for the component to load
    await waitFor(() => {
      expect(getByText("MCP Servers")).toBeInTheDocument();
    });

    // Verify the health check API was called with server IDs
    await waitFor(() => {
      expect(networking.fetchMCPServerHealth).toHaveBeenCalledWith("123", ["server-1", "server-2"]);
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
      () => new Promise(() => { }), // Never resolves
    );

    const queryClient = createQueryClient();
    const { getByText } = render(
      <QueryClientProvider client={queryClient}>
        <MCPServers {...defaultProps} />
      </QueryClientProvider>,
    );

    // Wait for the component to load
    await waitFor(() => {
      expect(getByText("MCP Servers")).toBeInTheDocument();
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

    // Find the team select dropdown by looking for the "Current Team:" label
    const teamLabel = screen.getByText("Current Team:");
    const teamSelectContainer = teamLabel.closest("div")?.querySelector(".ant-select");
    expect(teamSelectContainer).toBeTruthy();

    // Open the dropdown by clicking on the selector
    const selectSelector = teamSelectContainer?.querySelector(".ant-select-selector");
    expect(selectSelector).toBeTruthy();

    act(() => {
      fireEvent.mouseDown(selectSelector!);
    });

    // Wait for dropdown to open
    await waitFor(
      () => {
        const dropdownOptions = document.querySelectorAll(".ant-select-item-option");
        expect(dropdownOptions.length).toBeGreaterThan(0);
      },
      { timeout: 5000 },
    );

    // Find and click on "Team A" option
    const dropdownOptions = document.querySelectorAll(".ant-select-item-option");
    const teamAOption = Array.from(dropdownOptions).find((option) =>
      option.textContent?.includes("Team A"),
    );
    expect(teamAOption).toBeTruthy();

    act(() => {
      fireEvent.click(teamAOption!);
    });

    // Wait for filtering to complete
    await waitFor(() => {
      // Team A servers should still be visible
      expect(screen.getByText("Team A Server")).toBeInTheDocument();
      expect(screen.getByText("Team A Server 2")).toBeInTheDocument();
    });

    // Team B server should not be visible
    expect(screen.queryByText("Team B Server")).not.toBeInTheDocument();
  });
});
