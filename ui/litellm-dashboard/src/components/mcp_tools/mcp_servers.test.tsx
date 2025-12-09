import React from "react";
import { render, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import MCPServers from "./mcp_servers";
import * as networking from "../networking";

// Mock the networking module
vi.mock("../networking", () => ({
  fetchMCPServers: vi.fn(),
  deleteMCPServer: vi.fn(),
  getProxyBaseUrl: vi.fn().mockReturnValue("http://localhost:4000"),
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
    accessToken: "test-token",
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
    expect(networking.fetchMCPServers).toHaveBeenCalledWith("test-token");
  });
});
