import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, act } from "@testing-library/react";
import MCPServerLogs from "./mcp_server_logs";

vi.mock("../networking", () => ({
  fetchMCPServerLogs: vi.fn(),
}));

import { fetchMCPServerLogs } from "../networking";

describe("MCPServerLogs", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should render the component title", async () => {
    (fetchMCPServerLogs as any).mockResolvedValue({
      data: [],
      total: 0,
      page: 1,
      page_size: 25,
      total_pages: 0,
    });

    await act(async () => {
      render(
        <MCPServerLogs serverId="test-server" accessToken="test-token" />
      );
    });

    await waitFor(() => {
      expect(screen.getByText("Invocation Logs")).toBeInTheDocument();
    });
  });

  it("should show empty state when no logs", async () => {
    (fetchMCPServerLogs as any).mockResolvedValue({
      data: [],
      total: 0,
      page: 1,
      page_size: 25,
      total_pages: 0,
    });

    await act(async () => {
      render(
        <MCPServerLogs serverId="test-server" accessToken="test-token" />
      );
    });

    await waitFor(() => {
      expect(
        screen.getByText(
          /No MCP invocation logs found/
        )
      ).toBeInTheDocument();
    });
  });

  it("should display log entries when data is returned", async () => {
    (fetchMCPServerLogs as any).mockResolvedValue({
      data: [
        {
          request_id: "req-1",
          call_type: "call_mcp_tool",
          mcp_namespaced_tool_name: "github-mcp/list_repos",
          status: "success",
          spend: 0.001,
          total_tokens: null,
          request_duration_ms: 250,
          startTime: "2025-06-01T10:00:00Z",
          endTime: "2025-06-01T10:00:01Z",
          api_key: "sk-test123",
          team_id: null,
          end_user: null,
          metadata: {},
        },
      ],
      total: 1,
      page: 1,
      page_size: 25,
      total_pages: 1,
    });

    await act(async () => {
      render(
        <MCPServerLogs serverId="test-server" accessToken="test-token" />
      );
    });

    await waitFor(() => {
      expect(screen.getByText("list_repos")).toBeInTheDocument();
      expect(screen.getByText("Tool Call")).toBeInTheDocument();
      expect(screen.getByText("250ms")).toBeInTheDocument();
    });
  });

  it("should call fetchMCPServerLogs with correct params", async () => {
    (fetchMCPServerLogs as any).mockResolvedValue({
      data: [],
      total: 0,
      page: 1,
      page_size: 25,
      total_pages: 0,
    });

    await act(async () => {
      render(
        <MCPServerLogs serverId="my-server-id" accessToken="my-token" />
      );
    });

    await waitFor(() => {
      expect(fetchMCPServerLogs).toHaveBeenCalledWith(
        "my-token",
        "my-server-id",
        expect.any(String),
        expect.any(String),
        1,
        25,
      );
    });
  });
});
