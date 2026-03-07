import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, act, fireEvent } from "@testing-library/react";
import MCPServerDiagnostics from "./mcp_server_diagnostics";

vi.mock("../networking", () => ({
  fetchMCPServerDiagnostics: vi.fn(),
}));

import { fetchMCPServerDiagnostics } from "../networking";

describe("MCPServerDiagnostics", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should render the component title", async () => {
    (fetchMCPServerDiagnostics as any).mockResolvedValue({
      server_id: "test-server",
      server_name: "test-mcp",
      overall_status: "healthy",
      checks: [],
      suggestions: [],
    });

    await act(async () => {
      render(
        <MCPServerDiagnostics
          serverId="test-server"
          accessToken="test-token"
        />
      );
    });

    await waitFor(() => {
      expect(screen.getByText("Connection Diagnostics")).toBeInTheDocument();
    });
  });

  it("should show healthy status when all checks pass", async () => {
    (fetchMCPServerDiagnostics as any).mockResolvedValue({
      server_id: "test-server",
      server_name: "test-mcp",
      overall_status: "healthy",
      checks: [
        { name: "configuration", status: "pass", message: "OK" },
        { name: "connectivity", status: "pass", message: "OK" },
      ],
      suggestions: [],
    });

    await act(async () => {
      render(
        <MCPServerDiagnostics
          serverId="test-server"
          accessToken="test-token"
        />
      );
    });

    await waitFor(() => {
      expect(screen.getByText("All Checks Passing")).toBeInTheDocument();
    });
  });

  it("should show unhealthy status and suggestions when checks fail", async () => {
    (fetchMCPServerDiagnostics as any).mockResolvedValue({
      server_id: "bad-server",
      server_name: "bad-mcp",
      overall_status: "unhealthy",
      checks: [
        { name: "configuration", status: "pass", message: "OK" },
        {
          name: "connectivity",
          status: "fail",
          message: "Connection timed out",
        },
      ],
      suggestions: [
        "Check that the MCP server URL is reachable from the LiteLLM proxy network.",
      ],
    });

    await act(async () => {
      render(
        <MCPServerDiagnostics
          serverId="bad-server"
          accessToken="test-token"
        />
      );
    });

    await waitFor(() => {
      expect(screen.getByText("Issues Detected")).toBeInTheDocument();
      expect(screen.getByText("Suggestions")).toBeInTheDocument();
      expect(
        screen.getByText(
          /Check that the MCP server URL is reachable/
        )
      ).toBeInTheDocument();
    });
  });

  it("should show error alert when diagnostics fail", async () => {
    (fetchMCPServerDiagnostics as any).mockRejectedValue(
      new Error("Network error")
    );

    await act(async () => {
      render(
        <MCPServerDiagnostics
          serverId="test-server"
          accessToken="test-token"
        />
      );
    });

    await waitFor(() => {
      expect(screen.getByText("Diagnostics Failed")).toBeInTheDocument();
    });
  });

  it("should have a re-run button", async () => {
    (fetchMCPServerDiagnostics as any).mockResolvedValue({
      server_id: "test-server",
      server_name: "test-mcp",
      overall_status: "healthy",
      checks: [],
      suggestions: [],
    });

    await act(async () => {
      render(
        <MCPServerDiagnostics
          serverId="test-server"
          accessToken="test-token"
        />
      );
    });

    await waitFor(() => {
      expect(screen.getByText("Re-run")).toBeInTheDocument();
    });
  });
});
