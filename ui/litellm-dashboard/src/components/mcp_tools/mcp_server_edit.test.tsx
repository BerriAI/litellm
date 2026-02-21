import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, fireEvent, act } from "@testing-library/react";
import MCPServerEdit from "./mcp_server_edit";
import * as networking from "../networking";

vi.mock("../networking", () => ({
  updateMCPServer: vi.fn(),
  testMCPToolsListRequest: vi.fn().mockResolvedValue({ tools: [], error: null }),
}));

vi.mock("../molecules/notifications_manager", () => ({
  default: {
    success: vi.fn(),
    fromBackend: vi.fn(),
  },
}));

vi.mock("@/hooks/useMcpOAuthFlow", () => ({
  useMcpOAuthFlow: () => ({
    startOAuthFlow: vi.fn(),
    status: "idle",
    error: null,
    tokenResponse: null,
  }),
}));

vi.mock("./mcp_server_cost_config", () => ({
  default: () => <div data-testid="mcp-cost-config" />,
}));

vi.mock("./MCPPermissionManagement", () => ({
  default: () => <div data-testid="mcp-permissions" />,
}));

vi.mock("./mcp_tool_configuration", () => ({
  default: () => <div data-testid="mcp-tool-config" />,
}));

describe("MCPServerEdit (stdio)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should render without crashing", () => {
    render(
      <MCPServerEdit
        mcpServer={{
          server_id: "server-1",
          server_name: "TestServer",
          alias: "test",
          description: "desc",
          transport: "stdio",
          url: null,
          auth_type: "none",
          command: "npx",
          args: ["-y", "@circleci/mcp-server-circleci"],
          env: { CIRCLECI_TOKEN: "token" },
          created_at: "2024-01-01T00:00:00Z",
          created_by: "user-1",
          updated_at: "2024-01-01T00:00:00Z",
          updated_by: "user-1",
          mcp_access_groups: [],
        }}
        // Avoid triggering async tool fetch side-effects in this smoke test.
        accessToken={null}
        onCancel={vi.fn()}
        onSuccess={vi.fn()}
        availableAccessGroups={[]}
      />,
    );

    expect(screen.getByRole("tab", { name: "Server Configuration" })).toBeInTheDocument();
  });

  it("should allow updating stdio transport configuration", async () => {
    const onCancel = vi.fn();
    const onSuccess = vi.fn();

    vi.mocked(networking.updateMCPServer).mockResolvedValue({
      server_id: "server-1",
      server_name: "TestServer",
      alias: "test",
      transport: "stdio",
      url: null,
      command: "npx",
      args: ["-y", "@circleci/mcp-server-circleci"],
      env: { CIRCLECI_TOKEN: "***" },
      created_at: "2024-01-01T00:00:00Z",
      created_by: "user-1",
      updated_at: "2024-01-01T00:00:00Z",
      updated_by: "user-1",
    });

    render(
      <MCPServerEdit
        mcpServer={{
          server_id: "server-1",
          server_name: "TestServer",
          alias: "test",
          description: "desc",
          transport: "stdio",
          url: null,
          auth_type: "none",
          command: "npx",
          args: ["-y", "@circleci/mcp-server-circleci"],
          env: { CIRCLECI_TOKEN: "token" },
          created_at: "2024-01-01T00:00:00Z",
          created_by: "user-1",
          updated_at: "2024-01-01T00:00:00Z",
          updated_by: "user-1",
          mcp_access_groups: [],
        }}
        accessToken="access-token"
        onCancel={onCancel}
        onSuccess={onSuccess}
        availableAccessGroups={[]}
      />,
    );

    // Stdio section should be visible
    expect(screen.getByLabelText("Command")).toBeInTheDocument();

    // URL field should not be visible when transport=stdio
    expect(screen.queryByText("MCP Server URL")).not.toBeInTheDocument();

    // Update env_json
    const envTextarea = screen.getByLabelText("Environment (JSON object)");
    await act(async () => {
      fireEvent.change(envTextarea, {
        target: {
          value: JSON.stringify({ CIRCLECI_TOKEN: "new-token", CIRCLECI_BASE_URL: "https://circleci.com" }, null, 2),
        },
      });
    });

    const saveButtons = screen.getAllByRole("button", { name: "Save Changes" });
    const saveButton = saveButtons[0];
    await act(async () => {
      fireEvent.click(saveButton);
    });

    await waitFor(() => {
      expect(networking.updateMCPServer).toHaveBeenCalledTimes(1);
    });

    const [_token, payload] = vi.mocked(networking.updateMCPServer).mock.calls[0];
    expect(_token).toBe("access-token");
    expect(payload.transport).toBe("stdio");
    expect(payload.command).toBe("npx");
    expect(payload.args).toEqual(["-y", "@circleci/mcp-server-circleci"]);
    expect(payload.env).toEqual({ CIRCLECI_TOKEN: "new-token", CIRCLECI_BASE_URL: "https://circleci.com" });
  });
});
