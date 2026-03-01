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

const MockMCPToolConfiguration = vi.fn(() => <div data-testid="mcp-tool-config" />);
vi.mock("./mcp_tool_configuration", () => ({
  default: (props: any) => MockMCPToolConfiguration(props),
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

describe("MCPServerEdit (URL change clears allowed tools)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should pass live URL to MCPToolConfiguration when URL field changes", async () => {
    render(
      <MCPServerEdit
        mcpServer={{
          server_id: "server-1",
          server_name: "exa",
          alias: "exa",
          description: "Exa MCP server",
          transport: "http",
          url: "https://exa.example.com/mcp",
          auth_type: "none",
          allowed_tools: ["web_search_exa", "get_code_context_exa"],
          created_at: "2024-01-01T00:00:00Z",
          created_by: "user-1",
          updated_at: "2024-01-01T00:00:00Z",
          updated_by: "user-1",
          mcp_access_groups: [],
        }}
        accessToken="access-token"
        onCancel={vi.fn()}
        onSuccess={vi.fn()}
        availableAccessGroups={[]}
      />,
    );

    // Initially, MCPToolConfiguration should receive the original URL
    const initialCall = MockMCPToolConfiguration.mock.calls.find(
      (call) => call[0]?.formValues?.url === "https://exa.example.com/mcp"
    );
    expect(initialCall).toBeDefined();

    // Change the URL field
    const urlInput = screen.getByLabelText("MCP Server URL");
    await act(async () => {
      fireEvent.change(urlInput, {
        target: { value: "https://seolinkmap.com/mcp" },
      });
    });

    // After URL change, MCPToolConfiguration should receive the new URL
    await waitFor(() => {
      const latestCall = MockMCPToolConfiguration.mock.calls[MockMCPToolConfiguration.mock.calls.length - 1];
      expect(latestCall[0].formValues.url).toBe("https://seolinkmap.com/mcp");
    });
  });

  it("should clear allowedTools when URL changes from original", async () => {
    render(
      <MCPServerEdit
        mcpServer={{
          server_id: "server-1",
          server_name: "exa",
          alias: "exa",
          description: "Exa MCP server",
          transport: "http",
          url: "https://exa.example.com/mcp",
          auth_type: "none",
          allowed_tools: ["web_search_exa", "get_code_context_exa"],
          created_at: "2024-01-01T00:00:00Z",
          created_by: "user-1",
          updated_at: "2024-01-01T00:00:00Z",
          updated_by: "user-1",
          mcp_access_groups: [],
        }}
        accessToken="access-token"
        onCancel={vi.fn()}
        onSuccess={vi.fn()}
        availableAccessGroups={[]}
      />,
    );

    // Initially, allowedTools should be the existing tools
    const initialCall = MockMCPToolConfiguration.mock.calls[MockMCPToolConfiguration.mock.calls.length - 1];
    expect(initialCall[0].allowedTools).toEqual(["web_search_exa", "get_code_context_exa"]);

    // Change the URL field to a different server
    const urlInput = screen.getByLabelText("MCP Server URL");
    await act(async () => {
      fireEvent.change(urlInput, {
        target: { value: "https://seolinkmap.com/mcp" },
      });
    });

    // After URL change, allowedTools should be cleared
    await waitFor(() => {
      const latestCall = MockMCPToolConfiguration.mock.calls[MockMCPToolConfiguration.mock.calls.length - 1];
      expect(latestCall[0].allowedTools).toEqual([]);
    });
  });
});
