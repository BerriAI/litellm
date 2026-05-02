import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, fireEvent, act } from "@testing-library/react";
import MCPServerEdit from "./mcp_server_edit";
import * as networking from "../networking";
import NotificationsManager from "../molecules/notifications_manager";

vi.mock("../networking", () => ({
  updateMCPServer: vi.fn(),
  listMCPTools: vi.fn().mockResolvedValue({ tools: [], error: null }),
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

// ── fixtures ──────────────────────────────────────────────────────────────────

const interactiveOAuthServer = {
  server_id: "oauth_server_1",
  server_name: "OAuthServer",
  alias: "oauth_server",   // underscores: hyphens fail validateMCPServerName
  description: "Interactive OAuth MCP server",
  transport: "http",
  url: "https://example.com/mcp",
  auth_type: "oauth2",
  // No token_url → edit form defaults to INTERACTIVE flow
  token_url: null,
  authorization_url: null,
  registration_url: null,
  created_at: "2024-01-01T00:00:00Z",
  created_by: "user-1",
  updated_at: "2024-01-01T00:00:00Z",
  updated_by: "user-1",
  mcp_access_groups: [],
};

// ── test suites ───────────────────────────────────────────────────────────────

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

describe("MCPServerEdit (interactive OAuth)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders Token Validation Rules and Token Storage TTL fields for interactive OAuth server", async () => {
    render(
      <MCPServerEdit
        mcpServer={interactiveOAuthServer}
        accessToken={null}
        onCancel={vi.fn()}
        onSuccess={vi.fn()}
        availableAccessGroups={[]}
      />,
    );

    await waitFor(() => {
      expect(screen.getByText("Token Validation Rules (optional)")).toBeInTheDocument();
      expect(screen.getByText("Token Storage TTL (seconds, optional)")).toBeInTheDocument();
    });
  });

  // Note: The M2M flow hiding logic is tested via OAuthFormFields.test.tsx (isM2M prop directly),
  // since Form.useWatch doesn't synchronously reflect initialValues in jsdom.

  it("pre-populates token_validation_json from existing server token_validation", async () => {
    const tokenValidation = { organization: "my-org", "team.id": "123" };

    render(
      <MCPServerEdit
        mcpServer={{ ...interactiveOAuthServer, token_validation: tokenValidation }}
        accessToken={null}
        onCancel={vi.fn()}
        onSuccess={vi.fn()}
        availableAccessGroups={[]}
      />,
    );

    await waitFor(() => {
      const textarea = document.getElementById("token_validation_json") as HTMLTextAreaElement;
      expect(textarea).not.toBeNull();
      const parsed = JSON.parse(textarea.value);
      expect(parsed).toEqual(tokenValidation);
    });
  });

  it("includes token_validation in update payload when token_validation_json is filled", async () => {
    const onSuccess = vi.fn();
    vi.mocked(networking.updateMCPServer).mockResolvedValue({
      ...interactiveOAuthServer,
      token_validation: { organization: "my-org" },
    });

    render(
      <MCPServerEdit
        mcpServer={interactiveOAuthServer}
        accessToken="access-token"
        onCancel={vi.fn()}
        onSuccess={onSuccess}
        availableAccessGroups={[]}
      />,
    );

    // Wait for the form to mount and the token_validation_json field to appear
    await waitFor(() => {
      expect(screen.getByText("Token Validation Rules (optional)")).toBeInTheDocument();
    });

    const textarea = document.getElementById("token_validation_json") as HTMLTextAreaElement;
    await act(async () => {
      fireEvent.change(textarea, { target: { value: '{"organization": "my-org"}' } });
    });

    const saveButtons = screen.getAllByRole("button", { name: "Save Changes" });
    await act(async () => {
      fireEvent.click(saveButtons[0]);
    });

    await waitFor(() => {
      expect(networking.updateMCPServer).toHaveBeenCalledTimes(1);
    });

    const [, payload] = vi.mocked(networking.updateMCPServer).mock.calls[0];
    expect(payload.token_validation).toEqual({ organization: "my-org" });
  });

  it("does not include token_validation in payload when field is empty and server had none", async () => {
    vi.mocked(networking.updateMCPServer).mockResolvedValue(interactiveOAuthServer);

    render(
      <MCPServerEdit
        mcpServer={interactiveOAuthServer}
        accessToken="access-token"
        onCancel={vi.fn()}
        onSuccess={vi.fn()}
        availableAccessGroups={[]}
      />,
    );

    await waitFor(() => {
      expect(screen.getByText("Token Validation Rules (optional)")).toBeInTheDocument();
    });

    // Leave token_validation_json empty
    const saveButtons = screen.getAllByRole("button", { name: "Save Changes" });
    await act(async () => {
      fireEvent.click(saveButtons[0]);
    });

    await waitFor(() => {
      expect(networking.updateMCPServer).toHaveBeenCalledTimes(1);
    });

    const [, payload] = vi.mocked(networking.updateMCPServer).mock.calls[0];
    expect(payload.token_validation).toBeUndefined();
  });

  it("sends token_validation: null to clear an existing value when textarea is cleared", async () => {
    vi.mocked(networking.updateMCPServer).mockResolvedValue({
      ...interactiveOAuthServer,
      token_validation: null,
    });

    render(
      <MCPServerEdit
        mcpServer={{ ...interactiveOAuthServer, token_validation: { organization: "old-org" } }}
        accessToken="access-token"
        onCancel={vi.fn()}
        onSuccess={vi.fn()}
        availableAccessGroups={[]}
      />,
    );

    await waitFor(() => {
      const textarea = document.getElementById("token_validation_json") as HTMLTextAreaElement;
      expect(textarea?.value).toContain("old-org");
    });

    // Clear the textarea
    const textarea = document.getElementById("token_validation_json") as HTMLTextAreaElement;
    await act(async () => {
      fireEvent.change(textarea, { target: { value: "" } });
    });

    const saveButtons = screen.getAllByRole("button", { name: "Save Changes" });
    await act(async () => {
      fireEvent.click(saveButtons[0]);
    });

    await waitFor(() => {
      expect(networking.updateMCPServer).toHaveBeenCalledTimes(1);
    });

    const [, payload] = vi.mocked(networking.updateMCPServer).mock.calls[0];
    // null signals the backend to clear the existing validation rules
    expect(payload.token_validation).toBeNull();
  });

  it("shows inline validation error and does not submit on invalid JSON in token_validation_json", async () => {
    render(
      <MCPServerEdit
        mcpServer={interactiveOAuthServer}
        accessToken="access-token"
        onCancel={vi.fn()}
        onSuccess={vi.fn()}
        availableAccessGroups={[]}
      />,
    );

    await waitFor(() => {
      expect(screen.getByText("Token Validation Rules (optional)")).toBeInTheDocument();
    });

    const textarea = document.getElementById("token_validation_json") as HTMLTextAreaElement;
    await act(async () => {
      fireEvent.change(textarea, { target: { value: "{ bad json" } });
    });

    const saveButtons = screen.getAllByRole("button", { name: "Save Changes" });
    await act(async () => {
      fireEvent.click(saveButtons[0]);
    });

    // The Form.Item inline validator intercepts invalid JSON before handleSave runs,
    // so the inline error message appears and updateMCPServer is never called.
    await waitFor(() => {
      expect(screen.getByText("Must be valid JSON")).toBeInTheDocument();
    });
    expect(networking.updateMCPServer).not.toHaveBeenCalled();
  });

  it("includes token_storage_ttl_seconds in payload when set", async () => {
    vi.mocked(networking.updateMCPServer).mockResolvedValue({
      ...interactiveOAuthServer,
      token_storage_ttl_seconds: 7200,
    });

    render(
      <MCPServerEdit
        mcpServer={{ ...interactiveOAuthServer, token_storage_ttl_seconds: 7200 }}
        accessToken="access-token"
        onCancel={vi.fn()}
        onSuccess={vi.fn()}
        availableAccessGroups={[]}
      />,
    );

    await waitFor(() => {
      expect(screen.getByText("Token Storage TTL (seconds, optional)")).toBeInTheDocument();
    });

    const saveButtons = screen.getAllByRole("button", { name: "Save Changes" });
    await act(async () => {
      fireEvent.click(saveButtons[0]);
    });

    await waitFor(() => {
      expect(networking.updateMCPServer).toHaveBeenCalledTimes(1);
    });

    const [, payload] = vi.mocked(networking.updateMCPServer).mock.calls[0];
    expect(payload.token_storage_ttl_seconds).toBe(7200);
  });
});
