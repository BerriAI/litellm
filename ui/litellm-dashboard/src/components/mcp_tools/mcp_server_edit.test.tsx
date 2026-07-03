import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, fireEvent, act } from "@testing-library/react";
import MCPServerEdit from "./mcp_server_edit";
import * as networking from "../networking";
import NotificationsManager from "../molecules/notifications_manager";
import { selectAntOption } from "./testUtils";

vi.mock("../networking", () => ({
  updateMCPServer: vi.fn(),
  listMCPTools: vi.fn().mockResolvedValue({ tools: [], error: null }),
  storeMCPOAuthUserCredential: vi.fn().mockResolvedValue({}),
}));

vi.mock("../molecules/notifications_manager", () => ({
  default: {
    success: vi.fn(),
    fromBackend: vi.fn(),
  },
}));

const mockOauth: { tokenResponse: any } = { tokenResponse: null };
vi.mock("@/hooks/useMcpOAuthFlow", () => ({
  useMcpOAuthFlow: () => ({
    startOAuthFlow: vi.fn(),
    status: "idle",
    error: null,
    tokenResponse: mockOauth.tokenResponse,
  }),
}));

vi.mock("./mcp_server_cost_config", () => ({
  default: () => <div data-testid="mcp-cost-config" />,
}));

vi.mock("./MCPPermissionManagement", () => ({
  default: () => <div data-testid="mcp-permissions" />,
}));

vi.mock("./mcp_tool_configuration", () => ({
  default: ({
    existingAllowedTools,
    externalTools,
    externalError,
    onAllowedToolsChange,
    onToolAllowlistInteraction,
    onToolNameToDisplayNameChange,
    onToolNameToDescriptionChange,
  }: any) => (
    <div
      data-testid="mcp-tool-config"
      data-existing-allowed-tools={JSON.stringify(existingAllowedTools)}
      data-external-tools={JSON.stringify(externalTools)}
      data-external-error={externalError ?? ""}
    >
      <button
        type="button"
        onClick={() => {
          onToolAllowlistInteraction?.();
          onAllowedToolsChange([]);
        }}
      >
        Disable all tools
      </button>
      <button
        type="button"
        onClick={() => {
          onToolNameToDisplayNameChange({ read_user: "Read User" });
          onToolNameToDescriptionChange({ read_user: "Reads users" });
        }}
      >
        Set tool overrides
      </button>
    </div>
  ),
}));

const mockGetToken = vi.fn();
const mockIsTokenValid = vi.fn();
const mockSetToken = vi.fn();
vi.mock("@/utils/mcpTokenStore", () => ({
  getToken: (...args: any[]) => mockGetToken(...args),
  isTokenValid: (...args: any[]) => mockIsTokenValid(...args),
  setToken: (...args: any[]) => mockSetToken(...args),
}));

// ── fixtures ──────────────────────────────────────────────────────────────────

const interactiveOAuthServer = {
  server_id: "oauth_server_1",
  server_name: "OAuthServer",
  alias: "oauth_server", // underscores: hyphens fail validateMCPServerName
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

describe("MCPServerEdit (delegate auth)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should clear delegate auth flag when saving a non-oauth2 server", async () => {
    vi.mocked(networking.updateMCPServer).mockResolvedValue({
      ...interactiveOAuthServer,
      auth_type: "none",
      delegate_auth_to_upstream: false,
    });

    render(
      <MCPServerEdit
        mcpServer={{
          ...interactiveOAuthServer,
          auth_type: "none",
          delegate_auth_to_upstream: true,
        }}
        accessToken="access-token"
        onCancel={vi.fn()}
        onSuccess={vi.fn()}
        availableAccessGroups={[]}
      />,
    );

    const saveButtons = screen.getAllByRole("button", { name: "Save Changes" });
    await act(async () => {
      fireEvent.click(saveButtons[0]);
    });

    await waitFor(() => {
      expect(networking.updateMCPServer).toHaveBeenCalledTimes(1);
    });

    const [, payload] = vi.mocked(networking.updateMCPServer).mock.calls[0];
    expect(payload.auth_type).toBe("none");
    expect(payload.delegate_auth_to_upstream).toBe(false);
  });

  it("does not enable oauth_passthrough for an oauth2 server", async () => {
    vi.mocked(networking.updateMCPServer).mockResolvedValue({
      ...interactiveOAuthServer,
      oauth_passthrough: false,
    });

    render(
      <MCPServerEdit
        mcpServer={{
          ...interactiveOAuthServer,
          extra_headers: ["Authorization"],
          oauth_passthrough: true,
        }}
        accessToken="access-token"
        onCancel={vi.fn()}
        onSuccess={vi.fn()}
        availableAccessGroups={[]}
      />,
    );

    const saveButtons = screen.getAllByRole("button", { name: "Save Changes" });
    await act(async () => {
      fireEvent.click(saveButtons[0]);
    });

    await waitFor(() => {
      expect(networking.updateMCPServer).toHaveBeenCalledTimes(1);
    });

    const [, payload] = vi.mocked(networking.updateMCPServer).mock.calls[0];
    expect(payload.auth_type).toBe("oauth2");
    // oauth_passthrough is non-oauth2 only — must be forced false here.
    expect(payload.oauth_passthrough).toBe(false);
  });
});

describe("MCPServerEdit (tool allowlist)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("treats legacy empty allowed_tools as unrestricted", () => {
    render(
      <MCPServerEdit
        mcpServer={{
          ...interactiveOAuthServer,
          allowed_tools: [],
          mcp_info: { server_name: "OAuthServer" },
        }}
        accessToken={null}
        onCancel={vi.fn()}
        onSuccess={vi.fn()}
        availableAccessGroups={[]}
      />,
    );

    expect(screen.getByTestId("mcp-tool-config")).toHaveAttribute("data-existing-allowed-tools", "null");
  });

  it("honors enforced empty allowed_tools", () => {
    render(
      <MCPServerEdit
        mcpServer={{
          ...interactiveOAuthServer,
          allowed_tools: [],
          mcp_info: { server_name: "OAuthServer", tool_allowlist_enforced: true },
        }}
        accessToken={null}
        onCancel={vi.fn()}
        onSuccess={vi.fn()}
        availableAccessGroups={[]}
      />,
    );

    expect(screen.getByTestId("mcp-tool-config")).toHaveAttribute("data-existing-allowed-tools", "[]");
  });

  it("saves an explicit empty allowlist after legacy unrestricted tools are disabled", async () => {
    vi.mocked(networking.updateMCPServer).mockResolvedValue({
      ...interactiveOAuthServer,
      allowed_tools: [],
      mcp_info: { server_name: "OAuthServer", tool_allowlist_enforced: true },
    });

    render(
      <MCPServerEdit
        mcpServer={{
          ...interactiveOAuthServer,
          allowed_tools: [],
          mcp_info: { server_name: "OAuthServer" },
        }}
        accessToken="access-token"
        onCancel={vi.fn()}
        onSuccess={vi.fn()}
        availableAccessGroups={[]}
      />,
    );

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Disable all tools" }));
    });

    const saveButtons = screen.getAllByRole("button", { name: "Save Changes" });
    await act(async () => {
      fireEvent.click(saveButtons[0]);
    });

    await waitFor(() => {
      expect(networking.updateMCPServer).toHaveBeenCalledTimes(1);
    });

    const [, payload] = vi.mocked(networking.updateMCPServer).mock.calls[0];
    expect(payload.mcp_info.tool_allowlist_enforced).toBe(true);
    expect(payload.allowed_tools).toEqual([]);
  });

  it("saves tool overrides for legacy unrestricted servers", async () => {
    vi.mocked(networking.updateMCPServer).mockResolvedValue({
      ...interactiveOAuthServer,
      tool_name_to_display_name: { read_user: "Read User" },
      tool_name_to_description: { read_user: "Reads users" },
    });

    render(
      <MCPServerEdit
        mcpServer={{
          ...interactiveOAuthServer,
          allowed_tools: [],
          mcp_info: { server_name: "OAuthServer" },
        }}
        accessToken="access-token"
        onCancel={vi.fn()}
        onSuccess={vi.fn()}
        availableAccessGroups={[]}
      />,
    );

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Set tool overrides" }));
    });

    const saveButtons = screen.getAllByRole("button", { name: "Save Changes" });
    await act(async () => {
      fireEvent.click(saveButtons[0]);
    });

    await waitFor(() => {
      expect(networking.updateMCPServer).toHaveBeenCalledTimes(1);
    });

    const [, payload] = vi.mocked(networking.updateMCPServer).mock.calls[0];
    expect(payload.mcp_info.tool_allowlist_enforced).toBe(false);
    expect(payload.allowed_tools).toBeUndefined();
    expect(payload.tool_name_to_display_name).toEqual({ read_user: "Read User" });
    expect(payload.tool_name_to_description).toEqual({ read_user: "Reads users" });
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

  it("includes credentials.token_endpoint_auth_method in update payload when client_secret_basic is selected", async () => {
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
      expect(screen.getByText("Token Endpoint Auth Method (optional)")).toBeInTheDocument();
    });

    await selectAntOption("Token Endpoint Auth Method (optional)", "Client Secret Basic");

    const saveButtons = screen.getAllByRole("button", { name: "Save Changes" });
    await act(async () => {
      fireEvent.click(saveButtons[0]);
    });

    await waitFor(() => {
      expect(networking.updateMCPServer).toHaveBeenCalledTimes(1);
    });

    const [, payload] = vi.mocked(networking.updateMCPServer).mock.calls[0];
    expect(payload.credentials?.token_endpoint_auth_method).toBe("client_secret_basic");
  });

  it("omits token_endpoint_auth_method from the update payload when the selector is left blank", async () => {
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
      expect(screen.getByText("Token Endpoint Auth Method (optional)")).toBeInTheDocument();
    });

    const saveButtons = screen.getAllByRole("button", { name: "Save Changes" });
    await act(async () => {
      fireEvent.click(saveButtons[0]);
    });

    await waitFor(() => {
      expect(networking.updateMCPServer).toHaveBeenCalledTimes(1);
    });

    const [, payload] = vi.mocked(networking.updateMCPServer).mock.calls[0];
    expect(payload.credentials?.token_endpoint_auth_method).toBeUndefined();
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

describe("MCPServerEdit (tool list fetch)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(networking.listMCPTools).mockResolvedValue({ tools: [], error: null });
    mockOauth.tokenResponse = null;
  });

  it("loads an OBO server's tools via GET listMCPTools with no passthrough headers", async () => {
    vi.mocked(networking.listMCPTools).mockResolvedValue({
      tools: [{ name: "read_user" }],
      error: null,
    });

    render(
      <MCPServerEdit
        mcpServer={interactiveOAuthServer}
        accessToken="access-token"
        userID="user-1"
        onCancel={vi.fn()}
        onSuccess={vi.fn()}
        availableAccessGroups={[]}
      />,
    );

    await waitFor(() => {
      // includeDisabledTools=true so the config screen gets the full catalog.
      expect(networking.listMCPTools).toHaveBeenCalledWith("access-token", "oauth_server_1", undefined, true);
    });
    // OBO uses the backend-stored token; the browser passthrough store is never consulted.
    expect(mockIsTokenValid).not.toHaveBeenCalled();

    await waitFor(() => {
      expect(screen.getByTestId("mcp-tool-config")).toHaveAttribute(
        "data-external-tools",
        JSON.stringify([{ name: "read_user" }]),
      );
    });
  });

  it("forwards the sessionStorage token as the x-mcp passthrough header for a passthrough server", async () => {
    mockIsTokenValid.mockReturnValue(true);
    mockGetToken.mockReturnValue({ access_token: "browser-token" });

    render(
      <MCPServerEdit
        mcpServer={{ ...interactiveOAuthServer, delegate_auth_to_upstream: true }}
        accessToken="access-token"
        userID="user-1"
        onCancel={vi.fn()}
        onSuccess={vi.fn()}
        availableAccessGroups={[]}
      />,
    );

    await waitFor(() => {
      expect(networking.listMCPTools).toHaveBeenCalledWith(
        "access-token",
        "oauth_server_1",
        { "x-mcp-oauth_server-authorization": "Bearer browser-token" },
        true,
      );
    });
    expect(mockGetToken).toHaveBeenCalledWith("oauth_server_1", "user-1");
  });

  it("uses the staged OAuth token to load passthrough tools after authorize", async () => {
    const passthroughServer = { ...interactiveOAuthServer, delegate_auth_to_upstream: true };
    mockIsTokenValid.mockReturnValue(false);
    vi.mocked(networking.listMCPTools).mockResolvedValue({
      tools: [{ name: "read_user" }],
      error: null,
    });

    const props = {
      mcpServer: passthroughServer,
      accessToken: "access-token",
      userID: "user-1",
      onCancel: vi.fn(),
      onSuccess: vi.fn(),
      availableAccessGroups: [],
    };

    const { rerender } = render(<MCPServerEdit {...props} />);

    await waitFor(() => {
      expect(screen.getByTestId("mcp-tool-config").getAttribute("data-external-error")).toContain(
        "Authenticate with this server in the Tools tab",
      );
    });
    expect(networking.listMCPTools).not.toHaveBeenCalled();

    mockOauth.tokenResponse = { access_token: "staged-token", expires_in: 1800 };
    rerender(<MCPServerEdit {...props} />);

    await waitFor(() => {
      expect(networking.listMCPTools).toHaveBeenCalledWith(
        "access-token",
        "oauth_server_1",
        { "x-mcp-oauth_server-authorization": "Bearer staged-token" },
        true,
      );
    });
    expect(mockGetToken).not.toHaveBeenCalled();
  });

  it("prompts to authenticate and does not fetch when a passthrough server has no session token", async () => {
    mockIsTokenValid.mockReturnValue(false);

    render(
      <MCPServerEdit
        mcpServer={{ ...interactiveOAuthServer, delegate_auth_to_upstream: true }}
        accessToken="access-token"
        userID="user-1"
        onCancel={vi.fn()}
        onSuccess={vi.fn()}
        availableAccessGroups={[]}
      />,
    );

    await waitFor(() => {
      expect(screen.getByTestId("mcp-tool-config").getAttribute("data-external-error")).toContain(
        "Authenticate with this server in the Tools tab",
      );
    });
    expect(networking.listMCPTools).not.toHaveBeenCalled();
  });
});

describe("MCPServerEdit (form resync)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(networking.listMCPTools).mockResolvedValue({ tools: [], error: null });
  });

  it("repopulates the form when the server data arrives after mount", async () => {
    const props = {
      accessToken: "access-token",
      onCancel: vi.fn(),
      onSuccess: vi.fn(),
      availableAccessGroups: [],
    };

    // Mount before the server is loaded (mirrors landing on the page mid OAuth return).
    const { rerender } = render(<MCPServerEdit mcpServer={{ server_id: "" } as any} {...props} />);
    expect(screen.queryByDisplayValue("https://example.com/mcp")).not.toBeInTheDocument();

    // Server data arrives; the form must repopulate rather than staying blank.
    rerender(<MCPServerEdit mcpServer={interactiveOAuthServer} {...props} />);

    await waitFor(() => {
      expect(screen.getByDisplayValue("https://example.com/mcp")).toBeInTheDocument();
    });
  });
});

describe("MCPServerEdit (OAuth token persistence on save)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(networking.listMCPTools).mockResolvedValue({ tools: [], error: null });
    mockOauth.tokenResponse = null;
  });

  it("persists the OBO token to the DB on save after authorize", async () => {
    mockOauth.tokenResponse = {
      access_token: "obo-tok",
      refresh_token: "obo-refresh",
      expires_in: 3600,
      scope: "read write",
    };
    vi.mocked(networking.updateMCPServer).mockResolvedValue({ ...interactiveOAuthServer });

    render(
      <MCPServerEdit
        mcpServer={interactiveOAuthServer}
        accessToken="access-token"
        userID="user-1"
        onCancel={vi.fn()}
        onSuccess={vi.fn()}
        availableAccessGroups={[]}
      />,
    );

    await act(async () => {
      fireEvent.click(screen.getAllByRole("button", { name: "Save Changes" })[0]);
    });

    await waitFor(() => {
      expect(networking.storeMCPOAuthUserCredential).toHaveBeenCalledWith(
        "access-token",
        "oauth_server_1",
        expect.objectContaining({
          access_token: "obo-tok",
          refresh_token: "obo-refresh",
          expires_in: 3600,
          scopes: ["read", "write"],
        }),
      );
    });
    expect(mockSetToken).not.toHaveBeenCalled();
  });

  it("does not show success when OBO token persistence fails after update", async () => {
    mockOauth.tokenResponse = {
      access_token: "obo-tok",
      refresh_token: "obo-refresh",
      expires_in: 3600,
      scope: "read write",
    };
    vi.mocked(networking.updateMCPServer).mockResolvedValue({ ...interactiveOAuthServer });
    vi.mocked(networking.storeMCPOAuthUserCredential).mockRejectedValueOnce(new Error("write failed"));
    const onSuccess = vi.fn();

    render(
      <MCPServerEdit
        mcpServer={interactiveOAuthServer}
        accessToken="access-token"
        userID="user-1"
        onCancel={vi.fn()}
        onSuccess={onSuccess}
        availableAccessGroups={[]}
      />,
    );

    await act(async () => {
      fireEvent.click(screen.getAllByRole("button", { name: "Save Changes" })[0]);
    });

    await waitFor(() => {
      expect(networking.storeMCPOAuthUserCredential).toHaveBeenCalled();
    });
    expect(NotificationsManager.fromBackend).toHaveBeenCalledWith(
      "MCP Server updated, but failed to persist OAuth token: write failed",
    );
    expect(NotificationsManager.success).not.toHaveBeenCalledWith("MCP Server updated successfully");
    expect(onSuccess).not.toHaveBeenCalled();
  });

  it("persists the passthrough token to sessionStorage on save after authorize", async () => {
    mockOauth.tokenResponse = { access_token: "pt-tok", expires_in: 1800, token_type: "bearer" };
    vi.mocked(networking.updateMCPServer).mockResolvedValue({
      ...interactiveOAuthServer,
      delegate_auth_to_upstream: true,
    });

    render(
      <MCPServerEdit
        mcpServer={{ ...interactiveOAuthServer, delegate_auth_to_upstream: true }}
        accessToken="access-token"
        userID="user-1"
        onCancel={vi.fn()}
        onSuccess={vi.fn()}
        availableAccessGroups={[]}
      />,
    );

    await act(async () => {
      fireEvent.click(screen.getAllByRole("button", { name: "Save Changes" })[0]);
    });

    await waitFor(() => {
      expect(mockSetToken).toHaveBeenCalledWith(
        "oauth_server_1",
        expect.objectContaining({ access_token: "pt-tok", expires_in: 1800, token_type: "bearer" }),
        "user-1",
      );
    });
    expect(networking.storeMCPOAuthUserCredential).not.toHaveBeenCalled();
  });

  it("persists nothing on save when no token was fetched", async () => {
    mockOauth.tokenResponse = null;
    vi.mocked(networking.updateMCPServer).mockResolvedValue({ ...interactiveOAuthServer });

    render(
      <MCPServerEdit
        mcpServer={interactiveOAuthServer}
        accessToken="access-token"
        userID="user-1"
        onCancel={vi.fn()}
        onSuccess={vi.fn()}
        availableAccessGroups={[]}
      />,
    );

    await act(async () => {
      fireEvent.click(screen.getAllByRole("button", { name: "Save Changes" })[0]);
    });

    await waitFor(() => {
      expect(networking.updateMCPServer).toHaveBeenCalled();
    });
    expect(networking.storeMCPOAuthUserCredential).not.toHaveBeenCalled();
    expect(mockSetToken).not.toHaveBeenCalled();
  });
});
