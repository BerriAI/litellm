import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, fireEvent, act } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import MCPServerEdit, { EDIT_OAUTH_UI_STATE_KEY } from "./mcp_server_edit";
import { setSecureItem } from "@/utils/secureStorage";
import * as networking from "@/components/networking";
import NotificationsManager from "@/components/molecules/notifications_manager";
import { selectAntOption } from "./testUtils";

vi.mock("@/components/networking", () => ({
  updateMCPServer: vi.fn(),
  listMCPTools: vi.fn().mockResolvedValue({ tools: [], error: null }),
  storeMCPOAuthUserCredential: vi.fn().mockResolvedValue({}),
  testMCPToolsListRequest: vi.fn().mockResolvedValue({ tools: [], error: null }),
}));

vi.mock("@/components/molecules/notifications_manager", () => ({
  default: {
    success: vi.fn(),
    fromBackend: vi.fn(),
  },
}));

const mockOauth: {
  tokenResponse: any;
  getTemporaryPayload: (() => Record<string, unknown> | null) | null;
  onTokenReceived: ((token: Record<string, unknown> | null) => void) | null;
  reset: ReturnType<typeof vi.fn>;
} = { tokenResponse: null, getTemporaryPayload: null, onTokenReceived: null, reset: vi.fn() };
vi.mock("@/hooks/useMcpOAuthFlow", () => ({
  useMcpOAuthFlow: (opts: {
    getTemporaryPayload?: () => Record<string, unknown> | null;
    onTokenReceived?: (token: Record<string, unknown> | null) => void;
  }) => {
    mockOauth.getTemporaryPayload = opts?.getTemporaryPayload ?? null;
    mockOauth.onTokenReceived = opts?.onTokenReceived ?? null;
    return {
      startOAuthFlow: vi.fn(),
      status: "idle",
      error: null,
      tokenResponse: mockOauth.tokenResponse,
      reset: mockOauth.reset,
    };
  },
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
          onToolNameToDisplayNameChange({ read_user: "ReadUser" });
          onToolNameToDescriptionChange({ read_user: "Reads users" });
        }}
      >
        Set tool overrides
      </button>
      <button
        type="button"
        onClick={() => {
          onToolNameToDisplayNameChange({ read_user: "Read User" });
        }}
      >
        Set invalid tool override
      </button>
    </div>
  ),
}));

const mockGetToken = vi.fn();
const mockIsTokenValid = vi.fn();
const mockSetToken = vi.fn();
const mockRemoveToken = vi.fn();
vi.mock("@/utils/mcpTokenStore", () => ({
  getToken: (...args: any[]) => mockGetToken(...args),
  isTokenValid: (...args: any[]) => mockIsTokenValid(...args),
  setToken: (...args: any[]) => mockSetToken(...args),
  removeToken: (...args: unknown[]) => mockRemoveToken(...args),
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

describe("MCPServerEdit (true passthrough warning)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  const renderWithAuthType = (authType: string) =>
    render(
      <MCPServerEdit
        mcpServer={{
          ...interactiveOAuthServer,
          auth_type: authType,
        }}
        accessToken="access-token"
        onCancel={vi.fn()}
        onSuccess={vi.fn()}
        availableAccessGroups={[]}
      />,
    );

  it("warns that LiteLLM auth is disabled for a true_passthrough server", async () => {
    renderWithAuthType("true_passthrough");

    await waitFor(() => {
      expect(screen.getByText("True Passthrough disables LiteLLM authentication for this server")).toBeInTheDocument();
    });
  });

  it("does not warn for an oauth_delegate server", async () => {
    renderWithAuthType("oauth_delegate");

    await waitFor(() => {
      expect(screen.getAllByRole("button", { name: "Save Changes" }).length).toBeGreaterThan(0);
    });
    expect(
      screen.queryByText("True Passthrough disables LiteLLM authentication for this server"),
    ).not.toBeInTheDocument();
  });

  it("browser-authorize temp payload uses the selected auth_type, not the stored one", async () => {
    // Stored server is oauth2; the admin switches the dropdown to true_passthrough before saving.
    // The temp OAuth-relay payload must reflect the selection so the exchange is treated as
    // browser-held (no DB persistence), matching onTokenReceived and the create form.
    render(
      <MCPServerEdit
        mcpServer={{ ...interactiveOAuthServer, auth_type: "oauth2" }}
        accessToken="access-token"
        onCancel={vi.fn()}
        onSuccess={vi.fn()}
        availableAccessGroups={[]}
      />,
    );

    await selectAntOption("Authentication", "True Passthrough (no LiteLLM auth)");

    await waitFor(() => {
      expect(mockOauth.getTemporaryPayload).toBeTruthy();
    });
    const payload = mockOauth.getTemporaryPayload!();
    expect(payload).toBeTruthy();
    expect(payload?.auth_type).toBe("true_passthrough");
  });
});

describe("MCPServerEdit (auth type switch)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("clears stale oauth2 endpoint overrides when switching to token exchange", async () => {
    vi.mocked(networking.updateMCPServer).mockResolvedValue({
      ...interactiveOAuthServer,
      auth_type: "oauth2_token_exchange",
    });

    render(
      <MCPServerEdit
        mcpServer={{
          ...interactiveOAuthServer,
          token_url: "https://old-idp.example.com/oauth/token",
          authorization_url: "https://old-idp.example.com/oauth/authorize",
          registration_url: "https://old-idp.example.com/oauth/register",
        }}
        accessToken="access-token"
        onCancel={vi.fn()}
        onSuccess={vi.fn()}
        availableAccessGroups={[]}
      />,
    );

    await selectAntOption("Authentication", "OAuth Token Exchange (OBO)");

    const saveButtons = screen.getAllByRole("button", { name: "Save Changes" });
    await act(async () => {
      fireEvent.click(saveButtons[0]);
    });

    await waitFor(() => {
      expect(networking.updateMCPServer).toHaveBeenCalledTimes(1);
    });

    const [, payload] = vi.mocked(networking.updateMCPServer).mock.calls[0];
    expect(payload.auth_type).toBe("oauth2_token_exchange");
    expect(payload.token_url).toBeNull();
    expect(payload.authorization_url).toBeNull();
    expect(payload.registration_url).toBeNull();
  });

  it("keeps oauth2 endpoint overrides when the auth type is unchanged", async () => {
    vi.mocked(networking.updateMCPServer).mockResolvedValue({ ...interactiveOAuthServer });

    render(
      <MCPServerEdit
        mcpServer={{
          ...interactiveOAuthServer,
          token_url: "https://idp.example.com/oauth/token",
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
    expect(payload.token_url).toBe("https://idp.example.com/oauth/token");
  });
});

describe("MCPServerEdit OAuth token invalidation", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  const renderOAuthEdit = () =>
    render(
      <MCPServerEdit
        mcpServer={{ ...interactiveOAuthServer }}
        accessToken="access-token"
        onCancel={vi.fn()}
        onSuccess={vi.fn()}
        availableAccessGroups={[]}
      />,
    );

  it("invalidates a session-authorized token when the transport switches to stdio", async () => {
    // Switching to stdio clears url/auth_type via programmatic form.setFieldsValue, which antd does
    // not report through onValuesChange; the explicit recheck in handleTransportChange must catch it.
    // Regression: the token used to survive this switch (sessionStorage + hook state kept the old
    // token minted for the http url).
    renderOAuthEdit();

    act(() => {
      mockOauth.onTokenReceived?.({ access_token: "tok-1" });
    });
    mockOauth.reset.mockClear();

    await selectAntOption("Transport Type", "Standard Input/Output (stdio)");

    await waitFor(() => expect(mockOauth.reset).toHaveBeenCalled());
    expect(mockRemoveToken).toHaveBeenCalledWith("oauth_server_1", undefined);
  });

  it("invalidates a session-authorized token when the server URL changes", async () => {
    renderOAuthEdit();

    act(() => {
      mockOauth.onTokenReceived?.({ access_token: "tok-1" });
    });
    mockOauth.reset.mockClear();

    const urlInput = screen.getByPlaceholderText("https://your-mcp-server.com");
    await act(async () => {
      fireEvent.change(urlInput, { target: { value: "https://other.example.com/mcp" } });
    });

    await waitFor(() => expect(mockOauth.reset).toHaveBeenCalled());
    expect(mockRemoveToken).toHaveBeenCalledWith("oauth_server_1", undefined);
  });

  it("previews tools with a staged interactive OAuth token before it is saved", async () => {
    // Regression: for authorization_code the fetch went by server_id only, relying on the stored DB
    // credential, so a token authorized in this edit session gave an empty preview until the admin
    // saved; the create form previews the identical state via the config-based preview endpoint.
    mockOauth.tokenResponse = { access_token: "staged-obo-tok" };

    renderOAuthEdit();

    await waitFor(() => {
      expect(vi.mocked(networking.testMCPToolsListRequest)).toHaveBeenCalledWith(
        "access-token",
        // oauth2_flow must be explicit: the preview endpoint infers client_credentials from
        // inherited client_id/client_secret/token_url and would strip the staged bearer.
        expect.objectContaining({
          server_id: "oauth_server_1",
          url: "https://example.com/mcp",
          oauth2_flow: "authorization_code",
        }),
        "staged-obo-tok",
      );
    });
    expect(networking.listMCPTools).not.toHaveBeenCalled();
    // Previewing must stay stateless: the staged token is committed only by an explicit Save
    // (storeMCPOAuthUserCredential for authorization_code, setToken for the client-forwarded modes).
    expect(networking.storeMCPOAuthUserCredential).not.toHaveBeenCalled();
    expect(mockSetToken).not.toHaveBeenCalled();
    expect(networking.updateMCPServer).not.toHaveBeenCalled();
    mockOauth.tokenResponse = null;
  });

  it("previews an OpenAPI server's staged token against its spec_path", async () => {
    mockOauth.tokenResponse = { access_token: "staged-obo-tok" };

    render(
      <MCPServerEdit
        mcpServer={{
          ...interactiveOAuthServer,
          transport: "openapi",
          url: null,
          spec_path: "https://example.com/openapi.json",
        }}
        accessToken="access-token"
        onCancel={vi.fn()}
        onSuccess={vi.fn()}
        availableAccessGroups={[]}
      />,
    );

    await waitFor(() => {
      expect(vi.mocked(networking.testMCPToolsListRequest)).toHaveBeenCalledWith(
        "access-token",
        expect.objectContaining({ spec_path: "https://example.com/openapi.json" }),
        "staged-obo-tok",
      );
    });
    mockOauth.tokenResponse = null;
  });

  it("keeps the admin's in-flight endpoint edits when the token is invalidated", async () => {
    // Regression: invalidation used to form.resetFields the endpoint fields; with the edit Form's
    // initialValues that silently reverted an admin-corrected token_url back to the saved (wrong)
    // value while still looking plausible. Only credentials (the minted material) may be wiped.
    renderOAuthEdit();

    const tokenUrlInput = screen.getByPlaceholderText("https://example.com/oauth/token");
    await act(async () => {
      fireEvent.change(tokenUrlInput, { target: { value: "https://corrected.example.com/token" } });
    });

    act(() => {
      mockOauth.onTokenReceived?.({ access_token: "tok-1" });
    });
    mockOauth.reset.mockClear();

    const urlInput = screen.getByPlaceholderText("https://your-mcp-server.com");
    await act(async () => {
      fireEvent.change(urlInput, { target: { value: "https://moved.example.com/mcp" } });
    });

    await waitFor(() => expect(mockOauth.reset).toHaveBeenCalled());
    expect((screen.getByPlaceholderText("https://example.com/oauth/token") as HTMLInputElement).value).toBe(
      "https://corrected.example.com/token",
    );
  });

  it("keeps a session-authorized token on an http to sse switch with the same url", async () => {
    // Same url means the same resource/audience (RFC 8707): the minted token is still valid, so a
    // pure transport swap between the two MCP wire protocols must not force a re-authorize.
    renderOAuthEdit();

    act(() => {
      mockOauth.onTokenReceived?.({ access_token: "tok-1" });
    });
    mockOauth.reset.mockClear();

    await selectAntOption("Transport Type", "Server-Sent Events (SSE)");

    expect(mockOauth.reset).not.toHaveBeenCalled();
    expect(mockRemoveToken).not.toHaveBeenCalled();
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
      tool_name_to_display_name: { read_user: "ReadUser" },
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
    expect(payload.tool_name_to_display_name).toEqual({ read_user: "ReadUser" });
    expect(payload.tool_name_to_description).toEqual({ read_user: "Reads users" });
  });

  it("blocks save and does not call the API when a tool display name contains a space", async () => {
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
      fireEvent.click(screen.getByRole("button", { name: "Set invalid tool override" }));
    });

    const saveButtons = screen.getAllByRole("button", { name: "Save Changes" });
    await act(async () => {
      fireEvent.click(saveButtons[0]);
    });

    expect(networking.updateMCPServer).not.toHaveBeenCalled();
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

  it("forwards the sessionStorage token as the x-mcp header for an oauth_delegate server", async () => {
    mockIsTokenValid.mockReturnValue(true);
    mockGetToken.mockReturnValue({ access_token: "browser-token" });

    render(
      <MCPServerEdit
        mcpServer={{ ...interactiveOAuthServer, auth_type: "oauth_delegate" }}
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

  it("prompts for the browser-only authorize when a true_passthrough server has no token", async () => {
    mockIsTokenValid.mockReturnValue(false);

    render(
      <MCPServerEdit
        mcpServer={{ ...interactiveOAuthServer, auth_type: "true_passthrough" }}
        accessToken="access-token"
        userID="user-1"
        onCancel={vi.fn()}
        onSuccess={vi.fn()}
        availableAccessGroups={[]}
      />,
    );

    await waitFor(() => {
      expect(screen.getByTestId("mcp-tool-config").getAttribute("data-external-error")).toContain(
        "Authorize with the upstream (browser-only",
      );
    });
    expect(networking.listMCPTools).not.toHaveBeenCalled();
    expect(screen.getByRole("button", { name: "Authorize & Fetch Tools (browser-only)" })).toBeInTheDocument();
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

  it.each([["true_passthrough"], ["oauth_delegate"]])(
    "persists the staged token to sessionStorage on save for the %s mode",
    async (authType) => {
      // Regression: the save path classified the staged token with getMcpOAuthMode, which returns
      // null for the client-forwarded modes, so setToken was never called and the browser-held
      // token was dropped on save; the create form's submit path already committed it.
      mockOauth.tokenResponse = { access_token: "cf-tok", expires_in: 1800, token_type: "bearer" };
      vi.mocked(networking.updateMCPServer).mockResolvedValue({
        ...interactiveOAuthServer,
        auth_type: authType,
      });

      render(
        <MCPServerEdit
          mcpServer={{ ...interactiveOAuthServer, auth_type: authType }}
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
          expect.objectContaining({ access_token: "cf-tok" }),
          "user-1",
        );
      });
      expect(networking.storeMCPOAuthUserCredential).not.toHaveBeenCalled();
      const [, payload] = vi.mocked(networking.updateMCPServer).mock.calls[0];
      expect(payload.credentials).toBeUndefined();
      expect(JSON.stringify(payload)).not.toContain("cf-tok");
    },
  );

  it.each([["true_passthrough"], ["oauth_delegate"]])(
    "persists admin-entered OAuth app credentials in the update payload for the %s mode",
    async (authType) => {
      mockOauth.tokenResponse = { access_token: "cf-tok", expires_in: 1800, token_type: "bearer" };
      vi.mocked(networking.updateMCPServer).mockResolvedValue({
        ...interactiveOAuthServer,
        auth_type: authType,
      });

      render(
        <MCPServerEdit
          mcpServer={{ ...interactiveOAuthServer, auth_type: authType }}
          accessToken="access-token"
          userID="user-1"
          onCancel={vi.fn()}
          onSuccess={vi.fn()}
          availableAccessGroups={[]}
        />,
      );

      const user = userEvent.setup({ delay: null });
      await user.type(
        screen.getByPlaceholderText("Leave blank to keep the currently saved app (if any)"),
        "org-app-client-id",
      );
      await user.type(
        screen.getByPlaceholderText("Leave blank to keep the currently saved secret (if any)"),
        "org-app-secret",
      );

      await act(async () => {
        fireEvent.click(screen.getAllByRole("button", { name: "Save Changes" })[0]);
      });

      await waitFor(() => expect(networking.updateMCPServer).toHaveBeenCalledTimes(1));
      const [, payload] = vi.mocked(networking.updateMCPServer).mock.calls[0];

      // The declared app is config and persists onto the row; the browser-held token still never
      // reaches the payload or the per-user credential store.
      expect(payload.credentials).toMatchObject({
        client_id: "org-app-client-id",
        client_secret: "org-app-secret",
      });
      expect(JSON.stringify(payload)).not.toContain("cf-tok");
      expect(networking.storeMCPOAuthUserCredential).not.toHaveBeenCalled();
    },
  );

  it.each([["true_passthrough"], ["oauth_delegate"]])(
    "preserves admin-entered app credentials when the URL changes after authorize for the %s mode",
    async (authType) => {
      vi.mocked(networking.updateMCPServer).mockResolvedValue({
        ...interactiveOAuthServer,
        auth_type: authType,
      });

      render(
        <MCPServerEdit
          mcpServer={{ ...interactiveOAuthServer, auth_type: authType }}
          accessToken="access-token"
          userID="user-1"
          onCancel={vi.fn()}
          onSuccess={vi.fn()}
          availableAccessGroups={[]}
        />,
      );

      const user = userEvent.setup({ delay: null });
      await user.type(
        screen.getByPlaceholderText("Leave blank to keep the currently saved app (if any)"),
        "org-app-client-id",
      );
      await user.type(
        screen.getByPlaceholderText("Leave blank to keep the currently saved secret (if any)"),
        "org-app-secret",
      );

      act(() => {
        mockOauth.onTokenReceived?.({ access_token: "cf-tok", token_type: "bearer" });
      });

      // The URL edit invalidates the held browser token (removeToken fires), but the declared app
      // is config and must survive the invalidation into the update payload.
      await act(async () => {
        fireEvent.change(screen.getByPlaceholderText("https://your-mcp-server.com"), {
          target: { value: "https://other.example.com/mcp" },
        });
      });
      expect(mockRemoveToken).toHaveBeenCalledWith("oauth_server_1", "user-1");

      await act(async () => {
        fireEvent.click(screen.getAllByRole("button", { name: "Save Changes" })[0]);
      });

      await waitFor(() => expect(networking.updateMCPServer).toHaveBeenCalledTimes(1));
      const [, payload] = vi.mocked(networking.updateMCPServer).mock.calls[0];
      expect(payload.url).toBe("https://other.example.com/mcp");
      expect(payload.credentials).toMatchObject({
        client_id: "org-app-client-id",
        client_secret: "org-app-secret",
      });
      expect(JSON.stringify(payload)).not.toContain("cf-tok");
    },
  );

  it("sends an explicit-null credential write when removing the saved app for true_passthrough", async () => {
    vi.mocked(networking.updateMCPServer).mockResolvedValue({
      ...interactiveOAuthServer,
      auth_type: "true_passthrough",
    });

    render(
      <MCPServerEdit
        mcpServer={{ ...interactiveOAuthServer, auth_type: "true_passthrough" }}
        accessToken="access-token"
        userID="user-1"
        onCancel={vi.fn()}
        onSuccess={vi.fn()}
        availableAccessGroups={[]}
      />,
    );

    // Blank fields keep the stored app (the backend merges partial credential updates), so the
    // edit form states that convention and removal is an explicit checkbox that saves nulls.
    expect(screen.getByPlaceholderText("Leave blank to keep the currently saved app (if any)")).toBeInTheDocument();

    fireEvent.click(
      screen.getByRole("checkbox", {
        name: /Remove the saved OAuth app on save/,
      }),
    );

    await act(async () => {
      fireEvent.click(screen.getAllByRole("button", { name: "Save Changes" })[0]);
    });

    await waitFor(() => expect(networking.updateMCPServer).toHaveBeenCalledTimes(1));
    const [, payload] = vi.mocked(networking.updateMCPServer).mock.calls[0];
    expect(payload.credentials).toEqual({ client_id: null, client_secret: null });
  });

  it("warns that the saved app may not match after a URL change on a client-forwarded server", async () => {
    render(
      <MCPServerEdit
        mcpServer={{
          ...interactiveOAuthServer,
          auth_type: "true_passthrough",
          credentials: { client_id: "stored-client" },
        }}
        accessToken="access-token"
        userID="user-1"
        onCancel={vi.fn()}
        onSuccess={vi.fn()}
        availableAccessGroups={[]}
      />,
    );

    // No warning until the upstream changes.
    expect(screen.queryByText(/registered for the previous upstream/)).not.toBeInTheDocument();

    await act(async () => {
      fireEvent.change(screen.getByPlaceholderText("https://your-mcp-server.com"), {
        target: { value: "https://different.example.com/mcp" },
      });
    });

    // Keep + warn parity with the create form: the stored app is kept, and the banner appears.
    expect(screen.getByText(/registered for the previous upstream/)).toBeInTheDocument();
  });

  it("preserves a stored client_id on OAuth-resume restore even when the saved snapshot is token-only", async () => {
    // Post-redirect restore: the sessionStorage snapshot carries only a minted token (no client keys),
    // while the loaded server has a stored client_id. The restore must merge the server's declared app
    // under the snapshot before stripping tokens, so the stored client_id is never cleared to blank.
    setSecureItem(
      EDIT_OAUTH_UI_STATE_KEY,
      JSON.stringify({
        serverId: "oauth_server_1",
        formValues: { auth_type: "true_passthrough", credentials: { access_token: "leftover-token" } },
      }),
    );

    render(
      <MCPServerEdit
        mcpServer={{
          ...interactiveOAuthServer,
          auth_type: "true_passthrough",
          credentials: { client_id: "stored-client" },
        }}
        accessToken="access-token"
        userID="user-1"
        onCancel={vi.fn()}
        onSuccess={vi.fn()}
        availableAccessGroups={[]}
      />,
    );

    const clientIdField = await screen.findByPlaceholderText("Leave blank to keep the currently saved app (if any)");
    await waitFor(() => expect((clientIdField as HTMLInputElement).value).toBe("stored-client"));
    // The leftover minted token must not have rehydrated anywhere.
    expect(document.body.innerHTML).not.toContain("leftover-token");
  });

  it("resets the remove-app checkbox on a server switch so it never deletes the next server's stored app", async () => {
    vi.mocked(networking.updateMCPServer).mockResolvedValue({
      ...interactiveOAuthServer,
      auth_type: "true_passthrough",
    });

    const { rerender } = render(
      <MCPServerEdit
        mcpServer={{ ...interactiveOAuthServer, server_id: "server-A", auth_type: "true_passthrough" }}
        accessToken="access-token"
        userID="user-1"
        onCancel={vi.fn()}
        onSuccess={vi.fn()}
        availableAccessGroups={[]}
      />,
    );

    // Check "remove saved app" on server A.
    fireEvent.click(screen.getByRole("checkbox", { name: /Remove the saved OAuth app on save/ }));
    expect(
      (screen.getByRole("checkbox", { name: /Remove the saved OAuth app on save/ }) as HTMLInputElement).checked,
    ).toBe(true);

    // Switch the panel to server B without unmounting.
    rerender(
      <MCPServerEdit
        mcpServer={{ ...interactiveOAuthServer, server_id: "server-B", auth_type: "true_passthrough" }}
        accessToken="access-token"
        userID="user-1"
        onCancel={vi.fn()}
        onSuccess={vi.fn()}
        availableAccessGroups={[]}
      />,
    );

    // The checkbox must have reset, so saving server B does not send the explicit-null delete write.
    expect(
      (screen.getByRole("checkbox", { name: /Remove the saved OAuth app on save/ }) as HTMLInputElement).checked,
    ).toBe(false);

    await act(async () => {
      fireEvent.click(screen.getAllByRole("button", { name: "Save Changes" })[0]);
    });

    await waitFor(() => expect(networking.updateMCPServer).toHaveBeenCalledTimes(1));
    const [, payload] = vi.mocked(networking.updateMCPServer).mock.calls[0];
    expect(payload.credentials).not.toEqual({ client_id: null, client_secret: null });
  });

  it("forwards a newly authorized browser-held token for tool loading before the form is saved", async () => {
    // Regression: fetchTools keyed the browser-held decision off the saved mcpServer.auth_type, so
    // after switching the form to true_passthrough and authorizing, the fresh token was not sent as
    // the x-mcp header until the server was saved.
    vi.mocked(networking.listMCPTools).mockResolvedValue({ tools: [], error: null });
    mockIsTokenValid.mockReturnValue(false);

    render(
      <MCPServerEdit
        mcpServer={{ ...interactiveOAuthServer, auth_type: "api_key" }}
        accessToken="access-token"
        userID="user-1"
        onCancel={vi.fn()}
        onSuccess={vi.fn()}
        availableAccessGroups={[]}
      />,
    );

    await selectAntOption("Authentication", "OAuth Delegate (client-supplied upstream token)");
    mockOauth.tokenResponse = { access_token: "fresh-tok", token_type: "bearer" };
    await selectAntOption("Authentication", "True Passthrough (no LiteLLM auth)");

    await waitFor(() => {
      const withHeaders = vi
        .mocked(networking.listMCPTools)
        .mock.calls.find(([, , headers]) => headers && JSON.stringify(headers).includes("fresh-tok"));
      expect(withHeaders).toBeTruthy();
    });
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

describe("MCPServerEdit oauth2_flow selector", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  async function saveAndGetPayload(server: Record<string, unknown>) {
    vi.mocked(networking.updateMCPServer).mockResolvedValue({ ...interactiveOAuthServer });

    render(
      <MCPServerEdit
        mcpServer={{ ...interactiveOAuthServer, ...server }}
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
    return payload;
  }

  it("never writes oauth2_flow for a legacy null-flow server with a token_url", async () => {
    const payload = await saveAndGetPayload({
      token_url: "https://idp.example.com/oauth/token",
      oauth2_flow: null,
    });
    expect(payload).not.toHaveProperty("oauth2_flow");
  });

  it("re-writes an explicit client_credentials row with its own prefilled value", async () => {
    const payload = await saveAndGetPayload({
      oauth2_flow: "client_credentials",
      token_url: "https://idp.example.com/oauth/token",
    });
    expect(payload.oauth2_flow).toBe("client_credentials");
  });

  it("re-writes the DCR authorization_code stamp with its own prefilled value", async () => {
    const payload = await saveAndGetPayload({
      oauth2_flow: "authorization_code",
      token_url: "https://idp.example.com/oauth/token",
    });
    expect(payload.oauth2_flow).toBe("authorization_code");
  });

  it("persists client_credentials when the admin selects M2M on a legacy null-flow row", async () => {
    vi.mocked(networking.updateMCPServer).mockResolvedValue({ ...interactiveOAuthServer });

    render(
      <MCPServerEdit
        mcpServer={{
          ...interactiveOAuthServer,
          token_url: "https://idp.example.com/oauth/token",
          oauth2_flow: null,
        }}
        accessToken="access-token"
        onCancel={vi.fn()}
        onSuccess={vi.fn()}
        availableAccessGroups={[]}
      />,
    );

    await selectAntOption("OAuth Flow Type", "Machine-to-Machine (M2M)");

    const saveButtons = screen.getAllByRole("button", { name: "Save Changes" });
    await act(async () => {
      fireEvent.click(saveButtons[0]);
    });

    await waitFor(() => {
      expect(networking.updateMCPServer).toHaveBeenCalledTimes(1);
    });

    const [, payload] = vi.mocked(networking.updateMCPServer).mock.calls[0];
    expect(payload.oauth2_flow).toBe("client_credentials");
  });

  it("persists authorization_code when the admin selects Interactive on a legacy null-flow row", async () => {
    vi.mocked(networking.updateMCPServer).mockResolvedValue({ ...interactiveOAuthServer });

    render(
      <MCPServerEdit
        mcpServer={{
          ...interactiveOAuthServer,
          token_url: "https://idp.example.com/oauth/token",
          oauth2_flow: null,
        }}
        accessToken="access-token"
        onCancel={vi.fn()}
        onSuccess={vi.fn()}
        availableAccessGroups={[]}
      />,
    );

    await selectAntOption("OAuth Flow Type", "Interactive (PKCE)");

    const saveButtons = screen.getAllByRole("button", { name: "Save Changes" });
    await act(async () => {
      fireEvent.click(saveButtons[0]);
    });

    await waitFor(() => {
      expect(networking.updateMCPServer).toHaveBeenCalledTimes(1);
    });

    const [, payload] = vi.mocked(networking.updateMCPServer).mock.calls[0];
    expect(payload.oauth2_flow).toBe("authorization_code");
  });
});

describe("MCPServerEdit OAuth flow prefill display", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  function renderEdit(server: Record<string, unknown>) {
    render(
      <MCPServerEdit
        mcpServer={{ ...interactiveOAuthServer, ...server }}
        accessToken="access-token"
        onCancel={vi.fn()}
        onSuccess={vi.fn()}
        availableAccessGroups={[]}
      />,
    );
  }

  it("shows the placeholder and preselects nothing for a null-flow server (prompts the user to define it)", () => {
    renderEdit({ oauth2_flow: null, token_url: "https://idp.example.com/oauth/token" });

    // The select renders its placeholder (undefined value), not a guessed option.
    expect(screen.getByText("Select OAuth flow")).toBeInTheDocument();
    // Neither flow is preselected as the current value.
    expect(screen.queryByText("Machine-to-Machine (M2M)")).not.toBeInTheDocument();
    expect(screen.queryByText("Interactive (PKCE)")).not.toBeInTheDocument();
  });

  it("prefills Machine-to-Machine (M2M) for a stored client_credentials server", () => {
    renderEdit({ oauth2_flow: "client_credentials" });

    expect(screen.getByText("Machine-to-Machine (M2M)")).toBeInTheDocument();
    expect(screen.queryByText("Select OAuth flow")).not.toBeInTheDocument();
  });

  it("prefills Interactive (PKCE) for a stored authorization_code server", () => {
    renderEdit({ oauth2_flow: "authorization_code" });

    expect(screen.getByText("Interactive (PKCE)")).toBeInTheDocument();
    expect(screen.queryByText("Select OAuth flow")).not.toBeInTheDocument();
  });

  it("warns when a server has no OAuth flow set", () => {
    renderEdit({ oauth2_flow: null, token_url: "https://idp.example.com/oauth/token" });

    expect(screen.getByText("This server has no OAuth flow set")).toBeInTheDocument();
  });

  it("does not warn when the flow is already set", () => {
    renderEdit({ oauth2_flow: "client_credentials" });

    expect(screen.queryByText("This server has no OAuth flow set")).not.toBeInTheDocument();
  });

  it("does not warn for a delegate (PKCE passthrough) server even with no flow set", () => {
    renderEdit({ oauth2_flow: null, delegate_auth_to_upstream: true });

    expect(screen.queryByText("This server has no OAuth flow set")).not.toBeInTheDocument();
  });

  it("clears the warning once the admin selects a flow", async () => {
    renderEdit({ oauth2_flow: null, token_url: "https://idp.example.com/oauth/token" });

    expect(screen.getByText("This server has no OAuth flow set")).toBeInTheDocument();

    await selectAntOption("OAuth Flow Type", "Machine-to-Machine (M2M)");

    await waitFor(() => {
      expect(screen.queryByText("This server has no OAuth flow set")).not.toBeInTheDocument();
    });
  });
});

describe("MCPServerEdit (max concurrent requests)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  const limitedServer = {
    ...interactiveOAuthServer,
    auth_type: "none",
    max_concurrent_requests: 5,
  };

  it("prefills the existing limit and sends an updated value in the payload", async () => {
    vi.mocked(networking.updateMCPServer).mockResolvedValue({
      ...limitedServer,
      max_concurrent_requests: 2,
    });

    render(
      <MCPServerEdit
        mcpServer={limitedServer}
        accessToken="access-token"
        onCancel={vi.fn()}
        onSuccess={vi.fn()}
        availableAccessGroups={[]}
      />,
    );

    const limitInput = screen.getByPlaceholderText("e.g. 10") as HTMLInputElement;
    expect(limitInput.value).toBe("5");

    fireEvent.change(limitInput, { target: { value: "2" } });

    const saveButtons = screen.getAllByRole("button", { name: "Save Changes" });
    await act(async () => {
      fireEvent.click(saveButtons[0]);
    });

    await waitFor(() => {
      expect(networking.updateMCPServer).toHaveBeenCalledTimes(1);
    });

    const [, payload] = vi.mocked(networking.updateMCPServer).mock.calls[0];
    expect(payload.max_concurrent_requests).toBe(2);
  });

  it("sends null when the limit is cleared so the backend unsets it", async () => {
    vi.mocked(networking.updateMCPServer).mockResolvedValue({
      ...limitedServer,
      max_concurrent_requests: null,
    });

    render(
      <MCPServerEdit
        mcpServer={limitedServer}
        accessToken="access-token"
        onCancel={vi.fn()}
        onSuccess={vi.fn()}
        availableAccessGroups={[]}
      />,
    );

    const limitInput = screen.getByPlaceholderText("e.g. 10") as HTMLInputElement;
    expect(limitInput.value).toBe("5");

    fireEvent.change(limitInput, { target: { value: "" } });

    const saveButtons = screen.getAllByRole("button", { name: "Save Changes" });
    await act(async () => {
      fireEvent.click(saveButtons[0]);
    });

    await waitFor(() => {
      expect(networking.updateMCPServer).toHaveBeenCalledTimes(1);
    });

    const [, payload] = vi.mocked(networking.updateMCPServer).mock.calls[0];
    expect(payload.max_concurrent_requests).toBeNull();
  });
});

describe("MCPServerEdit (dcr_bridge toggle)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockOauth.tokenResponse = null;
  });

  const getDcrToggle = () => document.getElementById("dcr_bridge");

  function renderEdit(server: Record<string, unknown>) {
    render(
      <MCPServerEdit
        mcpServer={{ ...interactiveOAuthServer, ...server }}
        accessToken="access-token"
        onCancel={vi.fn()}
        onSuccess={vi.fn()}
        availableAccessGroups={[]}
      />,
    );
  }

  async function saveAndGetPayload() {
    const saveButtons = screen.getAllByRole("button", { name: "Save Changes" });
    await act(async () => {
      fireEvent.click(saveButtons[0]);
    });
    await waitFor(() => {
      expect(networking.updateMCPServer).toHaveBeenCalledTimes(1);
    });
    const [, payload] = vi.mocked(networking.updateMCPServer).mock.calls[0];
    return payload;
  }

  it.each([["true_passthrough"], ["oauth_delegate"]])("renders the toggle for a %s server", async (authType) => {
    renderEdit({ auth_type: authType });

    await waitFor(() => {
      expect(getDcrToggle()).toBeInTheDocument();
    });
    expect(screen.getByText("Gateway-hosted sign-in (DCR bridge)")).toBeInTheDocument();
  });

  it.each([["oauth2"], ["api_key"], ["none"]])("does not render the toggle for an %s server", async (authType) => {
    renderEdit({ auth_type: authType });

    await waitFor(() => {
      expect(screen.getAllByRole("button", { name: "Save Changes" }).length).toBeGreaterThan(0);
    });
    expect(screen.queryByText("Gateway-hosted sign-in (DCR bridge)")).not.toBeInTheDocument();
    expect(getDcrToggle()).not.toBeInTheDocument();
  });

  it("renders the toggle between the OAuth client fields and the Authorize button", async () => {
    renderEdit({ auth_type: "true_passthrough" });

    await waitFor(() => {
      expect(getDcrToggle()).toBeInTheDocument();
    });
    const toggle = getDcrToggle() as HTMLElement;
    const secretInput = screen.getByPlaceholderText("Leave blank to keep the currently saved secret (if any)");
    const authorizeButton = screen.getByRole("button", { name: "Authorize & Fetch Tools (browser-only)" });
    expect(secretInput.compareDocumentPosition(toggle) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(toggle.compareDocumentPosition(authorizeButton) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });

  it("initializes unchecked from a null stored value and saves an explicit false", async () => {
    vi.mocked(networking.updateMCPServer).mockResolvedValue({
      ...interactiveOAuthServer,
      auth_type: "true_passthrough",
    });
    renderEdit({ auth_type: "true_passthrough", dcr_bridge: null });

    await waitFor(() => {
      expect(getDcrToggle()).toBeInTheDocument();
    });
    expect(getDcrToggle()).toHaveAttribute("aria-checked", "false");

    const payload = await saveAndGetPayload();
    expect(payload.dcr_bridge).toBe(false);
  });

  it("initializes checked from a stored true and saves an explicit true", async () => {
    vi.mocked(networking.updateMCPServer).mockResolvedValue({
      ...interactiveOAuthServer,
      auth_type: "oauth_delegate",
      dcr_bridge: true,
    });
    renderEdit({ auth_type: "oauth_delegate", dcr_bridge: true });

    await waitFor(() => {
      expect(getDcrToggle()).toBeInTheDocument();
    });
    expect(getDcrToggle()).toHaveAttribute("aria-checked", "true");

    const payload = await saveAndGetPayload();
    expect(payload.dcr_bridge).toBe(true);
  });

  it("saves an explicit false after the admin unchecks a stored true", async () => {
    vi.mocked(networking.updateMCPServer).mockResolvedValue({
      ...interactiveOAuthServer,
      auth_type: "true_passthrough",
      dcr_bridge: false,
    });
    renderEdit({ auth_type: "true_passthrough", dcr_bridge: true });

    await waitFor(() => {
      expect(getDcrToggle()).toBeInTheDocument();
    });
    await act(async () => {
      fireEvent.click(getDcrToggle()!);
    });
    expect(getDcrToggle()).toHaveAttribute("aria-checked", "false");

    const payload = await saveAndGetPayload();
    expect(payload.dcr_bridge).toBe(false);
  });

  it("forces dcr_bridge: false when the auth type is switched away", async () => {
    vi.mocked(networking.updateMCPServer).mockResolvedValue({
      ...interactiveOAuthServer,
      auth_type: "api_key",
    });
    renderEdit({ auth_type: "true_passthrough", dcr_bridge: true });

    await waitFor(() => {
      expect(getDcrToggle()).toBeInTheDocument();
    });

    await selectAntOption("Authentication", "API Key");
    await waitFor(() => {
      expect(getDcrToggle()).not.toBeInTheDocument();
    });

    // Mirrors the sibling delegate_auth_to_upstream / oauth_passthrough force-false: a stale true is
    // never left behind to silently re-activate if the mode is switched back.
    const payload = await saveAndGetPayload();
    expect(payload.dcr_bridge).toBe(false);
  });

  it("preserves the toggle value when switching between the two client-forwarded modes", async () => {
    vi.mocked(networking.updateMCPServer).mockResolvedValue({
      ...interactiveOAuthServer,
      auth_type: "oauth_delegate",
      dcr_bridge: true,
    });
    renderEdit({ auth_type: "true_passthrough", dcr_bridge: true });

    await waitFor(() => {
      expect(getDcrToggle()).toBeInTheDocument();
    });
    expect(getDcrToggle()).toHaveAttribute("aria-checked", "true");

    // The Form.Item stays mounted across the two client-forwarded modes, so the live toggle value is
    // preserved rather than forced false by the switch.
    await selectAntOption("Authentication", "OAuth Delegate (client-supplied upstream token)");
    await waitFor(() => {
      expect(getDcrToggle()).toBeInTheDocument();
    });
    expect(getDcrToggle()).toHaveAttribute("aria-checked", "true");

    const payload = await saveAndGetPayload();
    expect(payload.dcr_bridge).toBe(true);
  });
});
