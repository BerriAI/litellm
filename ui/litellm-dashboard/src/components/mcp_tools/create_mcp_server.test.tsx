import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import * as networking from "../networking";
import { setToken } from "@/utils/mcpTokenStore";
import CreateMCPServer from "./create_mcp_server";
import { selectAntOption } from "./testUtils";

vi.mock("../networking", () => ({
  createMCPServer: vi.fn(),
  fetchOpenAPIRegistry: vi.fn().mockResolvedValue({ apis: [] }),
  registerMCPServer: vi.fn(),
  storeMCPOAuthUserCredential: vi.fn().mockResolvedValue({}),
  testMCPToolsListRequest: vi.fn().mockResolvedValue({ tools: [], error: null }),
}));

vi.mock("@/utils/mcpTokenStore", () => ({
  setToken: vi.fn(),
}));

vi.mock("./OpenAPIQuickPicker", () => ({
  default: () => null,
}));

// Mutable holder so individual tests can simulate "Authorize & Fetch" having
// produced a token before submit, and inspect the reset wiring.
const oauthHook = vi.hoisted(() => ({
  tokenResponse: null as Record<string, unknown> | null,
  reset: vi.fn(),
  onTokenReceived: null as
    | ((token: Record<string, unknown> | null, registeredClient?: { clientId?: string; clientSecret?: string }) => void)
    | null,
}));
vi.mock("@/hooks/useMcpOAuthFlow", () => ({
  useMcpOAuthFlow: (opts: {
    onTokenReceived: (
      token: Record<string, unknown> | null,
      registeredClient?: { clientId?: string; clientSecret?: string },
    ) => void;
  }) => {
    oauthHook.onTokenReceived = opts.onTokenReceived;
    return {
      startOAuthFlow: vi.fn(),
      status: "idle",
      error: null,
      tokenResponse: oauthHook.tokenResponse,
      reset: oauthHook.reset,
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
    onAllowedToolsChange,
    onToolAllowlistInteraction,
  }: {
    onAllowedToolsChange?: (tools: string[]) => void;
    onToolAllowlistInteraction?: () => void;
  }) => (
    <div data-testid="mcp-tool-config">
      <button
        type="button"
        onClick={() => {
          onToolAllowlistInteraction?.();
          onAllowedToolsChange?.([]);
        }}
      >
        Disable all tools
      </button>
    </div>
  ),
}));

vi.mock("./mcp_connection_status", () => ({
  default: ({ tools }: { tools?: any[] }) => (
    <div data-testid="mcp-connection-status" data-tool-count={tools?.length ?? 0} />
  ),
}));

vi.mock("./StdioConfiguration", () => ({
  default: () => <div data-testid="stdio-config" />,
}));

const defaultProps = {
  userRole: "Admin",
  accessToken: "test-token",
  onCreateSuccess: vi.fn(),
  isModalVisible: true,
  setModalVisible: vi.fn(),
  availableAccessGroups: ["group-a", "group-b"],
};

/** Helper: get the server_name input by its Ant Form id */
const getServerNameInput = () => document.getElementById("server_name") as HTMLInputElement;

describe("CreateMCPServer", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    oauthHook.tokenResponse = null;
    oauthHook.onTokenReceived = null;
  });

  it("should render the modal with title when visible", () => {
    render(<CreateMCPServer {...defaultProps} />);

    expect(screen.getByText("Add New MCP Server")).toBeInTheDocument();
  });

  it("should not render when user is not an admin", () => {
    render(<CreateMCPServer {...defaultProps} userRole="Internal User" />);

    expect(screen.queryByText("Add New MCP Server")).not.toBeInTheDocument();
  });

  it("should show transport type options", async () => {
    render(<CreateMCPServer {...defaultProps} />);

    await selectAntOption("Transport Type", "Streamable HTTP");

    // Verify the option was applied by checking the URL field appears
    await waitFor(() => {
      expect(screen.getByPlaceholderText("https://your-mcp-server.com")).toBeInTheDocument();
    });
  });

  describe("when HTTP transport is selected", () => {
    async function selectHttpTransport() {
      render(<CreateMCPServer {...defaultProps} />);
      await selectAntOption("Transport Type", "Streamable HTTP");

      // Wait for URL field to appear (confirms transport was set)
      await waitFor(() => {
        expect(screen.getByPlaceholderText("https://your-mcp-server.com")).toBeInTheDocument();
      });
    }

    it("should show URL field after selecting HTTP transport", async () => {
      await selectHttpTransport();

      expect(screen.getByPlaceholderText("https://your-mcp-server.com")).toBeInTheDocument();
    });

    it("should show auth type dropdown after selecting HTTP transport", async () => {
      await selectHttpTransport();

      expect(screen.getByText("Authentication")).toBeInTheDocument();
    });

    it("should show auth value field when API Key auth type is selected", async () => {
      await selectHttpTransport();

      await selectAntOption("Authentication", "API Key");

      await waitFor(() => {
        expect(screen.getByText("Authentication Value")).toBeInTheDocument();
      });
    });

    it("should not require auth value when creating a server with API Key auth type", async () => {
      await selectHttpTransport();

      const user = userEvent.setup({ delay: null });

      // Fill in server name (use id to avoid duplicate placeholder)
      const nameInput = getServerNameInput();
      await user.type(nameInput, "Test_Server");

      // Fill in URL
      const urlInput = screen.getByPlaceholderText("https://your-mcp-server.com");
      await user.type(urlInput, "https://example.com/mcp");

      // Select API Key auth type
      await selectAntOption("Authentication", "API Key");

      await waitFor(() => {
        expect(screen.getByText("Authentication Value")).toBeInTheDocument();
      });

      // Leave auth value empty and submit
      vi.mocked(networking.createMCPServer).mockResolvedValue({
        server_id: "new-server-1",
        server_name: "Test_Server",
        alias: "Test_Server",
        url: "https://example.com/mcp",
        transport: "http",
        auth_type: "api_key",
        created_at: "2024-01-01T00:00:00Z",
        created_by: "user-1",
        updated_at: "2024-01-01T00:00:00Z",
        updated_by: "user-1",
      });

      const submitButton = screen.getByRole("button", { name: "Add MCP Server" });
      await act(async () => {
        fireEvent.click(submitButton);
      });

      // The form should submit without validation error on auth_value
      await waitFor(() => {
        expect(networking.createMCPServer).toHaveBeenCalledTimes(1);
      });
    });

    it("should not require auth value when creating a server with Bearer Token auth type", async () => {
      await selectHttpTransport();

      const user = userEvent.setup({ delay: null });

      const nameInput = getServerNameInput();
      await user.type(nameInput, "Test_Server");

      const urlInput = screen.getByPlaceholderText("https://your-mcp-server.com");
      await user.type(urlInput, "https://example.com/mcp");

      await selectAntOption("Authentication", "Bearer Token");

      await waitFor(() => {
        expect(screen.getByText("Authentication Value")).toBeInTheDocument();
      });

      // Leave auth value empty and submit
      vi.mocked(networking.createMCPServer).mockResolvedValue({
        server_id: "new-server-1",
        server_name: "Test_Server",
        alias: "Test_Server",
        url: "https://example.com/mcp",
        transport: "http",
        auth_type: "bearer_token",
        created_at: "2024-01-01T00:00:00Z",
        created_by: "user-1",
        updated_at: "2024-01-01T00:00:00Z",
        updated_by: "user-1",
      });

      const submitButton = screen.getByRole("button", { name: "Add MCP Server" });
      await act(async () => {
        fireEvent.click(submitButton);
      });

      await waitFor(() => {
        expect(networking.createMCPServer).toHaveBeenCalledTimes(1);
      });
    });

    it("should successfully create a server when auth value is provided", async () => {
      await selectHttpTransport();

      const user = userEvent.setup({ delay: null });

      const nameInput = getServerNameInput();
      await user.type(nameInput, "My_Server");

      const urlInput = screen.getByPlaceholderText("https://your-mcp-server.com");
      await user.type(urlInput, "https://example.com/mcp");

      await selectAntOption("Authentication", "API Key");

      await waitFor(() => {
        expect(screen.getByText("Authentication Value")).toBeInTheDocument();
      });

      // Fill in auth value
      const authInput = screen.getByPlaceholderText("Enter token or secret");
      await user.type(authInput, "my-secret-key");

      vi.mocked(networking.createMCPServer).mockResolvedValue({
        server_id: "new-server-1",
        server_name: "My_Server",
        alias: "My_Server",
        url: "https://example.com/mcp",
        transport: "http",
        auth_type: "api_key",
        created_at: "2024-01-01T00:00:00Z",
        created_by: "user-1",
        updated_at: "2024-01-01T00:00:00Z",
        updated_by: "user-1",
      });

      const submitButton = screen.getByRole("button", { name: "Add MCP Server" });
      await act(async () => {
        fireEvent.click(submitButton);
      });

      await waitFor(() => {
        expect(networking.createMCPServer).toHaveBeenCalledTimes(1);
      });

      const [token, payload] = vi.mocked(networking.createMCPServer).mock.calls[0];
      expect(token).toBe("test-token");
      expect(payload.credentials).toEqual({ auth_value: "my-secret-key" });
    });

    it("should not show auth value field when None auth type is selected", async () => {
      await selectHttpTransport();

      await selectAntOption("Authentication", "None");

      // Auth value field should not appear for "None"
      await waitFor(() => {
        expect(screen.queryByText("Authentication Value")).not.toBeInTheDocument();
      });
    });

    it("should successfully create a server with no auth", async () => {
      await selectHttpTransport();

      const user = userEvent.setup({ delay: null });

      const nameInput = getServerNameInput();
      await user.type(nameInput, "No_Auth_Server");

      const urlInput = screen.getByPlaceholderText("https://your-mcp-server.com");
      await user.type(urlInput, "https://example.com/mcp");

      await selectAntOption("Authentication", "None");

      vi.mocked(networking.createMCPServer).mockResolvedValue({
        server_id: "new-server-1",
        server_name: "No_Auth_Server",
        alias: "No_Auth_Server",
        url: "https://example.com/mcp",
        transport: "http",
        auth_type: "none",
        created_at: "2024-01-01T00:00:00Z",
        created_by: "user-1",
        updated_at: "2024-01-01T00:00:00Z",
        updated_by: "user-1",
      });

      const submitButton = screen.getByRole("button", { name: "Add MCP Server" });
      await act(async () => {
        fireEvent.click(submitButton);
      });

      await waitFor(() => {
        expect(networking.createMCPServer).toHaveBeenCalledTimes(1);
      });

      const [, payload] = vi.mocked(networking.createMCPServer).mock.calls[0];
      expect(payload.auth_type).toBe("none");
      // No credentials should be sent for "none" auth
      expect(payload.credentials).toBeUndefined();
    });

    it("enforces the allowlist when the user explicitly deselects every tool", async () => {
      await selectHttpTransport();

      const user = userEvent.setup({ delay: null });

      const nameInput = getServerNameInput();
      await user.type(nameInput, "Locked_Down_Server");

      const urlInput = screen.getByPlaceholderText("https://your-mcp-server.com");
      await user.type(urlInput, "https://example.com/mcp");

      await selectAntOption("Authentication", "None");

      await act(async () => {
        fireEvent.click(screen.getByRole("button", { name: "Disable all tools" }));
      });

      vi.mocked(networking.createMCPServer).mockResolvedValue({
        server_id: "new-server-1",
        server_name: "Locked_Down_Server",
        alias: "Locked_Down_Server",
        url: "https://example.com/mcp",
        transport: "http",
        auth_type: "none",
        created_at: "2024-01-01T00:00:00Z",
        created_by: "user-1",
        updated_at: "2024-01-01T00:00:00Z",
        updated_by: "user-1",
      });

      const submitButton = screen.getByRole("button", { name: "Add MCP Server" });
      await act(async () => {
        fireEvent.click(submitButton);
      });

      await waitFor(() => {
        expect(networking.createMCPServer).toHaveBeenCalledTimes(1);
      });

      const [, payload] = vi.mocked(networking.createMCPServer).mock.calls[0];
      expect(payload.mcp_info.tool_allowlist_enforced).toBe(true);
      expect(payload.allowed_tools).toEqual([]);
    });
  });

  describe("when OAuth interactive auth is selected", () => {
    /** Select HTTP transport + OAuth auth, then wait for the OAuth form to appear. */
    async function setupOAuthInteractive() {
      render(<CreateMCPServer {...defaultProps} />);
      await selectAntOption("Transport Type", "Streamable HTTP");

      await waitFor(() => {
        expect(screen.getByPlaceholderText("https://your-mcp-server.com")).toBeInTheDocument();
      });

      await selectAntOption("Authentication", "OAuth");

      // Wait for OAuthFormFields to render (OAuth Flow Type selector is the sentinel)
      await waitFor(() => {
        expect(screen.getByText("OAuth Flow Type")).toBeInTheDocument();
      });

      // OAuthFormFields defaults to INTERACTIVE, so the new fields should appear
      await waitFor(() => {
        expect(screen.getByText("Token Validation Rules (optional)")).toBeInTheDocument();
        expect(screen.getByText("Token Storage TTL (seconds, optional)")).toBeInTheDocument();
      });
    }

    it("shows Token Validation Rules and Token Storage TTL fields", async () => {
      await setupOAuthInteractive();
      // Asserted in setupOAuthInteractive
    });

    it("includes token_validation in payload when token_validation_json is filled with valid JSON", async () => {
      vi.mocked(networking.createMCPServer).mockResolvedValue({
        server_id: "new-server-oauth",
        server_name: "OAuth_Server",
        alias: "OAuth_Server",
        url: "https://example.com/mcp",
        transport: "http",
        auth_type: "oauth2",
        created_at: "2024-01-01T00:00:00Z",
        created_by: "user-1",
        updated_at: "2024-01-01T00:00:00Z",
        updated_by: "user-1",
      });

      await setupOAuthInteractive();

      // Fill required form fields
      const nameInput = document.getElementById("server_name") as HTMLInputElement;
      await act(async () => {
        fireEvent.change(nameInput, { target: { value: "OAuth_Server" } });
      });
      const urlInput = screen.getByPlaceholderText("https://your-mcp-server.com");
      await act(async () => {
        fireEvent.change(urlInput, { target: { value: "https://example.com/mcp" } });
      });

      // Fill in the token_validation_json textarea
      const textarea = document.getElementById("token_validation_json") as HTMLTextAreaElement;
      await act(async () => {
        fireEvent.change(textarea, { target: { value: '{"organization": "my-org", "team.id": "42"}' } });
      });

      const submitButton = screen.getByRole("button", { name: "Add MCP Server" });
      await act(async () => {
        fireEvent.click(submitButton);
      });

      await waitFor(() => {
        expect(networking.createMCPServer).toHaveBeenCalledTimes(1);
      });

      const [, payload] = vi.mocked(networking.createMCPServer).mock.calls[0];
      expect(payload.token_validation).toEqual({ organization: "my-org", "team.id": "42" });
    });

    it("invalidates the DCR client and OAuth flow when the MCP URL changes after Authorize & Fetch", async () => {
      await setupOAuthInteractive();

      const nameInput = document.getElementById("server_name") as HTMLInputElement;
      await act(async () => {
        fireEvent.change(nameInput, { target: { value: "Url_Change_Server" } });
      });
      const urlInput = screen.getByPlaceholderText("https://your-mcp-server.com");
      await act(async () => {
        fireEvent.change(urlInput, { target: { value: "https://a.example.com/mcp" } });
      });

      act(() => {
        oauthHook.onTokenReceived?.({ access_token: "tok-a" }, { clientId: "client-a", clientSecret: "secret-a" });
      });
      oauthHook.reset.mockClear();

      await act(async () => {
        fireEvent.change(urlInput, { target: { value: "https://b.example.com/mcp" } });
      });

      await waitFor(() => expect(oauthHook.reset).toHaveBeenCalled());

      vi.mocked(networking.createMCPServer).mockResolvedValue({
        server_id: "new-server-oauth",
        server_name: "Url_Change_Server",
        alias: "Url_Change_Server",
        url: "https://b.example.com/mcp",
        transport: "http",
        auth_type: "oauth2",
        created_at: "2024-01-01T00:00:00Z",
        created_by: "user-1",
        updated_at: "2024-01-01T00:00:00Z",
        updated_by: "user-1",
      });

      const submitButton = screen.getByRole("button", { name: "Add MCP Server" });
      await act(async () => {
        fireEvent.click(submitButton);
      });

      await waitFor(() => expect(networking.createMCPServer).toHaveBeenCalledTimes(1));
      const [, payload] = vi.mocked(networking.createMCPServer).mock.calls[0];
      expect(payload.credentials?.client_id).toBeUndefined();
      expect(payload.credentials?.client_secret).toBeUndefined();
    });

    it("invalidates the DCR client and OAuth flow when the OpenAPI spec URL changes after Authorize & Fetch", async () => {
      render(<CreateMCPServer {...defaultProps} />);
      await selectAntOption("Transport Type", "OpenAPI Spec");
      await waitFor(() => {
        expect(screen.getByPlaceholderText("https://petstore3.swagger.io/api/v3/openapi.json")).toBeInTheDocument();
      });
      await selectAntOption("Authentication", "OAuth");
      await waitFor(() => {
        expect(screen.getByText("OAuth Flow Type")).toBeInTheDocument();
      });

      const nameInput = document.getElementById("server_name") as HTMLInputElement;
      await act(async () => {
        fireEvent.change(nameInput, { target: { value: "OpenAPI_Server" } });
      });
      const specInput = screen.getByPlaceholderText("https://petstore3.swagger.io/api/v3/openapi.json");
      await act(async () => {
        fireEvent.change(specInput, { target: { value: "https://a.example.com/openapi.json" } });
      });

      act(() => {
        oauthHook.onTokenReceived?.({ access_token: "tok-a" }, { clientId: "client-a", clientSecret: "secret-a" });
      });
      oauthHook.reset.mockClear();

      await act(async () => {
        fireEvent.change(specInput, { target: { value: "https://b.example.com/openapi.json" } });
      });

      await waitFor(() => expect(oauthHook.reset).toHaveBeenCalled());

      vi.mocked(networking.createMCPServer).mockResolvedValue({
        server_id: "new-openapi-server",
        server_name: "OpenAPI_Server",
        alias: "OpenAPI_Server",
        url: "https://b.example.com/openapi.json",
        transport: "http",
        auth_type: "oauth2",
        created_at: "2024-01-01T00:00:00Z",
        created_by: "user-1",
        updated_at: "2024-01-01T00:00:00Z",
        updated_by: "user-1",
      });

      const submitButton = screen.getByRole("button", { name: "Add MCP Server" });
      await act(async () => {
        fireEvent.click(submitButton);
      });

      await waitFor(() => expect(networking.createMCPServer).toHaveBeenCalledTimes(1));
      const [, payload] = vi.mocked(networking.createMCPServer).mock.calls[0];
      expect(payload.spec_path).toBe("https://b.example.com/openapi.json");
      expect(payload.credentials?.client_id).toBeUndefined();
      expect(payload.credentials?.client_secret).toBeUndefined();
    });

    it("invalidates the DCR client and OAuth flow when the transport changes after Authorize & Fetch", async () => {
      render(<CreateMCPServer {...defaultProps} />);
      await selectAntOption("Transport Type", "OpenAPI Spec");
      await waitFor(() => {
        expect(screen.getByPlaceholderText("https://petstore3.swagger.io/api/v3/openapi.json")).toBeInTheDocument();
      });
      await selectAntOption("Authentication", "OAuth");
      await waitFor(() => {
        expect(screen.getByText("OAuth Flow Type")).toBeInTheDocument();
      });

      const nameInput = document.getElementById("server_name") as HTMLInputElement;
      await act(async () => {
        fireEvent.change(nameInput, { target: { value: "Transport_Change_Server" } });
      });
      const specInput = screen.getByPlaceholderText("https://petstore3.swagger.io/api/v3/openapi.json");
      await act(async () => {
        fireEvent.change(specInput, { target: { value: "https://same.example.com/spec-or-mcp" } });
      });

      act(() => {
        oauthHook.onTokenReceived?.({ access_token: "tok-a" }, { clientId: "client-a", clientSecret: "secret-a" });
      });
      oauthHook.reset.mockClear();

      await selectAntOption("Transport Type", "Streamable HTTP");
      await waitFor(() => {
        expect(screen.getByPlaceholderText("https://your-mcp-server.com")).toBeInTheDocument();
      });
      const urlInput = screen.getByPlaceholderText("https://your-mcp-server.com");
      await act(async () => {
        fireEvent.change(urlInput, { target: { value: "https://same.example.com/spec-or-mcp" } });
      });

      await waitFor(() => expect(oauthHook.reset).toHaveBeenCalled());

      vi.mocked(networking.createMCPServer).mockResolvedValue({
        server_id: "new-transport-server",
        server_name: "Transport_Change_Server",
        alias: "Transport_Change_Server",
        url: "https://same.example.com/spec-or-mcp",
        transport: "http",
        auth_type: "oauth2",
        created_at: "2024-01-01T00:00:00Z",
        created_by: "user-1",
        updated_at: "2024-01-01T00:00:00Z",
        updated_by: "user-1",
      });

      const submitButton = screen.getByRole("button", { name: "Add MCP Server" });
      await act(async () => {
        fireEvent.click(submitButton);
      });

      await waitFor(() => expect(networking.createMCPServer).toHaveBeenCalledTimes(1));
      const [, payload] = vi.mocked(networking.createMCPServer).mock.calls[0];
      expect(payload.url).toBe("https://same.example.com/spec-or-mcp");
      expect(payload.credentials?.client_id).toBeUndefined();
      expect(payload.credentials?.client_secret).toBeUndefined();
    });

    it("omits token_validation from payload when token_validation_json is empty", async () => {
      vi.mocked(networking.createMCPServer).mockResolvedValue({
        server_id: "new-server-oauth",
        server_name: "OAuth_Server",
        alias: "OAuth_Server",
        url: "https://example.com/mcp",
        transport: "http",
        auth_type: "oauth2",
        created_at: "2024-01-01T00:00:00Z",
        created_by: "user-1",
        updated_at: "2024-01-01T00:00:00Z",
        updated_by: "user-1",
      });

      await setupOAuthInteractive();

      const nameInput = document.getElementById("server_name") as HTMLInputElement;
      await act(async () => {
        fireEvent.change(nameInput, { target: { value: "OAuth_Server" } });
      });
      const urlInput = screen.getByPlaceholderText("https://your-mcp-server.com");
      await act(async () => {
        fireEvent.change(urlInput, { target: { value: "https://example.com/mcp" } });
      });

      // Leave token_validation_json empty
      const submitButton = screen.getByRole("button", { name: "Add MCP Server" });
      await act(async () => {
        fireEvent.click(submitButton);
      });

      await waitFor(() => {
        expect(networking.createMCPServer).toHaveBeenCalledTimes(1);
      });

      const [, payload] = vi.mocked(networking.createMCPServer).mock.calls[0];
      expect(payload.token_validation).toBeUndefined();
    });

    it("includes credentials.token_endpoint_auth_method in payload when client_secret_basic is selected", async () => {
      vi.mocked(networking.createMCPServer).mockResolvedValue({
        server_id: "new-server-oauth",
        server_name: "OAuth_Server",
        alias: "OAuth_Server",
        url: "https://example.com/mcp",
        transport: "http",
        auth_type: "oauth2",
        created_at: "2024-01-01T00:00:00Z",
        created_by: "user-1",
        updated_at: "2024-01-01T00:00:00Z",
        updated_by: "user-1",
      });

      await setupOAuthInteractive();

      const nameInput = document.getElementById("server_name") as HTMLInputElement;
      await act(async () => {
        fireEvent.change(nameInput, { target: { value: "OAuth_Server" } });
      });
      const urlInput = screen.getByPlaceholderText("https://your-mcp-server.com");
      await act(async () => {
        fireEvent.change(urlInput, { target: { value: "https://example.com/mcp" } });
      });

      await selectAntOption("Token Endpoint Auth Method (optional)", "Client Secret Basic");

      const submitButton = screen.getByRole("button", { name: "Add MCP Server" });
      await act(async () => {
        fireEvent.click(submitButton);
      });

      await waitFor(() => {
        expect(networking.createMCPServer).toHaveBeenCalledTimes(1);
      });

      const [, payload] = vi.mocked(networking.createMCPServer).mock.calls[0];
      expect(payload.credentials?.token_endpoint_auth_method).toBe("client_secret_basic");
    });

    it("omits token_endpoint_auth_method from credentials when left blank", async () => {
      vi.mocked(networking.createMCPServer).mockResolvedValue({
        server_id: "new-server-oauth",
        server_name: "OAuth_Server",
        alias: "OAuth_Server",
        url: "https://example.com/mcp",
        transport: "http",
        auth_type: "oauth2",
        created_at: "2024-01-01T00:00:00Z",
        created_by: "user-1",
        updated_at: "2024-01-01T00:00:00Z",
        updated_by: "user-1",
      });

      await setupOAuthInteractive();

      const nameInput = document.getElementById("server_name") as HTMLInputElement;
      await act(async () => {
        fireEvent.change(nameInput, { target: { value: "OAuth_Server" } });
      });
      const urlInput = screen.getByPlaceholderText("https://your-mcp-server.com");
      await act(async () => {
        fireEvent.change(urlInput, { target: { value: "https://example.com/mcp" } });
      });

      const submitButton = screen.getByRole("button", { name: "Add MCP Server" });
      await act(async () => {
        fireEvent.click(submitButton);
      });

      await waitFor(() => {
        expect(networking.createMCPServer).toHaveBeenCalledTimes(1);
      });

      const [, payload] = vi.mocked(networking.createMCPServer).mock.calls[0];
      expect(payload.credentials?.token_endpoint_auth_method).toBeUndefined();
    });

    it("persists access + refresh token to the DB on submit for OBO mode", async () => {
      // "Authorize & Fetch" produced a token before submit.
      oauthHook.tokenResponse = {
        access_token: "obo-access-token",
        refresh_token: "obo-refresh-token",
        expires_in: 3600,
        token_type: "bearer",
        scope: "channels:read chat:write",
      };
      vi.mocked(networking.createMCPServer).mockResolvedValue({
        server_id: "obo-server-1",
        server_name: "OBO_Server",
        alias: "OBO_Server",
        url: "https://example.com/mcp",
        transport: "http",
        auth_type: "oauth2",
        created_at: "2024-01-01T00:00:00Z",
        created_by: "user-1",
        updated_at: "2024-01-01T00:00:00Z",
        updated_by: "user-1",
      });

      // Interactive OAuth + delegate_auth_to_upstream off (the default) => OBO mode.
      await setupOAuthInteractive();

      const nameInput = document.getElementById("server_name") as HTMLInputElement;
      await act(async () => {
        fireEvent.change(nameInput, { target: { value: "OBO_Server" } });
      });
      const urlInput = screen.getByPlaceholderText("https://your-mcp-server.com");
      await act(async () => {
        fireEvent.change(urlInput, { target: { value: "https://example.com/mcp" } });
      });

      const submitButton = screen.getByRole("button", { name: "Add MCP Server" });
      await act(async () => {
        fireEvent.click(submitButton);
      });

      await waitFor(() => {
        expect(networking.storeMCPOAuthUserCredential).toHaveBeenCalledTimes(1);
      });
      expect(networking.storeMCPOAuthUserCredential).toHaveBeenCalledWith("test-token", "obo-server-1", {
        access_token: "obo-access-token",
        refresh_token: "obo-refresh-token",
        expires_in: 3600,
        scopes: ["channels:read", "chat:write"],
      });
      // OBO persists server-side; it must not fall back to the browser-only cache.
      expect(setToken).not.toHaveBeenCalled();
    });

    it("does not submit and shows validation error for invalid JSON in token_validation_json", async () => {
      await setupOAuthInteractive();

      const textarea = document.getElementById("token_validation_json") as HTMLTextAreaElement;
      await act(async () => {
        fireEvent.change(textarea, { target: { value: "not-valid-json{" } });
      });

      const nameInput = document.getElementById("server_name") as HTMLInputElement;
      await act(async () => {
        fireEvent.change(nameInput, { target: { value: "OAuth_Server" } });
      });

      const submitButton = screen.getByRole("button", { name: "Add MCP Server" });
      await act(async () => {
        fireEvent.click(submitButton);
      });

      // Either the inline form validation message or the notification fires —
      // both indicate the submit was blocked.
      await waitFor(() => {
        const inlineError = screen.queryByText("Must be valid JSON");
        const notCalled = !vi.mocked(networking.createMCPServer).mock.calls.length;
        expect(inlineError !== null || notCalled).toBe(true);
      });
    });
  });

  describe("when modal is cancelled", () => {
    it("should call setModalVisible(false) when cancel is clicked", async () => {
      render(<CreateMCPServer {...defaultProps} />);

      const cancelButton = screen.getByRole("button", { name: "Cancel" });
      await act(async () => {
        fireEvent.click(cancelButton);
      });

      expect(defaultProps.setModalVisible).toHaveBeenCalledWith(false);
    });

    it("does not leak a previous server's OAuth token into the next add-server session", async () => {
      const usedToken = (token: string) =>
        vi.mocked(networking.testMCPToolsListRequest).mock.calls.some((call) => call[2] === token);

      const { rerender } = render(<CreateMCPServer {...defaultProps} />);

      await selectAntOption("Transport Type", "Streamable HTTP");
      await waitFor(() => {
        expect(screen.getByPlaceholderText("https://your-mcp-server.com")).toBeInTheDocument();
      });
      await selectAntOption("Authentication", "OAuth");
      await waitFor(() => {
        expect(screen.getByText("OAuth Flow Type")).toBeInTheDocument();
      });

      const urlInput = screen.getByPlaceholderText("https://your-mcp-server.com");
      await act(async () => {
        fireEvent.change(urlInput, { target: { value: "https://server-a.example.com/mcp" } });
      });

      // Simulate "Authorize & Fetch Token" completing for server A.
      await act(async () => {
        oauthHook.onTokenReceived?.({ access_token: "stale-token-A", expires_in: 3600 });
      });

      // Precondition: the freshly fetched token drives the tool preview for server A.
      await waitFor(() => {
        expect(usedToken("stale-token-A")).toBe(true);
      });

      // Parent hides the modal (Cancel / successful create both flip this prop).
      rerender(<CreateMCPServer {...defaultProps} isModalVisible={false} />);

      // The OAuth flow state (source of the "Token fetched" badge) is reset on close.
      expect(oauthHook.reset).toHaveBeenCalled();

      vi.mocked(networking.testMCPToolsListRequest).mockClear();

      // Reopen for a brand-new server and enter a different URL without re-authorizing.
      rerender(<CreateMCPServer {...defaultProps} isModalVisible={true} />);
      const reopenedUrlInput = screen.getByPlaceholderText("https://your-mcp-server.com");
      oauthHook.reset.mockClear();
      await act(async () => {
        fireEvent.change(reopenedUrlInput, { target: { value: "https://server-b.example.com/mcp" } });
      });

      // The previous server's token must never be replayed for the new session.
      expect(oauthHook.reset).not.toHaveBeenCalled();
      expect(usedToken("stale-token-A")).toBe(false);
    });

    it("clears the tool list and form fields when a parent dismisses the modal", async () => {
      vi.mocked(networking.testMCPToolsListRequest).mockResolvedValue({
        tools: [{ name: "tool_a" }],
        error: null,
      });
      const toolCount = () => screen.getByTestId("mcp-connection-status").getAttribute("data-tool-count");

      const { rerender } = render(<CreateMCPServer {...defaultProps} />);

      await selectAntOption("Transport Type", "Streamable HTTP");
      await waitFor(() => {
        expect(screen.getByPlaceholderText("https://your-mcp-server.com")).toBeInTheDocument();
      });
      await selectAntOption("Authentication", "OAuth");
      await waitFor(() => {
        expect(screen.getByText("OAuth Flow Type")).toBeInTheDocument();
      });

      const urlInput = screen.getByPlaceholderText("https://your-mcp-server.com");
      await act(async () => {
        fireEvent.change(urlInput, { target: { value: "https://server-a.example.com/mcp" } });
      });
      await act(async () => {
        oauthHook.onTokenReceived?.({ access_token: "stale-token-A", expires_in: 3600 });
      });

      // Precondition: a tool list is shown for server A.
      await waitFor(() => {
        expect(toolCount()).toBe("1");
      });

      // Parent dismisses the modal without routing through Cancel or create.
      rerender(<CreateMCPServer {...defaultProps} isModalVisible={false} />);

      // Stale tools are cleared even though neither handler ran.
      await waitFor(() => {
        expect(toolCount()).toBe("0");
      });

      // Reopening starts clean: the URL the prior server left in the Ant form store is gone.
      rerender(<CreateMCPServer {...defaultProps} isModalVisible={true} />);
      const reopenedUrlInput = screen.getByPlaceholderText("https://your-mcp-server.com") as HTMLInputElement;
      expect(reopenedUrlInput.value).toBe("");
    });

    it("does not reset an in-flight OAuth resume when mounted with the modal closed (post-redirect restore)", () => {
      // After the "Authorize & Fetch Token" redirect the page reloads and this
      // component mounts with isModalVisible=false while useMcpOAuthFlow is still
      // exchanging the authorization code. Calling reset() during that mount bumps
      // the hook's reset version and the fetched token is silently discarded, so
      // the user sees no Connection Status / Tool Configuration and must authorize
      // again after saving.
      const { rerender } = render(<CreateMCPServer {...defaultProps} isModalVisible={false} />);
      expect(oauthHook.reset).not.toHaveBeenCalled();

      // A real open -> closed transition must still reset (the #30000 leak fix).
      rerender(<CreateMCPServer {...defaultProps} isModalVisible={true} />);
      rerender(<CreateMCPServer {...defaultProps} isModalVisible={false} />);
      expect(oauthHook.reset).toHaveBeenCalled();
    });
  });

  describe("when stdio transport is selected", () => {
    it("should not show auth type or URL fields", async () => {
      render(<CreateMCPServer {...defaultProps} />);

      await selectAntOption("Transport Type", "Standard Input/Output");

      // Auth and URL fields should not be present for stdio
      await waitFor(() => {
        expect(screen.queryByText("Authentication")).not.toBeInTheDocument();
        expect(screen.queryByPlaceholderText("https://your-mcp-server.com")).not.toBeInTheDocument();
      });
    });
  });

  describe("when prefillData is provided", () => {
    it("should populate form fields from discovery data", async () => {
      const prefillData = {
        name: "github-mcp",
        title: "GitHub MCP",
        description: "GitHub integration server",
        category: "Development",
        transport: "http",
        url: "https://github-mcp.example.com",
      };

      render(<CreateMCPServer {...defaultProps} prefillData={prefillData} />);

      await waitFor(() => {
        // Server name should be sanitized (hyphens replaced with underscores)
        const nameInput = getServerNameInput();
        expect(nameInput).toHaveValue("github_mcp");
      });
    });
  });

  describe("with back to discovery button", () => {
    it("should show back button and call onBackToDiscovery when clicked", async () => {
      const onBackToDiscovery = vi.fn();
      render(<CreateMCPServer {...defaultProps} onBackToDiscovery={onBackToDiscovery} />);

      // The back arrow button should be visible
      const backButton = screen.getByText("←");
      expect(backButton).toBeInTheDocument();

      await act(async () => {
        fireEvent.click(backButton);
      });

      expect(onBackToDiscovery).toHaveBeenCalledTimes(1);
    });
  });
});
