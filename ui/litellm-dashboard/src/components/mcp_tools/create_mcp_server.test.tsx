import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import * as networking from "../networking";
import CreateMCPServer from "./create_mcp_server";

vi.mock("../networking", () => ({
  createMCPServer: vi.fn(),
  testMCPToolsListRequest: vi.fn().mockResolvedValue({ tools: [], error: null }),
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

vi.mock("./mcp_connection_status", () => ({
  default: () => <div data-testid="mcp-connection-status" />,
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

/** Helper: select a dropdown option by opening a select near a label and clicking an option */
async function selectAntOption(labelText: string, optionText: string) {
  const label = screen.getByText(labelText);
  const formItem = label.closest(".ant-form-item")!;
  const select = formItem.querySelector(".ant-select");
  act(() => {
    fireEvent.mouseDown(select!.querySelector(".ant-select-selector")!);
  });

  await waitFor(() => {
    const options = document.querySelectorAll(".ant-select-item-option");
    expect(options.length).toBeGreaterThan(0);
  });

  const option = Array.from(document.querySelectorAll(".ant-select-item-option")).find((el) =>
    el.textContent?.includes(optionText),
  );
  expect(option).toBeTruthy();
  act(() => {
    fireEvent.click(option!);
  });
}

describe("CreateMCPServer", () => {
  beforeEach(() => {
    vi.clearAllMocks();
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

      const user = userEvent.setup();

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

      const user = userEvent.setup();

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

      const user = userEvent.setup();

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

      const user = userEvent.setup();

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
