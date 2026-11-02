import { render, screen, fireEvent, act, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import MakeMCPPublicForm from "./MakeMCPPublicForm";
import userEvent from "@testing-library/user-event";
import { toast } from "@/lib/toast";
import { MCPServerData } from "@/components/AIHub/MCPHubTableColumns";

// Mock the networking function
vi.mock("../../networking", () => ({
  makeMCPPublicCall: vi.fn(),
}));

vi.mock("@/lib/toast", () => ({
  toast: { success: vi.fn(), fromError: vi.fn() },
}));

// Import the mocked function
import { makeMCPPublicCall } from "../../networking";
const mockMakeMCPPublicCall = vi.mocked(makeMCPPublicCall);

const expectDisabledControl = (element: HTMLElement) =>
  expect(element.hasAttribute("disabled") || element.getAttribute("aria-disabled") === "true").toBe(true);

describe("MakeMCPPublicForm", () => {
  const mockProps = {
    visible: true,
    onClose: vi.fn(),
    accessToken: "test-token",
    mcpHubData: [
      {
        server_id: "server-1",
        server_name: "Test Server 1",
        description: "Description 1",
        url: "http://example.com/server1",
        transport: "http",
        status: "active",
        mcp_info: { is_public: false, is_public_explicit: false },
        allowed_tools: ["tool-1", "tool-2"],
        auth_type: "bearer",
        credentials: {},
        created_at: "2024-01-01T00:00:00Z",
        created_by: "user1",
        updated_at: "2024-01-01T00:00:00Z",
        updated_by: "user1",
        teams: [],
        mcp_access_groups: [],
        extra_headers: [],
        static_headers: {},
        args: [],
        env: {},
      },
      {
        server_id: "server-2",
        server_name: "Test Server 2",
        description: "Description 2",
        url: "http://example.com/server2",
        transport: "websocket",
        status: "inactive",
        mcp_info: { is_public: true, is_public_explicit: true },
        allowed_tools: [],
        auth_type: "none",
        credentials: {},
        created_at: "2024-01-01T00:00:00Z",
        created_by: "user2",
        updated_at: "2024-01-01T00:00:00Z",
        updated_by: "user2",
        teams: [],
        mcp_access_groups: [],
        extra_headers: [],
        static_headers: {},
        args: [],
        env: {},
      },
    ] as MCPServerData[],
    onSuccess: vi.fn(),
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.resetAllMocks();
  });

  it("should render the component", () => {
    render(<MakeMCPPublicForm {...mockProps} />);

    expect(screen.getByText("Manage MCP Hub Visibility")).toBeInTheDocument();
    expect(screen.getByText("Select MCP Servers for the Hub")).toBeInTheDocument();
  });

  it("should initialize with correct state", () => {
    render(<MakeMCPPublicForm {...mockProps} />);

    // Check that the component renders with the correct title and content
    expect(screen.getByText("Manage MCP Hub Visibility")).toBeInTheDocument();
    expect(screen.getByText("Select MCP Servers for the Hub")).toBeInTheDocument();

    // Check that all server checkboxes are present
    const checkboxes = screen.getAllByRole("checkbox");
    expect(checkboxes).toHaveLength(3); // Select all + 2 servers

    // Check that the Next button is enabled (servers are preselected)
    const nextButton = screen.getByRole("button", { name: "Next" });
    expect(nextButton).toBeEnabled();
  });

  it("should handle server selection and navigation", async () => {
    render(<MakeMCPPublicForm {...mockProps} />);

    // Initially on step 1
    expect(screen.getByText("Select MCP Servers for the Hub")).toBeInTheDocument();

    // Select all servers using the select all checkbox
    const selectAllCheckbox = screen.getByRole("checkbox", { name: "Select All (2)" });
    await act(async () => {
      fireEvent.click(selectAllCheckbox);
    });

    // Verify Next button is enabled
    const nextButton = screen.getByRole("button", { name: "Next" });
    expect(nextButton).toBeEnabled();

    // Click Next
    await act(async () => {
      fireEvent.click(nextButton);
    });

    // Should move to step 2
    await waitFor(() => {
      expect(screen.getByText("Confirm MCP Hub Publication")).toBeInTheDocument();
    });
  });

  it("should submit selected servers successfully", async () => {
    mockMakeMCPPublicCall.mockResolvedValueOnce({});

    render(<MakeMCPPublicForm {...mockProps} />);

    // Select all servers
    const selectAllCheckbox = screen.getByRole("checkbox", { name: "Select All (2)" });
    await act(async () => {
      fireEvent.click(selectAllCheckbox);
    });

    const nextButton = screen.getByRole("button", { name: "Next" });
    await act(async () => {
      fireEvent.click(nextButton);
    });

    // Wait for navigation to complete
    await waitFor(() => {
      expect(screen.getByText("Confirm MCP Hub Publication")).toBeInTheDocument();
    });

    const submitButton = screen.getByRole("button", { name: "Save Publication List" });
    await act(async () => {
      fireEvent.click(submitButton);
    });

    await waitFor(() => {
      expect(mockMakeMCPPublicCall).toHaveBeenCalledWith("test-token", ["server-1", "server-2"]);
      expect(mockProps.onSuccess).toHaveBeenCalled();
      expect(mockProps.onClose).toHaveBeenCalled();
    });
  });

  it("should handle select all functionality", async () => {
    render(<MakeMCPPublicForm {...mockProps} />);

    const checkboxes = screen.getAllByRole("checkbox");
    const selectAllCheckbox = checkboxes[0];

    // Select all
    await act(async () => {
      fireEvent.click(selectAllCheckbox);
    });

    // All checkboxes should be checked
    checkboxes.forEach((checkbox) => {
      expect(checkbox).toBeChecked();
    });

    // Deselect all
    await act(async () => {
      fireEvent.click(selectAllCheckbox);
    });

    // All checkboxes should be unchecked except the indeterminate state
    expect(checkboxes[0]).not.toBeChecked();
    expect(checkboxes[1]).not.toBeChecked();
    expect(checkboxes[2]).not.toBeChecked();
  });

  it("submits an empty publication list after the last server is deselected", async () => {
    mockMakeMCPPublicCall.mockResolvedValueOnce({});
    render(<MakeMCPPublicForm {...mockProps} />);

    fireEvent.click(screen.getAllByRole("checkbox")[2]);
    expect(screen.getByRole("button", { name: "Next" })).toBeEnabled();
    fireEvent.click(screen.getByRole("button", { name: "Next" }));
    fireEvent.click(screen.getByRole("button", { name: "Save Publication List" }));

    await waitFor(() => expect(mockMakeMCPPublicCall).toHaveBeenCalledWith("test-token", []));
    expect(mockProps.onSuccess).toHaveBeenCalled();
  });

  it("keeps legacy listings separate from explicitly published selections", () => {
    render(
      <MakeMCPPublicForm
        {...mockProps}
        mcpHubData={[
          { ...mockProps.mcpHubData[0], mcp_info: { is_public: true, is_public_explicit: false } },
          { ...mockProps.mcpHubData[1], mcp_info: { is_public: true, is_public_explicit: true } },
        ]}
      />,
    );

    expect(screen.getAllByRole("checkbox")[1]).not.toBeChecked();
    expect(screen.getAllByRole("checkbox")[2]).toBeChecked();
    expect(screen.getByText("Listed by legacy mode")).toBeInTheDocument();
  });

  it.each([
    { mode: "all missing, stale true", info: { is_public: true }, mixed: false },
    { mode: "all missing, stale false", info: { is_public: false }, mixed: false },
    { mode: "mixed, stale true", info: { is_public: true }, mixed: true },
    { mode: "mixed, stale false", info: { is_public: false }, mixed: true },
    { mode: "null explicit status", info: { is_public: true, is_public_explicit: null }, mixed: true },
    { mode: "nonboolean explicit status", info: { is_public: true, is_public_explicit: "true" }, mixed: true },
  ])("blocks unknown explicit publication metadata: $mode", ({ info, mixed }) => {
    const unknownServer = { ...mockProps.mcpHubData[0], mcp_info: info };
    const catalog = mixed ? [unknownServer, mockProps.mcpHubData[1]] : [unknownServer];
    render(<MakeMCPPublicForm {...mockProps} mcpHubData={catalog} />);

    expect(screen.getByRole("alert")).toHaveTextContent("explicit publication status");
    expect(screen.queryByRole("checkbox")).not.toBeInTheDocument();
    expect(screen.queryByText("Configure in YAML")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Copy code" })).not.toBeInTheDocument();
    const nextButton = screen.getByRole("button", { name: "Next" });
    expect(nextButton).toBeDisabled();
    fireEvent.click(nextButton);
    expect(screen.queryByText("Confirm MCP Hub Publication")).not.toBeInTheDocument();
    expect(mockMakeMCPPublicCall).not.toHaveBeenCalled();
  });

  it.each([true, false])("blocks confirmation when explicit metadata disappears with stale listing %s", (listed) => {
    const { rerender } = render(<MakeMCPPublicForm {...mockProps} />);
    fireEvent.click(screen.getByRole("button", { name: "Next" }));
    expect(screen.getByRole("button", { name: "Save Publication List" })).toBeEnabled();

    const catalog = [{ ...mockProps.mcpHubData[0], mcp_info: { is_public: listed } }, mockProps.mcpHubData[1]];
    rerender(<MakeMCPPublicForm {...mockProps} mcpHubData={catalog} />);

    expect(screen.getByRole("alert")).toHaveTextContent("explicit publication status");
    expect(screen.queryByText("Confirm MCP Hub Publication")).not.toBeInTheDocument();
    const saveButton = screen.getByRole("button", { name: "Save Publication List" });
    expect(saveButton).toBeDisabled();
    fireEvent.click(saveButton);
    expect(mockMakeMCPPublicCall).not.toHaveBeenCalled();
    expect(screen.queryByRole("button", { name: "Copy code" })).not.toBeInTheDocument();

    const refreshedCatalog = [
      { ...mockProps.mcpHubData[0], mcp_info: { is_public: true, is_public_explicit: true } },
      { ...mockProps.mcpHubData[1], mcp_info: { is_public: false, is_public_explicit: false } },
    ];
    rerender(<MakeMCPPublicForm {...mockProps} mcpHubData={refreshedCatalog} />);
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Next" })).toBeEnabled();
    expect(screen.getByRole("checkbox", { name: "Publish Test Server 1" })).toBeChecked();
    expect(screen.getByRole("checkbox", { name: "Publish Test Server 2" })).not.toBeChecked();
  });

  it("copies publication YAML using the selected server IDs", async () => {
    const user = userEvent.setup();
    render(<MakeMCPPublicForm {...mockProps} />);

    await user.click(screen.getByText("Configure in YAML"));
    await user.click(screen.getByRole("button", { name: "Copy code" }));

    expect(await navigator.clipboard.readText()).toBe(
      'litellm_settings:\n  public_mcp_hub_strict_whitelist: true\n  public_mcp_servers:\n    - "server-2"',
    );

    await user.click(screen.getByRole("checkbox", { name: "Publish Test Server 2" }));
    await user.click(screen.getByRole("button", { name: "Copy code" }));
    expect(await navigator.clipboard.readText()).toBe(
      "litellm_settings:\n  public_mcp_hub_strict_whitelist: true\n  public_mcp_servers: []",
    );
  });

  it("allows clearing publication IDs when the loaded server catalog is empty", async () => {
    mockMakeMCPPublicCall.mockResolvedValueOnce({});
    const emptyProps = {
      ...mockProps,
      mcpHubData: [] as MCPServerData[],
    };

    render(<MakeMCPPublicForm {...emptyProps} />);

    expect(screen.getByText("No MCP servers available.")).toBeInTheDocument();

    // Select All checkbox should be disabled
    const selectAllCheckbox = screen.getByRole("checkbox", { name: "Select All" });
    expectDisabledControl(selectAllCheckbox);

    const nextButton = screen.getByRole("button", { name: "Next" });
    expect(nextButton).toBeEnabled();
    fireEvent.click(nextButton);
    fireEvent.click(screen.getByRole("button", { name: "Save Publication List" }));

    await waitFor(() => expect(mockMakeMCPPublicCall).toHaveBeenCalledWith("test-token", []));
    expect(mockProps.onSuccess).toHaveBeenCalled();
  });

  it("should handle Cancel button functionality", async () => {
    render(<MakeMCPPublicForm {...mockProps} />);

    // Click Cancel button
    const cancelButton = screen.getByRole("button", { name: "Cancel" });
    await act(async () => {
      fireEvent.click(cancelButton);
    });

    // Should call onClose
    expect(mockProps.onClose).toHaveBeenCalled();
  });

  it("should handle Previous button functionality", async () => {
    render(<MakeMCPPublicForm {...mockProps} />);

    // Navigate to step 1
    const nextButton = screen.getByRole("button", { name: "Next" });
    await act(async () => {
      fireEvent.click(nextButton);
    });

    // Verify we're on step 1
    await waitFor(() => {
      expect(screen.getByText("Confirm MCP Hub Publication")).toBeInTheDocument();
    });

    // Click Previous button
    const previousButton = screen.getByRole("button", { name: "Previous" });
    await act(async () => {
      fireEvent.click(previousButton);
    });

    // Should go back to step 0
    expect(screen.getByText("Select MCP Servers for the Hub")).toBeInTheDocument();
  });

  it("should handle individual server selection", async () => {
    render(<MakeMCPPublicForm {...mockProps} />);

    // Get all checkboxes (select all + individual servers)
    const checkboxes = screen.getAllByRole("checkbox");
    expect(checkboxes).toHaveLength(3); // Select all + 2 servers

    // Initially, server-2 should be selected (it's already public)
    const server1Checkbox = checkboxes[1]; // First server checkbox
    const server2Checkbox = checkboxes[2]; // Second server checkbox

    expect(server2Checkbox).toBeChecked(); // server-2 is already public

    // Select server-1
    await act(async () => {
      fireEvent.click(server1Checkbox);
    });

    expect(server1Checkbox).toBeChecked();
    expect(server2Checkbox).toBeChecked();

    // Deselect server-2
    await act(async () => {
      fireEvent.click(server2Checkbox);
    });

    expect(server1Checkbox).toBeChecked();
    expect(server2Checkbox).not.toBeChecked();

    // Select all should be indeterminate now
    const selectAllCheckbox = checkboxes[0];
    expect(selectAllCheckbox).toBePartiallyChecked();
  });

  it("should display tools overflow text when server has more than 3 tools", () => {
    const serverWithManyTools = {
      ...mockProps.mcpHubData[0],
      allowed_tools: ["tool-1", "tool-2", "tool-3", "tool-4", "tool-5"],
    };

    const propsWithManyTools = {
      ...mockProps,
      mcpHubData: [serverWithManyTools],
    };

    render(<MakeMCPPublicForm {...propsWithManyTools} />);

    // Should show first 3 tools as badges
    expect(screen.getByText("tool-1")).toBeInTheDocument();
    expect(screen.getByText("tool-2")).toBeInTheDocument();
    expect(screen.getByText("tool-3")).toBeInTheDocument();

    // Should show "+2 more" text for the remaining tools
    expect(screen.getByText("+2 more")).toBeInTheDocument();
  });

  it("should handle submit error properly", async () => {
    const error = new Error("Update litellm_settings.public_mcp_servers in your YAML configuration");
    mockMakeMCPPublicCall.mockRejectedValueOnce(error);

    render(<MakeMCPPublicForm {...mockProps} />);

    const nextButton = screen.getByRole("button", { name: "Next" });
    await act(async () => {
      fireEvent.click(nextButton);
    });

    await waitFor(() => {
      expect(screen.getByText("Confirm MCP Hub Publication")).toBeInTheDocument();
    });

    const submitButton = screen.getByRole("button", { name: "Save Publication List" });
    await act(async () => {
      fireEvent.click(submitButton);
    });

    // Should handle error and show error notification
    await waitFor(() => {
      expect(mockMakeMCPPublicCall).toHaveBeenCalledWith("test-token", ["server-2"]);
    });

    expect(toast.fromError).toHaveBeenCalledWith(error);

    // Should not call onSuccess or onClose on error
    expect(mockProps.onSuccess).not.toHaveBeenCalled();
    expect(mockProps.onClose).not.toHaveBeenCalled();
  });

  it("should not complete the flow until the submit request resolves", async () => {
    let resolvePromise: (value: any) => void = () => {};
    const pendingPromise = new Promise((resolve) => {
      resolvePromise = resolve;
    });
    mockMakeMCPPublicCall.mockReturnValueOnce(pendingPromise);

    render(<MakeMCPPublicForm {...mockProps} />);

    const nextButton = screen.getByRole("button", { name: "Next" });
    await act(async () => {
      fireEvent.click(nextButton);
    });

    await waitFor(() => {
      expect(screen.getByText("Confirm MCP Hub Publication")).toBeInTheDocument();
    });

    const submitButton = screen.getByRole("button", { name: "Save Publication List" });
    await act(async () => {
      fireEvent.click(submitButton);
    });

    expectDisabledControl(submitButton);
    await act(async () => {
      fireEvent.click(submitButton);
    });
    expect(mockMakeMCPPublicCall).toHaveBeenCalledTimes(1);
    expect(mockProps.onSuccess).not.toHaveBeenCalled();
    expect(mockProps.onClose).not.toHaveBeenCalled();
    expect(screen.getByText("Confirm MCP Hub Publication")).toBeInTheDocument();

    resolvePromise({});
    await waitFor(() => {
      expect(mockProps.onSuccess).toHaveBeenCalled();
      expect(mockProps.onClose).toHaveBeenCalled();
    });
  });

  it("should not render modal when visible is false", () => {
    const invisibleProps = {
      ...mockProps,
      visible: false,
    };

    render(<MakeMCPPublicForm {...invisibleProps} />);

    // Modal should not be rendered
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(screen.queryByText("Manage MCP Hub Visibility")).not.toBeInTheDocument();
  });

  it("should preselect already public servers when modal opens", () => {
    // Test data where one server is public and one is not
    const mixedPublicProps = {
      ...mockProps,
      mcpHubData: [
        {
          server_id: "server-1",
          server_name: "Test Server 1",
          description: "Description 1",
          url: "http://example.com/server1",
          transport: "http",
          status: "active",
          mcp_info: { is_public: false, is_public_explicit: false }, // Not public
          allowed_tools: [],
          auth_type: "bearer",
          credentials: {},
          created_at: "2024-01-01T00:00:00Z",
          created_by: "user1",
          updated_at: "2024-01-01T00:00:00Z",
          updated_by: "user1",
          teams: [],
          mcp_access_groups: [],
          extra_headers: [],
          static_headers: {},
          args: [],
          env: {},
        },
        {
          server_id: "server-2",
          server_name: "Test Server 2",
          description: "Description 2",
          url: "http://example.com/server2",
          transport: "websocket",
          status: "inactive",
          mcp_info: { is_public: true, is_public_explicit: true }, // Already public
          allowed_tools: [],
          auth_type: "none",
          credentials: {},
          created_at: "2024-01-01T00:00:00Z",
          created_by: "user2",
          updated_at: "2024-01-01T00:00:00Z",
          updated_by: "user2",
          teams: [],
          mcp_access_groups: [],
          extra_headers: [],
          static_headers: {},
          args: [],
          env: {},
        },
        {
          server_id: "server-3",
          server_name: "Test Server 3",
          description: "Description 3",
          url: "http://example.com/server3",
          transport: "sse",
          status: "healthy",
          mcp_info: { is_public: true, is_public_explicit: true }, // Already public
          allowed_tools: [],
          auth_type: "oauth",
          credentials: {},
          created_at: "2024-01-01T00:00:00Z",
          created_by: "user3",
          updated_at: "2024-01-01T00:00:00Z",
          updated_by: "user3",
          teams: [],
          mcp_access_groups: [],
          extra_headers: [],
          static_headers: {},
          args: [],
          env: {},
        },
      ] as MCPServerData[],
    };

    render(<MakeMCPPublicForm {...mixedPublicProps} />);

    // Check that the correct checkboxes are selected
    const checkboxes = screen.getAllByRole("checkbox");
    expect(checkboxes).toHaveLength(4); // Select all + 3 servers

    // server-2 and server-3 should be checked (they're already public)
    const server1Checkbox = checkboxes[1];
    const server2Checkbox = checkboxes[2];
    const server3Checkbox = checkboxes[3];

    expect(server1Checkbox).not.toBeChecked(); // server-1 is not public
    expect(server2Checkbox).toBeChecked(); // server-2 is public
    expect(server3Checkbox).toBeChecked(); // server-3 is public

    // Select all should be indeterminate
    const selectAllCheckbox = checkboxes[0];
    expect(selectAllCheckbox).toBePartiallyChecked();
  });
});
