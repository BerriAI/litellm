import { render, screen, fireEvent, act, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import MakeMCPPublicForm from "./MakeMCPPublicForm";
import { MCPServerData } from "../../mcp_hub_table_columns";

// Mock the networking function
vi.mock("../../networking", () => ({
  makeMCPPublicCall: vi.fn(),
}));

// Import the mocked function
import { makeMCPPublicCall } from "../../networking";
const mockMakeMCPPublicCall = vi.mocked(makeMCPPublicCall);

// Mock antd components
vi.mock("antd", () => ({
  Modal: ({ open, title, children, onCancel, footer }: any) =>
    open ? (
      <div data-testid="modal">
        <div>{title}</div>
        {children}
        {footer}
      </div>
    ) : null,
  Form: Object.assign(({ children, form }: any) => <form data-testid="form">{children}</form>, {
    useForm: () => [
      {
        resetFields: vi.fn(),
        validateFields: vi.fn(),
        getFieldsValue: vi.fn(),
        setFieldsValue: vi.fn(),
      },
      vi.fn(),
    ],
    Item: ({ children }: any) => <div>{children}</div>,
  }),
  Steps: Object.assign(
    ({ children, current, className }: any) => (
      <div data-testid="steps" className={className}>
        {children}
      </div>
    ),
    {
      Step: ({ title }: any) => <div>{title}</div>,
    },
  ),
  Button: ({ children, onClick, disabled, loading, ...props }: any) => (
    <button onClick={onClick} disabled={disabled || loading} data-loading={loading} {...props}>
      {children}
    </button>
  ),
  Checkbox: ({ checked, indeterminate, onChange, children, disabled }: any) => (
    <label>
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange({ target: { checked: e.target.checked } })}
        disabled={disabled}
        data-indeterminate={indeterminate}
      />
      {children}
    </label>
  ),
}));

// Additional @tremor/react mocks (Button is already mocked globally)
vi.mock("@tremor/react", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@tremor/react")>();
  return {
    ...actual,
    Text: ({ children, className }: any) => <span className={className}>{children}</span>,
    Title: ({ children }: any) => <h3>{children}</h3>,
    Badge: ({ children, color, size }: any) => (
      <span data-color={color} data-size={size}>
        {children}
      </span>
    ),
  };
});

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
        mcp_info: { is_public: false },
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
        mcp_info: { is_public: true },
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

    expect(screen.getByText("Make MCP Servers Public")).toBeInTheDocument();
    expect(screen.getByText("Select MCP Servers to Make Public")).toBeInTheDocument();
  });

  it("should initialize with correct state", () => {
    render(<MakeMCPPublicForm {...mockProps} />);

    // Check that the component renders with the correct title and content
    expect(screen.getByText("Make MCP Servers Public")).toBeInTheDocument();
    expect(screen.getByText("Select MCP Servers to Make Public")).toBeInTheDocument();

    // Check that all server checkboxes are present
    const checkboxes = screen.getAllByRole("checkbox");
    expect(checkboxes).toHaveLength(3); // Select all + 2 servers

    // Check that the Next button is enabled (servers are preselected)
    const nextButton = screen.getByRole("button", { name: "Next" });
    expect(nextButton).not.toBeDisabled();
  });

  it("should handle server selection and navigation", async () => {
    render(<MakeMCPPublicForm {...mockProps} />);

    // Initially on step 1
    expect(screen.getByText("Select MCP Servers to Make Public")).toBeInTheDocument();

    // Select all servers using the select all checkbox
    const selectAllCheckbox = screen.getByLabelText("Select All (2)");
    await act(async () => {
      fireEvent.click(selectAllCheckbox);
    });

    // Verify Next button is enabled
    const nextButton = screen.getByRole("button", { name: "Next" });
    expect(nextButton).not.toBeDisabled();

    // Click Next
    await act(async () => {
      fireEvent.click(nextButton);
    });

    // Should move to step 2
    await waitFor(() => {
      expect(screen.getByText("Confirm Making MCP Servers Public")).toBeInTheDocument();
    });
  });

  it("should submit selected servers successfully", async () => {
    mockMakeMCPPublicCall.mockResolvedValueOnce({});

    render(<MakeMCPPublicForm {...mockProps} />);

    // Select all servers
    const selectAllCheckbox = screen.getByLabelText("Select All (2)");
    await act(async () => {
      fireEvent.click(selectAllCheckbox);
    });

    // Navigate to confirm step
    const nextButton = screen.getByRole("button", { name: "Next" });
    await act(async () => {
      fireEvent.click(nextButton);
    });

    // Wait for navigation to complete
    await waitFor(() => {
      expect(screen.getByText("Confirm Making MCP Servers Public")).toBeInTheDocument();
    });

    // Submit
    const submitButton = screen.getByRole("button", { name: "Make Public" });
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

  it("should show error when no servers selected", async () => {
    render(<MakeMCPPublicForm {...mockProps} />);

    // Deselect all servers first
    const checkboxes = screen.getAllByRole("checkbox");
    await act(async () => {
      fireEvent.click(checkboxes[0]); // Click select all to select all
      fireEvent.click(checkboxes[0]); // Click select all again to deselect all
    });

    // Try to go to next step
    const nextButton = screen.getByRole("button", { name: "Next" });
    await act(async () => {
      fireEvent.click(nextButton);
    });

    // Should stay on same step
    expect(screen.getByText("Select MCP Servers to Make Public")).toBeInTheDocument();
  });

  it("should display empty state when no servers are available", () => {
    const emptyProps = {
      ...mockProps,
      mcpHubData: [] as MCPServerData[],
    };

    render(<MakeMCPPublicForm {...emptyProps} />);

    expect(screen.getByText("No MCP servers available.")).toBeInTheDocument();

    // Select All checkbox should be disabled
    const selectAllCheckbox = screen.getByLabelText("Select All");
    expect(selectAllCheckbox).toBeDisabled();

    // Next button should be disabled
    const nextButton = screen.getByRole("button", { name: "Next" });
    expect(nextButton).toBeDisabled();
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
      expect(screen.getByText("Confirm Making MCP Servers Public")).toBeInTheDocument();
    });

    // Click Previous button
    const previousButton = screen.getByRole("button", { name: "Previous" });
    await act(async () => {
      fireEvent.click(previousButton);
    });

    // Should go back to step 0
    expect(screen.getByText("Select MCP Servers to Make Public")).toBeInTheDocument();
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
    expect(selectAllCheckbox).toHaveAttribute("data-indeterminate", "true");
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
    const errorMessage = "Network error";
    mockMakeMCPPublicCall.mockRejectedValueOnce(new Error(errorMessage));

    render(<MakeMCPPublicForm {...mockProps} />);

    // Navigate to confirm step
    const nextButton = screen.getByRole("button", { name: "Next" });
    await act(async () => {
      fireEvent.click(nextButton);
    });

    await waitFor(() => {
      expect(screen.getByText("Confirm Making MCP Servers Public")).toBeInTheDocument();
    });

    // Submit
    const submitButton = screen.getByRole("button", { name: "Make Public" });
    await act(async () => {
      fireEvent.click(submitButton);
    });

    // Should handle error and show error notification
    await waitFor(() => {
      expect(mockMakeMCPPublicCall).toHaveBeenCalledWith("test-token", ["server-2"]);
    });

    // Should not call onSuccess or onClose on error
    expect(mockProps.onSuccess).not.toHaveBeenCalled();
    expect(mockProps.onClose).not.toHaveBeenCalled();
  });

  it("should show loading state during submit", async () => {
    let resolvePromise: (value: any) => void = () => {};
    const pendingPromise = new Promise((resolve) => {
      resolvePromise = resolve;
    });
    mockMakeMCPPublicCall.mockReturnValueOnce(pendingPromise);

    render(<MakeMCPPublicForm {...mockProps} />);

    // Navigate to confirm step
    const nextButton = screen.getByRole("button", { name: "Next" });
    await act(async () => {
      fireEvent.click(nextButton);
    });

    await waitFor(() => {
      expect(screen.getByText("Confirm Making MCP Servers Public")).toBeInTheDocument();
    });

    // Submit
    const submitButton = screen.getByRole("button", { name: "Make Public" });
    await act(async () => {
      fireEvent.click(submitButton);
    });

    // Check loading state
    expect(submitButton).toHaveAttribute("data-loading", "true");
    expect(submitButton).toBeDisabled();

    // Resolve the promise
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
    expect(screen.queryByTestId("modal")).not.toBeInTheDocument();
    expect(screen.queryByText("Make MCP Servers Public")).not.toBeInTheDocument();
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
          mcp_info: { is_public: false }, // Not public
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
          mcp_info: { is_public: true }, // Already public
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
          mcp_info: { is_public: true }, // Already public
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
    expect(selectAllCheckbox).toHaveAttribute("data-indeterminate", "true");
  });
});
