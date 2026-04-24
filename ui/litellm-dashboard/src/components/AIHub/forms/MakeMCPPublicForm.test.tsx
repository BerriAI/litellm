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

    expect(screen.getByText("Make MCP Servers Public")).toBeInTheDocument();
    expect(screen.getByText("Select MCP Servers to Make Public")).toBeInTheDocument();

    const checkboxes = screen.getAllByRole("checkbox");
    expect(checkboxes).toHaveLength(3);

    const nextButton = screen.getByRole("button", { name: "Next" });
    expect(nextButton).not.toBeDisabled();
  });

  it("should handle server selection and navigation", async () => {
    render(<MakeMCPPublicForm {...mockProps} />);

    expect(screen.getByText("Select MCP Servers to Make Public")).toBeInTheDocument();

    const selectAllCheckbox = screen.getByLabelText("Select All (2)");
    await act(async () => {
      fireEvent.click(selectAllCheckbox);
    });

    const nextButton = screen.getByRole("button", { name: "Next" });
    expect(nextButton).not.toBeDisabled();

    await act(async () => {
      fireEvent.click(nextButton);
    });

    await waitFor(() => {
      expect(screen.getByText("Confirm Making MCP Servers Public")).toBeInTheDocument();
    });
  });

  it("should submit selected servers successfully", async () => {
    mockMakeMCPPublicCall.mockResolvedValueOnce({});

    render(<MakeMCPPublicForm {...mockProps} />);

    const selectAllCheckbox = screen.getByLabelText("Select All (2)");
    await act(async () => {
      fireEvent.click(selectAllCheckbox);
    });

    const nextButton = screen.getByRole("button", { name: "Next" });
    await act(async () => {
      fireEvent.click(nextButton);
    });

    await waitFor(() => {
      expect(screen.getByText("Confirm Making MCP Servers Public")).toBeInTheDocument();
    });

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

    await act(async () => {
      fireEvent.click(selectAllCheckbox);
    });

    checkboxes.forEach((checkbox) => {
      expect(checkbox).toBeChecked();
    });

    await act(async () => {
      fireEvent.click(selectAllCheckbox);
    });

    expect(checkboxes[0]).not.toBeChecked();
    expect(checkboxes[1]).not.toBeChecked();
    expect(checkboxes[2]).not.toBeChecked();
  });

  it("should show error when no servers selected", async () => {
    render(<MakeMCPPublicForm {...mockProps} />);

    const checkboxes = screen.getAllByRole("checkbox");
    await act(async () => {
      fireEvent.click(checkboxes[0]);
    });
    await act(async () => {
      fireEvent.click(checkboxes[0]);
    });

    const nextButton = screen.getByRole("button", { name: "Next" });
    await act(async () => {
      fireEvent.click(nextButton);
    });

    expect(screen.getByText("Select MCP Servers to Make Public")).toBeInTheDocument();
  });

  it("should display empty state when no servers are available", () => {
    const emptyProps = {
      ...mockProps,
      mcpHubData: [] as MCPServerData[],
    };

    render(<MakeMCPPublicForm {...emptyProps} />);

    expect(screen.getByText("No MCP servers available.")).toBeInTheDocument();

    const selectAllCheckbox = screen.getByLabelText("Select All");
    expect(selectAllCheckbox).toBeDisabled();

    const nextButton = screen.getByRole("button", { name: "Next" });
    expect(nextButton).toBeDisabled();
  });

  it("should handle Cancel button functionality", async () => {
    render(<MakeMCPPublicForm {...mockProps} />);

    const cancelButton = screen.getByRole("button", { name: "Cancel" });
    await act(async () => {
      fireEvent.click(cancelButton);
    });

    expect(mockProps.onClose).toHaveBeenCalled();
  });

  it("should handle Previous button functionality", async () => {
    render(<MakeMCPPublicForm {...mockProps} />);

    const nextButton = screen.getByRole("button", { name: "Next" });
    await act(async () => {
      fireEvent.click(nextButton);
    });

    await waitFor(() => {
      expect(screen.getByText("Confirm Making MCP Servers Public")).toBeInTheDocument();
    });

    const previousButton = screen.getByRole("button", { name: "Previous" });
    await act(async () => {
      fireEvent.click(previousButton);
    });

    expect(screen.getByText("Select MCP Servers to Make Public")).toBeInTheDocument();
  });

  it("should handle individual server selection", async () => {
    render(<MakeMCPPublicForm {...mockProps} />);

    const checkboxes = screen.getAllByRole("checkbox");
    expect(checkboxes).toHaveLength(3);

    const server1Checkbox = checkboxes[1];
    const server2Checkbox = checkboxes[2];

    expect(server2Checkbox).toBeChecked();

    await act(async () => {
      fireEvent.click(server1Checkbox);
    });

    expect(server1Checkbox).toBeChecked();
    expect(server2Checkbox).toBeChecked();

    await act(async () => {
      fireEvent.click(server2Checkbox);
    });

    expect(server1Checkbox).toBeChecked();
    expect(server2Checkbox).not.toBeChecked();

    const selectAllCheckbox = checkboxes[0];
    expect(selectAllCheckbox).toHaveAttribute("data-state", "indeterminate");
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

    expect(screen.getByText("tool-1")).toBeInTheDocument();
    expect(screen.getByText("tool-2")).toBeInTheDocument();
    expect(screen.getByText("tool-3")).toBeInTheDocument();
    expect(screen.getByText("+2 more")).toBeInTheDocument();
  });

  it("should handle submit error properly", async () => {
    const errorMessage = "Network error";
    mockMakeMCPPublicCall.mockRejectedValueOnce(new Error(errorMessage));

    render(<MakeMCPPublicForm {...mockProps} />);

    const nextButton = screen.getByRole("button", { name: "Next" });
    await act(async () => {
      fireEvent.click(nextButton);
    });

    await waitFor(() => {
      expect(screen.getByText("Confirm Making MCP Servers Public")).toBeInTheDocument();
    });

    const submitButton = screen.getByRole("button", { name: "Make Public" });
    await act(async () => {
      fireEvent.click(submitButton);
    });

    await waitFor(() => {
      expect(mockMakeMCPPublicCall).toHaveBeenCalledWith("test-token", ["server-2"]);
    });

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

    const nextButton = screen.getByRole("button", { name: "Next" });
    await act(async () => {
      fireEvent.click(nextButton);
    });

    await waitFor(() => {
      expect(screen.getByText("Confirm Making MCP Servers Public")).toBeInTheDocument();
    });

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Make Public" }));
    });

    const loadingButton = screen.getByRole("button", { name: "Making Public..." });
    expect(loadingButton).toHaveAttribute("data-loading", "true");
    expect(loadingButton).toBeDisabled();

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

    expect(screen.queryByText("Make MCP Servers Public")).not.toBeInTheDocument();
  });

  it("should preselect already public servers when modal opens", () => {
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
          mcp_info: { is_public: false },
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
        {
          server_id: "server-3",
          server_name: "Test Server 3",
          description: "Description 3",
          url: "http://example.com/server3",
          transport: "sse",
          status: "healthy",
          mcp_info: { is_public: true },
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

    const checkboxes = screen.getAllByRole("checkbox");
    expect(checkboxes).toHaveLength(4);

    const server1Checkbox = checkboxes[1];
    const server2Checkbox = checkboxes[2];
    const server3Checkbox = checkboxes[3];

    expect(server1Checkbox).not.toBeChecked();
    expect(server2Checkbox).toBeChecked();
    expect(server3Checkbox).toBeChecked();

    const selectAllCheckbox = checkboxes[0];
    expect(selectAllCheckbox).toHaveAttribute("data-state", "indeterminate");
  });
});
