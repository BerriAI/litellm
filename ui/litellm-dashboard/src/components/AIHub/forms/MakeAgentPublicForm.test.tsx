import { render, screen, fireEvent, act, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import MakeAgentPublicForm from "./MakeAgentPublicForm";
import { AgentHubData } from "@/components/AIHub/AgentHubTableColumns";

// Mock the networking function
vi.mock("../../networking", () => ({
  makeAgentsPublicCall: vi.fn(),
}));

// Import the mocked function
import { makeAgentsPublicCall } from "../../networking";
const mockMakeAgentsPublicCall = vi.mocked(makeAgentsPublicCall);

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

// Mock @tremor/react components
vi.mock("@tremor/react", () => ({
  Text: ({ children, className }: any) => <span className={className}>{children}</span>,
  Title: ({ children }: any) => <h3>{children}</h3>,
  Badge: ({ children, color, size }: any) => (
    <span data-color={color} data-size={size}>
      {children}
    </span>
  ),
}));

describe("MakeAgentPublicForm", () => {
  const mockProps = {
    visible: true,
    onClose: vi.fn(),
    accessToken: "test-token",
    agentHubData: [
      {
        agent_id: "agent-1",
        name: "Test Agent 1",
        description: "Description 1",
        version: "1.0",
        is_public: false,
        skills: [
          { id: "skill-1", name: "Skill 1", description: "Skill desc" },
          { id: "skill-2", name: "Skill 2", description: "Skill desc" },
        ],
        protocolVersion: "1.0",
      },
      {
        agent_id: "agent-2",
        name: "Test Agent 2",
        description: "Description 2",
        version: "2.0",
        is_public: true,
        skills: [],
        protocolVersion: "1.0",
      },
    ] as AgentHubData[],
    onSuccess: vi.fn(),
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.resetAllMocks();
  });

  it("should render the component", () => {
    render(<MakeAgentPublicForm {...mockProps} />);

    expect(screen.getByText("Make Agents Public")).toBeInTheDocument();
    expect(screen.getByText("Select Agents to Make Public")).toBeInTheDocument();
  });

  it("should initialize with correct state", () => {
    render(<MakeAgentPublicForm {...mockProps} />);

    // Check that the component renders with the correct title and content
    expect(screen.getByText("Make Agents Public")).toBeInTheDocument();
    expect(screen.getByText("Select Agents to Make Public")).toBeInTheDocument();

    // Check that all agent checkboxes are present
    const checkboxes = screen.getAllByRole("checkbox");
    expect(checkboxes).toHaveLength(3); // Select all + 2 agents

    // Check that the Next button is enabled (agents are preselected)
    const nextButton = screen.getByRole("button", { name: "Next" });
    expect(nextButton).not.toBeDisabled();
  });

  it("should handle agent selection and navigation", async () => {
    render(<MakeAgentPublicForm {...mockProps} />);

    // Initially on step 1
    expect(screen.getByText("Select Agents to Make Public")).toBeInTheDocument();

    // Select all agents using the select all checkbox
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
      expect(screen.getByText("Confirm Making Agents Public")).toBeInTheDocument();
    });
  });

  it("should submit selected agents successfully", async () => {
    mockMakeAgentsPublicCall.mockResolvedValueOnce({});

    render(<MakeAgentPublicForm {...mockProps} />);

    // Select all agents
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
      expect(screen.getByText("Confirm Making Agents Public")).toBeInTheDocument();
    });

    // Submit
    const submitButton = screen.getByRole("button", { name: "Make Public" });
    await act(async () => {
      fireEvent.click(submitButton);
    });

    await waitFor(() => {
      expect(mockMakeAgentsPublicCall).toHaveBeenCalledWith("test-token", ["agent-1", "agent-2"]);
      expect(mockProps.onSuccess).toHaveBeenCalled();
      expect(mockProps.onClose).toHaveBeenCalled();
    });
  });

  it("should handle select all functionality", async () => {
    render(<MakeAgentPublicForm {...mockProps} />);

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

  it("should show error when no agents selected", async () => {
    render(<MakeAgentPublicForm {...mockProps} />);

    // Deselect all agents first
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
    expect(screen.getByText("Select Agents to Make Public")).toBeInTheDocument();
  });

  it("should display empty state when no agents are available", () => {
    const emptyProps = {
      ...mockProps,
      agentHubData: [] as AgentHubData[],
    };

    render(<MakeAgentPublicForm {...emptyProps} />);

    expect(screen.getByText("No agents available.")).toBeInTheDocument();

    // Select All checkbox should be disabled
    const selectAllCheckbox = screen.getByLabelText("Select All");
    expect(selectAllCheckbox).toBeDisabled();

    // Next button should be disabled
    const nextButton = screen.getByRole("button", { name: "Next" });
    expect(nextButton).toBeDisabled();
  });

  it("should handle Cancel button functionality", async () => {
    render(<MakeAgentPublicForm {...mockProps} />);

    // Click Cancel button
    const cancelButton = screen.getByRole("button", { name: "Cancel" });
    await act(async () => {
      fireEvent.click(cancelButton);
    });

    // Should call onClose
    expect(mockProps.onClose).toHaveBeenCalled();
  });

  it("should handle Previous button functionality", async () => {
    render(<MakeAgentPublicForm {...mockProps} />);

    // Navigate to step 1
    const nextButton = screen.getByRole("button", { name: "Next" });
    await act(async () => {
      fireEvent.click(nextButton);
    });

    // Verify we're on step 1
    await waitFor(() => {
      expect(screen.getByText("Confirm Making Agents Public")).toBeInTheDocument();
    });

    // Click Previous button
    const previousButton = screen.getByRole("button", { name: "Previous" });
    await act(async () => {
      fireEvent.click(previousButton);
    });

    // Should go back to step 0
    expect(screen.getByText("Select Agents to Make Public")).toBeInTheDocument();
  });

  it("should handle individual agent selection", async () => {
    render(<MakeAgentPublicForm {...mockProps} />);

    // Get all checkboxes (select all + individual agents)
    const checkboxes = screen.getAllByRole("checkbox");
    expect(checkboxes).toHaveLength(3); // Select all + 2 agents

    // Initially, agent-2 should be selected (it's already public)
    const agent1Checkbox = checkboxes[1]; // First agent checkbox
    const agent2Checkbox = checkboxes[2]; // Second agent checkbox

    expect(agent2Checkbox).toBeChecked(); // agent-2 is already public

    // Select agent-1
    await act(async () => {
      fireEvent.click(agent1Checkbox);
    });

    expect(agent1Checkbox).toBeChecked();
    expect(agent2Checkbox).toBeChecked();

    // Deselect agent-2
    await act(async () => {
      fireEvent.click(agent2Checkbox);
    });

    expect(agent1Checkbox).toBeChecked();
    expect(agent2Checkbox).not.toBeChecked();

    // Select all should be indeterminate now
    const selectAllCheckbox = checkboxes[0];
    expect(selectAllCheckbox).toHaveAttribute("data-indeterminate", "true");
  });

  it("should display skills overflow text when agent has more than 3 skills", () => {
    const agentWithManySkills = {
      ...mockProps.agentHubData[0],
      skills: [
        { id: "skill-1", name: "Skill 1", description: "Skill desc" },
        { id: "skill-2", name: "Skill 2", description: "Skill desc" },
        { id: "skill-3", name: "Skill 3", description: "Skill desc" },
        { id: "skill-4", name: "Skill 4", description: "Skill desc" },
        { id: "skill-5", name: "Skill 5", description: "Skill desc" },
      ],
    };

    const propsWithManySkills = {
      ...mockProps,
      agentHubData: [agentWithManySkills],
    };

    render(<MakeAgentPublicForm {...propsWithManySkills} />);

    // Should show first 3 skills as badges
    expect(screen.getByText("Skill 1")).toBeInTheDocument();
    expect(screen.getByText("Skill 2")).toBeInTheDocument();
    expect(screen.getByText("Skill 3")).toBeInTheDocument();

    // Should show "+2 more" text for the remaining skills
    expect(screen.getByText("+2 more")).toBeInTheDocument();
  });

  it("should handle submit error properly", async () => {
    const errorMessage = "Network error";
    mockMakeAgentsPublicCall.mockRejectedValueOnce(new Error(errorMessage));

    render(<MakeAgentPublicForm {...mockProps} />);

    // Navigate to confirm step
    const nextButton = screen.getByRole("button", { name: "Next" });
    await act(async () => {
      fireEvent.click(nextButton);
    });

    await waitFor(() => {
      expect(screen.getByText("Confirm Making Agents Public")).toBeInTheDocument();
    });

    // Submit
    const submitButton = screen.getByRole("button", { name: "Make Public" });
    await act(async () => {
      fireEvent.click(submitButton);
    });

    // Should handle error and show error notification
    await waitFor(() => {
      expect(mockMakeAgentsPublicCall).toHaveBeenCalledWith("test-token", ["agent-2"]);
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
    mockMakeAgentsPublicCall.mockReturnValueOnce(pendingPromise);

    render(<MakeAgentPublicForm {...mockProps} />);

    // Navigate to confirm step
    const nextButton = screen.getByRole("button", { name: "Next" });
    await act(async () => {
      fireEvent.click(nextButton);
    });

    await waitFor(() => {
      expect(screen.getByText("Confirm Making Agents Public")).toBeInTheDocument();
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

    render(<MakeAgentPublicForm {...invisibleProps} />);

    // Modal should not be rendered
    expect(screen.queryByTestId("modal")).not.toBeInTheDocument();
    expect(screen.queryByText("Make Agents Public")).not.toBeInTheDocument();
  });

  it("should preselect already public agents when modal opens", () => {
    // Test data where one agent is public and one is not
    const mixedPublicProps = {
      ...mockProps,
      agentHubData: [
        {
          agent_id: "agent-1",
          name: "Test Agent 1",
          description: "Description 1",
          url: "http://example.com/agent1",
          version: "1.0",
          is_public: false, // Not public
          skills: [],
          protocolVersion: "1.0",
        },
        {
          agent_id: "agent-2",
          name: "Test Agent 2",
          description: "Description 2",
          url: "http://example.com/agent2",
          version: "2.0",
          is_public: true, // Already public
          skills: [],
          protocolVersion: "1.0",
        },
        {
          agent_id: "agent-3",
          name: "Test Agent 3",
          description: "Description 3",
          url: "http://example.com/agent3",
          version: "3.0",
          is_public: true, // Already public
          skills: [],
          protocolVersion: "1.0",
        },
      ] as AgentHubData[],
    };

    render(<MakeAgentPublicForm {...mixedPublicProps} />);

    // Check that the correct checkboxes are selected
    const checkboxes = screen.getAllByRole("checkbox");
    expect(checkboxes).toHaveLength(4); // Select all + 3 agents

    // agent-2 and agent-3 should be checked (they're already public)
    const agent1Checkbox = checkboxes[1];
    const agent2Checkbox = checkboxes[2];
    const agent3Checkbox = checkboxes[3];

    expect(agent1Checkbox).not.toBeChecked(); // agent-1 is not public
    expect(agent2Checkbox).toBeChecked(); // agent-2 is public
    expect(agent3Checkbox).toBeChecked(); // agent-3 is public

    // Select all should be indeterminate
    const selectAllCheckbox = checkboxes[0];
    expect(selectAllCheckbox).toHaveAttribute("data-indeterminate", "true");
  });
});
