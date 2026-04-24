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

    expect(screen.getByText("Make Agents Public")).toBeInTheDocument();
    expect(screen.getByText("Select Agents to Make Public")).toBeInTheDocument();

    // Select all + 2 agents
    const checkboxes = screen.getAllByRole("checkbox");
    expect(checkboxes).toHaveLength(3);

    const nextButton = screen.getByRole("button", { name: "Next" });
    expect(nextButton).not.toBeDisabled();
  });

  it("should handle agent selection and navigation", async () => {
    render(<MakeAgentPublicForm {...mockProps} />);

    expect(screen.getByText("Select Agents to Make Public")).toBeInTheDocument();

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
      expect(screen.getByText("Confirm Making Agents Public")).toBeInTheDocument();
    });
  });

  it("should submit selected agents successfully", async () => {
    mockMakeAgentsPublicCall.mockResolvedValueOnce({});

    render(<MakeAgentPublicForm {...mockProps} />);

    const selectAllCheckbox = screen.getByLabelText("Select All (2)");
    await act(async () => {
      fireEvent.click(selectAllCheckbox);
    });

    const nextButton = screen.getByRole("button", { name: "Next" });
    await act(async () => {
      fireEvent.click(nextButton);
    });

    await waitFor(() => {
      expect(screen.getByText("Confirm Making Agents Public")).toBeInTheDocument();
    });

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

    checkboxes.forEach((checkbox) => {
      expect(checkbox).toBeChecked();
    });

    // Deselect all
    await act(async () => {
      fireEvent.click(selectAllCheckbox);
    });

    expect(checkboxes[0]).not.toBeChecked();
    expect(checkboxes[1]).not.toBeChecked();
    expect(checkboxes[2]).not.toBeChecked();
  });

  it("should show error when no agents selected", async () => {
    render(<MakeAgentPublicForm {...mockProps} />);

    const checkboxes = screen.getAllByRole("checkbox");
    await act(async () => {
      fireEvent.click(checkboxes[0]); // select all
    });
    await act(async () => {
      fireEvent.click(checkboxes[0]); // deselect all
    });

    const nextButton = screen.getByRole("button", { name: "Next" });
    await act(async () => {
      fireEvent.click(nextButton);
    });

    // Stays on same step
    expect(screen.getByText("Select Agents to Make Public")).toBeInTheDocument();
  });

  it("should display empty state when no agents are available", () => {
    const emptyProps = {
      ...mockProps,
      agentHubData: [] as AgentHubData[],
    };

    render(<MakeAgentPublicForm {...emptyProps} />);

    expect(screen.getByText("No agents available.")).toBeInTheDocument();

    const selectAllCheckbox = screen.getByLabelText("Select All");
    expect(selectAllCheckbox).toBeDisabled();

    const nextButton = screen.getByRole("button", { name: "Next" });
    expect(nextButton).toBeDisabled();
  });

  it("should handle Cancel button functionality", async () => {
    render(<MakeAgentPublicForm {...mockProps} />);

    const cancelButton = screen.getByRole("button", { name: "Cancel" });
    await act(async () => {
      fireEvent.click(cancelButton);
    });

    expect(mockProps.onClose).toHaveBeenCalled();
  });

  it("should handle Previous button functionality", async () => {
    render(<MakeAgentPublicForm {...mockProps} />);

    const nextButton = screen.getByRole("button", { name: "Next" });
    await act(async () => {
      fireEvent.click(nextButton);
    });

    await waitFor(() => {
      expect(screen.getByText("Confirm Making Agents Public")).toBeInTheDocument();
    });

    const previousButton = screen.getByRole("button", { name: "Previous" });
    await act(async () => {
      fireEvent.click(previousButton);
    });

    expect(screen.getByText("Select Agents to Make Public")).toBeInTheDocument();
  });

  it("should handle individual agent selection", async () => {
    render(<MakeAgentPublicForm {...mockProps} />);

    const checkboxes = screen.getAllByRole("checkbox");
    expect(checkboxes).toHaveLength(3);

    const agent1Checkbox = checkboxes[1];
    const agent2Checkbox = checkboxes[2];

    // agent-2 preselected (already public)
    expect(agent2Checkbox).toBeChecked();

    await act(async () => {
      fireEvent.click(agent1Checkbox);
    });

    expect(agent1Checkbox).toBeChecked();
    expect(agent2Checkbox).toBeChecked();

    await act(async () => {
      fireEvent.click(agent2Checkbox);
    });

    expect(agent1Checkbox).toBeChecked();
    expect(agent2Checkbox).not.toBeChecked();

    // Select all should be indeterminate (Radix uses data-state)
    const selectAllCheckbox = checkboxes[0];
    expect(selectAllCheckbox).toHaveAttribute("data-state", "indeterminate");
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

    expect(screen.getByText("Skill 1")).toBeInTheDocument();
    expect(screen.getByText("Skill 2")).toBeInTheDocument();
    expect(screen.getByText("Skill 3")).toBeInTheDocument();
    expect(screen.getByText("+2 more")).toBeInTheDocument();
  });

  it("should handle submit error properly", async () => {
    const errorMessage = "Network error";
    mockMakeAgentsPublicCall.mockRejectedValueOnce(new Error(errorMessage));

    render(<MakeAgentPublicForm {...mockProps} />);

    const nextButton = screen.getByRole("button", { name: "Next" });
    await act(async () => {
      fireEvent.click(nextButton);
    });

    await waitFor(() => {
      expect(screen.getByText("Confirm Making Agents Public")).toBeInTheDocument();
    });

    const submitButton = screen.getByRole("button", { name: "Make Public" });
    await act(async () => {
      fireEvent.click(submitButton);
    });

    await waitFor(() => {
      expect(mockMakeAgentsPublicCall).toHaveBeenCalledWith("test-token", ["agent-2"]);
    });

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

    const nextButton = screen.getByRole("button", { name: "Next" });
    await act(async () => {
      fireEvent.click(nextButton);
    });

    await waitFor(() => {
      expect(screen.getByText("Confirm Making Agents Public")).toBeInTheDocument();
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

    render(<MakeAgentPublicForm {...invisibleProps} />);

    expect(screen.queryByText("Make Agents Public")).not.toBeInTheDocument();
  });

  it("should preselect already public agents when modal opens", () => {
    const mixedPublicProps = {
      ...mockProps,
      agentHubData: [
        {
          agent_id: "agent-1",
          name: "Test Agent 1",
          description: "Description 1",
          url: "http://example.com/agent1",
          version: "1.0",
          is_public: false,
          skills: [],
          protocolVersion: "1.0",
        },
        {
          agent_id: "agent-2",
          name: "Test Agent 2",
          description: "Description 2",
          url: "http://example.com/agent2",
          version: "2.0",
          is_public: true,
          skills: [],
          protocolVersion: "1.0",
        },
        {
          agent_id: "agent-3",
          name: "Test Agent 3",
          description: "Description 3",
          url: "http://example.com/agent3",
          version: "3.0",
          is_public: true,
          skills: [],
          protocolVersion: "1.0",
        },
      ] as AgentHubData[],
    };

    render(<MakeAgentPublicForm {...mixedPublicProps} />);

    const checkboxes = screen.getAllByRole("checkbox");
    expect(checkboxes).toHaveLength(4);

    const agent1Checkbox = checkboxes[1];
    const agent2Checkbox = checkboxes[2];
    const agent3Checkbox = checkboxes[3];

    expect(agent1Checkbox).not.toBeChecked();
    expect(agent2Checkbox).toBeChecked();
    expect(agent3Checkbox).toBeChecked();

    const selectAllCheckbox = checkboxes[0];
    expect(selectAllCheckbox).toHaveAttribute("data-state", "indeterminate");
  });
});
