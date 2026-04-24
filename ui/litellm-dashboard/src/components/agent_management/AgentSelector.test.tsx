import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi, beforeEach } from "vitest";

const mockGetAgentsList = vi.fn();
vi.mock("../networking", () => ({
  getAgentsList: (...args: any[]) => mockGetAgentsList(...args),
}));

import AgentSelector from "./AgentSelector";

describe("AgentSelector", () => {
  const defaultProps = {
    onChange: vi.fn(),
    accessToken: "test-token",
  };

  beforeEach(() => {
    vi.clearAllMocks();
    mockGetAgentsList.mockResolvedValue({
      agents: [
        { agent_id: "agent-1", agent_name: "Agent One" },
        {
          agent_id: "agent-2",
          agent_name: "Agent Two",
          agent_access_groups: ["group-a", "group-b"],
        },
      ],
    });
  });

  it("should render the selector trigger", () => {
    render(<AgentSelector {...defaultProps} />);
    expect(screen.getByRole("button", { name: /select agents/i })).toBeInTheDocument();
  });

  it("should fetch agents on mount with access token", async () => {
    render(<AgentSelector {...defaultProps} />);
    await waitFor(() => {
      expect(mockGetAgentsList).toHaveBeenCalledWith("test-token");
    });
  });

  it("should not fetch when accessToken is empty", () => {
    render(<AgentSelector {...defaultProps} accessToken="" />);
    expect(mockGetAgentsList).not.toHaveBeenCalled();
  });

  it("should render agent options after fetch when popover is opened", async () => {
    const user = userEvent.setup();
    render(<AgentSelector {...defaultProps} />);
    await waitFor(() => {
      expect(mockGetAgentsList).toHaveBeenCalled();
    });

    await user.click(screen.getByRole("button", { name: /select agents/i }));

    await waitFor(() => {
      expect(screen.getByText("Agent One")).toBeInTheDocument();
      expect(screen.getByText("Agent Two")).toBeInTheDocument();
    });
  });

  it("should render access group options with group prefix when popover is opened", async () => {
    const user = userEvent.setup();
    render(<AgentSelector {...defaultProps} />);
    await waitFor(() => {
      expect(mockGetAgentsList).toHaveBeenCalled();
    });

    await user.click(screen.getByRole("button", { name: /select agents/i }));

    await waitFor(() => {
      expect(screen.getByText("group-a")).toBeInTheDocument();
      expect(screen.getByText("group-b")).toBeInTheDocument();
    });
  });

  it("should respect disabled prop", () => {
    render(<AgentSelector {...defaultProps} disabled />);
    expect(screen.getByRole("button", { name: /select agents/i })).toBeDisabled();
  });

  it("should handle API error gracefully", async () => {
    const consoleSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    mockGetAgentsList.mockRejectedValue(new Error("API error"));

    render(<AgentSelector {...defaultProps} />);
    await waitFor(() => {
      expect(consoleSpy).toHaveBeenCalledWith(
        "Error fetching agents:",
        expect.any(Error),
      );
    });

    consoleSpy.mockRestore();
  });

  it("should render selected agents and access groups as badges", async () => {
    render(
      <AgentSelector
        {...defaultProps}
        value={{ agents: ["agent-1"], accessGroups: ["group-a"] }}
      />,
    );

    await waitFor(() => {
      expect(mockGetAgentsList).toHaveBeenCalled();
    });

    // Labels are resolved from the options list once the fetch completes
    await waitFor(() => {
      expect(screen.getByText("Agent One")).toBeInTheDocument();
      expect(screen.getByText("group-a")).toBeInTheDocument();
    });
  });

  it("should handle null response from API without error", async () => {
    mockGetAgentsList.mockResolvedValue(null);
    render(<AgentSelector {...defaultProps} />);
    await waitFor(() => {
      expect(mockGetAgentsList).toHaveBeenCalled();
    });
    // Trigger still renders with placeholder
    expect(screen.getByRole("button", { name: /select agents/i })).toBeInTheDocument();
  });

  it("should call onChange when an option is selected", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<AgentSelector {...defaultProps} onChange={onChange} />);
    await waitFor(() => {
      expect(mockGetAgentsList).toHaveBeenCalled();
    });

    await user.click(screen.getByRole("button", { name: /select agents/i }));
    await user.click(await screen.findByText("Agent One"));

    expect(onChange).toHaveBeenCalledWith({
      agents: ["agent-1"],
      accessGroups: [],
    });
  });
});
