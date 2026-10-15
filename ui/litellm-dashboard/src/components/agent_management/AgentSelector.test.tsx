import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi, beforeEach } from "vitest";

// Mock networking module
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
        { agent_id: "agent-2", agent_name: "Agent Two", agent_access_groups: ["group-a", "group-b"] },
      ],
    });
  });

  it("renders the selector", () => {
    render(<AgentSelector {...defaultProps} />);
    expect(screen.getByRole("combobox")).toBeInTheDocument();
  });

  it("fetches agents on mount with access token", async () => {
    render(<AgentSelector {...defaultProps} />);
    await waitFor(() => {
      expect(mockGetAgentsList).toHaveBeenCalledWith("test-token");
    });
  });

  it("does not fetch when accessToken is empty", () => {
    render(<AgentSelector {...defaultProps} accessToken="" />);
    expect(mockGetAgentsList).not.toHaveBeenCalled();
  });

  it("shows loading state while fetching", async () => {
    // Keep the promise pending
    let resolve: any;
    mockGetAgentsList.mockReturnValue(
      new Promise((r) => {
        resolve = r;
      }),
    );

    render(<AgentSelector {...defaultProps} />);
    expect(screen.getByRole("combobox")).toHaveAttribute("placeholder", "Loading...");

    // Resolve to clean up
    resolve({ agents: [] });
    await waitFor(() => {
      expect(screen.getByRole("combobox")).toHaveAttribute("placeholder", "Select agents");
    });
  });

  it("renders agent options after fetch", async () => {
    const user = userEvent.setup();
    render(<AgentSelector {...defaultProps} />);

    await user.click(screen.getByRole("combobox"));

    expect(await screen.findByRole("option", { name: /Agent One/ })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: /Agent Two/ })).toBeInTheDocument();
  });

  it("renders access group options with group prefix", async () => {
    const user = userEvent.setup();
    render(<AgentSelector {...defaultProps} />);

    await user.click(screen.getByRole("combobox"));

    expect(await screen.findByRole("option", { name: /group-a/ })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: /group-b/ })).toBeInTheDocument();
  });

  it("respects disabled prop", () => {
    render(<AgentSelector {...defaultProps} disabled />);
    expect(screen.getByRole("combobox")).toBeDisabled();
  });

  it("handles API error gracefully", async () => {
    const consoleSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    mockGetAgentsList.mockRejectedValue(new Error("API error"));

    render(<AgentSelector {...defaultProps} />);
    await waitFor(() => {
      expect(consoleSpy).toHaveBeenCalledWith("Error fetching agents:", expect.any(Error));
    });

    consoleSpy.mockRestore();
  });

  // An agent and an access group are stored under separate keys but share one control, so the
  // chips are what prove both halves reached it.
  it("shows both a selected agent and a selected access group", async () => {
    render(<AgentSelector {...defaultProps} value={{ agents: ["agent-1"], accessGroups: ["group-a"] }} />);

    expect(await screen.findByLabelText("Agent One")).toBeInTheDocument();
    expect(screen.getByLabelText("group-a")).toBeInTheDocument();
  });

  it("handles null response from API", async () => {
    mockGetAgentsList.mockResolvedValue(null);
    render(<AgentSelector {...defaultProps} />);
    await waitFor(() => {
      expect(screen.getByRole("combobox")).toHaveAttribute("placeholder", "Select agents");
    });
  });
});
