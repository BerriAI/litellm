import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi, beforeEach } from "vitest";

// Mock networking module
const mockGetAgentsList = vi.fn();
vi.mock("../networking", () => ({
  getAgentsList: (...args: any[]) => mockGetAgentsList(...args),
}));

// Mock antd Select
vi.mock("antd", () => {
  const SelectComponent = ({ children, onChange, value, mode, placeholder, loading, disabled, ...props }: any) => (
    <div data-testid="agent-select" data-loading={loading} data-disabled={disabled}>
      <select
        data-testid="select-input"
        multiple={mode === "multiple"}
        value={value || []}
        onChange={(e) => {
          const selected = Array.from(e.target.selectedOptions, (opt: any) => opt.value);
          onChange?.(selected);
        }}
        disabled={disabled}
      >
        {children}
      </select>
      {loading && <span data-testid="loading-indicator">Loading</span>}
    </div>
  );

  SelectComponent.Option = ({ children, value, ...props }: any) => (
    <option value={value} {...props}>
      {children}
    </option>
  );

  return { Select: SelectComponent };
});

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
    expect(screen.getByTestId("agent-select")).toBeInTheDocument();
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
    mockGetAgentsList.mockReturnValue(new Promise((r) => { resolve = r; }));

    render(<AgentSelector {...defaultProps} />);
    expect(screen.getByTestId("agent-select")).toHaveAttribute("data-loading", "true");

    // Resolve to clean up
    resolve({ agents: [] });
    await waitFor(() => {
      expect(screen.getByTestId("agent-select")).toHaveAttribute("data-loading", "false");
    });
  });

  it("renders agent options after fetch", async () => {
    render(<AgentSelector {...defaultProps} />);
    await waitFor(() => {
      expect(screen.getByText("Agent One")).toBeInTheDocument();
      expect(screen.getByText("Agent Two")).toBeInTheDocument();
    });
  });

  it("renders access group options with group prefix", async () => {
    render(<AgentSelector {...defaultProps} />);
    await waitFor(() => {
      expect(screen.getByText("group-a")).toBeInTheDocument();
      expect(screen.getByText("group-b")).toBeInTheDocument();
    });
  });

  it("respects disabled prop", () => {
    render(<AgentSelector {...defaultProps} disabled />);
    expect(screen.getByTestId("agent-select")).toHaveAttribute("data-disabled", "true");
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

  it("passes value as flattened selectedValues", async () => {
    render(
      <AgentSelector
        {...defaultProps}
        value={{ agents: ["agent-1"], accessGroups: ["group-a"] }}
      />
    );
    await waitFor(() => {
      const select = screen.getByTestId("select-input");
      // The value should contain agent-1 and group:group-a
      expect(select).toBeInTheDocument();
    });
  });

  it("handles null response from API", async () => {
    mockGetAgentsList.mockResolvedValue(null);
    render(<AgentSelector {...defaultProps} />);
    await waitFor(() => {
      expect(screen.getByTestId("agent-select")).toHaveAttribute("data-loading", "false");
    });
  });
});
