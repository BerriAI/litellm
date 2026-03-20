import { renderWithProviders, screen } from "../../../tests/test-utils";
import { vi } from "vitest";
import AgentCardGrid from "./agent_card_grid";
import type { Agent, AgentKeyInfo } from "./types";

vi.mock("./agent_card", () => ({
  default: ({ agent, onAgentClick }: any) => (
    <div data-testid={`agent-card-${agent.agent_id}`} onClick={() => onAgentClick(agent.agent_id)}>
      {agent.agent_name}
    </div>
  ),
}));

const mockAgents: Agent[] = [
  {
    agent_id: "agent-1",
    agent_name: "Test Agent 1",
    litellm_params: { model: "gpt-4" },
    agent_card_params: { description: "First agent" },
  },
  {
    agent_id: "agent-2",
    agent_name: "Test Agent 2",
    litellm_params: { model: "claude-3" },
    agent_card_params: { description: "Second agent" },
  },
];

const mockKeyInfoMap: Record<string, AgentKeyInfo> = {
  "agent-1": { has_key: true, key_alias: "key-1" },
  "agent-2": { has_key: false },
};

const defaultProps = {
  agentsList: mockAgents,
  keyInfoMap: mockKeyInfoMap,
  isLoading: false,
  onDeleteClick: vi.fn(),
  accessToken: "test-token",
  onAgentUpdated: vi.fn(),
  isAdmin: true,
  onAgentClick: vi.fn(),
};

describe("AgentCardGrid", () => {
  it("should render", () => {
    renderWithProviders(<AgentCardGrid {...defaultProps} />);
    expect(screen.getByText("Test Agent 1")).toBeInTheDocument();
  });

  it("should render all agent cards", () => {
    renderWithProviders(<AgentCardGrid {...defaultProps} />);
    expect(screen.getByText("Test Agent 1")).toBeInTheDocument();
    expect(screen.getByText("Test Agent 2")).toBeInTheDocument();
  });

  it("should show loading skeletons when isLoading is true", () => {
    renderWithProviders(<AgentCardGrid {...defaultProps} isLoading={true} />);
    expect(screen.queryByText("Test Agent 1")).not.toBeInTheDocument();
  });

  it("should show admin empty state message when no agents and isAdmin", () => {
    renderWithProviders(
      <AgentCardGrid {...defaultProps} agentsList={[]} isAdmin={true} />
    );
    expect(
      screen.getByText("No agents found. Create one to get started.")
    ).toBeInTheDocument();
  });

  it("should show non-admin empty state message when no agents and not admin", () => {
    renderWithProviders(
      <AgentCardGrid {...defaultProps} agentsList={[]} isAdmin={false} />
    );
    expect(
      screen.getByText("No agents found. Contact an admin to create agents.")
    ).toBeInTheDocument();
  });

  it("should call onAgentClick when a card is clicked", async () => {
    const onAgentClick = vi.fn();
    renderWithProviders(
      <AgentCardGrid {...defaultProps} onAgentClick={onAgentClick} />
    );
    const { default: userEvent } = await import("@testing-library/user-event");
    const user = userEvent.setup();
    await user.click(screen.getByTestId("agent-card-agent-1"));
    expect(onAgentClick).toHaveBeenCalledWith("agent-1");
  });
});
