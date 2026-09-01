import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import AgentPermissions from "./AgentPermissions";
import * as networking from "../networking";

vi.mock("../networking");

describe("AgentPermissions", () => {
  const accessToken = "test-token";
  const agentId = "90337622-756e-4f25-98f0-01fc8174aa24";

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("lists agents inherited from access groups with an Inherited tag and counts them", async () => {
    vi.mocked(networking.getAgentsList).mockResolvedValue({
      agents: [{ agent_id: agentId, agent_name: "support_agent" }],
    });

    render(<AgentPermissions agents={[]} inheritedAgents={[agentId]} accessToken={accessToken} />);

    expect(await screen.findByText(/support_agent/)).toBeInTheDocument();
    expect(screen.getByText("Inherited")).toBeInTheDocument();
    expect(screen.getByText("1")).toBeInTheDocument();
    expect(screen.queryByText("No agents or access groups configured")).not.toBeInTheDocument();
    expect(networking.getAgentsList).toHaveBeenCalledWith(accessToken);
  });

  it("does not double-list an agent that is both granted directly and inherited", async () => {
    vi.mocked(networking.getAgentsList).mockResolvedValue({
      agents: [{ agent_id: agentId, agent_name: "support_agent" }],
    });

    render(<AgentPermissions agents={[agentId]} inheritedAgents={[agentId]} accessToken={accessToken} />);

    expect(await screen.findByText(/support_agent/)).toBeInTheDocument();
    expect(screen.getAllByText(/support_agent/)).toHaveLength(1);
    expect(screen.queryByText("Inherited")).not.toBeInTheDocument();
    expect(screen.getByText("1")).toBeInTheDocument();
  });

  it("shows the empty state when nothing is granted directly or inherited", () => {
    render(<AgentPermissions agents={[]} inheritedAgents={[]} accessToken={accessToken} />);

    expect(screen.getByText("No agents or access groups configured")).toBeInTheDocument();
    expect(screen.getByText("0")).toBeInTheDocument();
    expect(networking.getAgentsList).not.toHaveBeenCalled();
  });
});
