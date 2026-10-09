import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import AgentPermissions from "./AgentPermissions";
import * as networking from "../networking";

vi.mock("../networking");

describe("AgentPermissions", () => {
  const accessToken = "test-token";
  const agentId = "90337622-756e-4f25-98f0-01fc8174aa24";

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("lists agents inherited from access groups, counts them, and names the groups on hover", async () => {
    const user = userEvent.setup();
    vi.mocked(networking.getAgentsList).mockResolvedValue({
      agents: [{ agent_id: agentId, agent_name: "support_agent" }],
    });

    render(
      <AgentPermissions
        agents={[]}
        inheritedAgents={[{ id: agentId, accessGroupNames: ["platform-tools", "support"] }]}
        accessToken={accessToken}
      />,
    );

    const row = await screen.findByText(/support_agent/);
    expect(screen.getByText("1")).toBeInTheDocument();
    expect(screen.queryByText("No agents or access groups configured")).not.toBeInTheDocument();
    expect(networking.getAgentsList).toHaveBeenCalledWith(accessToken);

    await user.hover(row);
    expect(
      await screen.findByText(`Granted via access groups platform-tools, support. Full ID: ${agentId}`),
    ).toBeInTheDocument();
  });

  it("does not double-list an agent that is both granted directly and inherited", async () => {
    const user = userEvent.setup();
    vi.mocked(networking.getAgentsList).mockResolvedValue({
      agents: [{ agent_id: agentId, agent_name: "support_agent" }],
    });

    render(
      <AgentPermissions
        agents={[agentId]}
        inheritedAgents={[{ id: agentId, accessGroupNames: ["support"] }]}
        accessToken={accessToken}
      />,
    );

    const row = await screen.findByText(/support_agent/);
    expect(screen.getAllByText(/support_agent/)).toHaveLength(1);
    expect(screen.getByText("1")).toBeInTheDocument();

    await user.hover(row);
    expect(await screen.findByText(`Full ID: ${agentId}`)).toBeInTheDocument();
  });

  it("shows the empty state when nothing is granted directly or inherited", () => {
    render(<AgentPermissions agents={[]} inheritedAgents={[]} accessToken={accessToken} />);

    expect(screen.getByText("No agents or access groups configured")).toBeInTheDocument();
    expect(screen.getByText("0")).toBeInTheDocument();
    expect(networking.getAgentsList).not.toHaveBeenCalled();
  });
});
