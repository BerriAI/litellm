import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi } from "vitest";

import AgentsTable from "./AgentsTable";
import { Agent } from "@/components/agents/types";

const baseProps = {
  isLoading: false,
  isAdmin: true,
  healthCheckEnabled: false,
  isHealthCheckLoading: false,
  onHealthCheckToggle: vi.fn(),
  onAgentClick: vi.fn(),
  onDeleteClick: vi.fn(),
};

const makeAgent = (overrides: Partial<Agent> = {}): Agent => ({
  agent_id: "agent-1",
  agent_name: "Test Agent",
  litellm_params: { model: "gpt-4" },
  spend: 0,
  keys: [{ token: "hash-1", key_alias: "primary", key_name: "sk-...1" }],
  created_at: "2023-01-01T00:00:00Z",
  ...overrides,
});

describe("AgentsTable", () => {
  it("renders every column header", () => {
    render(<AgentsTable agents={[]} {...baseProps} />);
    for (const header of ["Agent Name", "Agent ID", "Spend (USD)", "Model", "Created", "Status"]) {
      expect(screen.getByText(header)).toBeInTheDocument();
    }
  });

  it("renders the agent's model and opens the detail view when the ID cell is clicked", async () => {
    const user = userEvent.setup();
    const onAgentClick = vi.fn();
    const agent = makeAgent({ agent_id: "agent-xyz", agent_name: "Router", litellm_params: { model: "claude-3-5" } });
    render(<AgentsTable agents={[agent]} {...baseProps} onAgentClick={onAgentClick} />);

    expect(screen.getByText("claude-3-5")).toBeInTheDocument();

    await user.click(screen.getByText("agent-xyz"));
    expect(onAgentClick).toHaveBeenCalledWith("agent-xyz");
  });

  it("marks agents Active when they have keys and Needs Setup when they have none", () => {
    render(
      <AgentsTable
        agents={[
          makeAgent({ agent_id: "keyed", agent_name: "Keyed Agent", keys: [{ token: "k" }] }),
          makeAgent({ agent_id: "keyless", agent_name: "Keyless Agent", keys: [] }),
        ]}
        {...baseProps}
      />,
    );

    const keyedRow = screen.getByText("Keyed Agent").closest("tr")!;
    const keylessRow = screen.getByText("Keyless Agent").closest("tr")!;
    expect(within(keyedRow).getByText("Active")).toBeInTheDocument();
    expect(within(keylessRow).getByText("Needs Setup")).toBeInTheDocument();
  });

  it("deletes an agent through the ⋯ actions menu", async () => {
    const user = userEvent.setup();
    const onDeleteClick = vi.fn();
    const agent = makeAgent({ agent_id: "agent-9", agent_name: "Doomed Agent" });
    render(<AgentsTable agents={[agent]} {...baseProps} onDeleteClick={onDeleteClick} />);

    await user.click(screen.getByTestId("agent-actions-agent-9"));
    await user.click(await screen.findByTestId("agent-action-delete"));

    expect(onDeleteClick).toHaveBeenCalledWith("agent-9", "Doomed Agent");
  });

  it("hides the actions column entirely for non-admins", () => {
    const agent = makeAgent({ agent_id: "agent-2" });
    render(<AgentsTable agents={[agent]} {...baseProps} isAdmin={false} />);

    expect(screen.queryByTestId("agent-actions-agent-2")).not.toBeInTheDocument();
    expect(screen.queryByRole("columnheader", { name: /actions/i })).not.toBeInTheDocument();
    expect(screen.getByRole("table")).toBeInTheDocument();
  });

  it("shows the actions column for admins", () => {
    render(<AgentsTable agents={[makeAgent({ agent_id: "agent-3" })]} {...baseProps} isAdmin />);
    expect(screen.getByRole("columnheader", { name: /actions/i })).toBeInTheDocument();
    expect(screen.getByTestId("agent-actions-agent-3")).toBeInTheDocument();
  });

  it("defaults to sorting by created_at descending (newest first)", () => {
    render(
      <AgentsTable
        agents={[
          makeAgent({ agent_id: "old", agent_name: "Alpha Agent", created_at: "2021-06-01T00:00:00Z" }),
          makeAgent({ agent_id: "new", agent_name: "Beta Agent", created_at: "2023-06-01T00:00:00Z" }),
        ]}
        {...baseProps}
      />,
    );

    const bodyRows = screen.getAllByRole("row").slice(1);
    expect(bodyRows[0].textContent).toContain("Beta Agent");
    expect(bodyRows[1].textContent).toContain("Alpha Agent");
  });

  it("sorts agents with no created_at last, never ahead of dated ones", () => {
    render(
      <AgentsTable
        agents={[
          makeAgent({ agent_id: "old", agent_name: "Alpha Agent", created_at: "2021-06-01T00:00:00Z" }),
          makeAgent({ agent_id: "undated", agent_name: "Undated Agent", created_at: undefined }),
          makeAgent({ agent_id: "new", agent_name: "Beta Agent", created_at: "2023-06-01T00:00:00Z" }),
        ]}
        {...baseProps}
      />,
    );

    const bodyRows = screen.getAllByRole("row").slice(1);
    expect(bodyRows[0].textContent).toContain("Beta Agent");
    expect(bodyRows[1].textContent).toContain("Alpha Agent");
    expect(bodyRows[2].textContent).toContain("Undated Agent");
  });

  it("shows a rich empty state when there are no agents", () => {
    render(<AgentsTable agents={[]} {...baseProps} />);
    expect(screen.getByText("No agents yet")).toBeInTheDocument();
    expect(screen.queryByTestId("skeleton-row")).not.toBeInTheDocument();
  });

  it("renders loading skeleton rows on initial load instead of the empty state", () => {
    render(<AgentsTable agents={[]} {...baseProps} isLoading />);
    expect(screen.getAllByTestId("skeleton-row").length).toBeGreaterThan(0);
    expect(screen.queryByText("No agents yet")).not.toBeInTheDocument();
  });

  it("invokes the health-check toggle from the toolbar", async () => {
    const user = userEvent.setup();
    const onHealthCheckToggle = vi.fn();
    render(<AgentsTable agents={[]} {...baseProps} onHealthCheckToggle={onHealthCheckToggle} />);

    expect(screen.getByText("Health Check")).toBeInTheDocument();
    await user.click(screen.getByRole("switch"));
    expect(onHealthCheckToggle).toHaveBeenCalledWith(true, expect.anything());
  });
});
