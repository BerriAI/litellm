import React from "react";
import { describe, it, expect, vi } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "../../../tests/test-utils";
import AgentCard from "./agent_card";
import type { Agent } from "./types";

const baseAgent: Agent = {
  agent_id: "agent-123",
  agent_name: "Test Agent",
  litellm_params: { model: "gpt-4" },
  agent_card_params: {
    description: "A test agent for unit testing",
    url: "https://agent.example.com",
  },
};

const defaultProps = {
  agent: baseAgent,
  onAgentClick: vi.fn(),
  accessToken: "token-123",
  isAdmin: false,
  onAgentUpdated: vi.fn(),
};

describe("AgentCard", () => {
  it("should render the agent name and description", () => {
    renderWithProviders(<AgentCard {...defaultProps} />);

    expect(screen.getByText("Test Agent")).toBeInTheDocument();
    expect(screen.getByText("A test agent for unit testing")).toBeInTheDocument();
  });

  it("should show 'No description' when agent has no description", () => {
    const agent = { ...baseAgent, agent_card_params: {} };
    renderWithProviders(<AgentCard {...defaultProps} agent={agent} />);

    expect(screen.getByText("No description")).toBeInTheDocument();
  });

  it("should show the agent URL when provided", () => {
    renderWithProviders(<AgentCard {...defaultProps} />);

    expect(screen.getByText("https://agent.example.com")).toBeInTheDocument();
  });

  it("should show 'Needs Setup' badge when agent has no key", () => {
    renderWithProviders(<AgentCard {...defaultProps} />);

    expect(screen.getByText("Needs Setup")).toBeInTheDocument();
    expect(screen.getByText("No key assigned")).toBeInTheDocument();
  });

  it("should show 'Active' badge and key info when agent has a key", () => {
    const keyInfo = { has_key: true, key_alias: "my-key" };
    renderWithProviders(<AgentCard {...defaultProps} keyInfo={keyInfo} />);

    expect(screen.getByText("Active")).toBeInTheDocument();
    expect(screen.getByText("my-key")).toBeInTheDocument();
  });

  it("should call onAgentClick when card is clicked", async () => {
    const user = userEvent.setup();
    const onAgentClick = vi.fn();
    renderWithProviders(<AgentCard {...defaultProps} onAgentClick={onAgentClick} />);

    await user.click(screen.getByText("Test Agent"));

    expect(onAgentClick).toHaveBeenCalledWith("agent-123");
  });

  it("should show delete button only for admins", () => {
    const onDeleteClick = vi.fn();
    const { unmount } = renderWithProviders(
      <AgentCard {...defaultProps} isAdmin={false} onDeleteClick={onDeleteClick} />
    );
    expect(screen.queryByRole("button", { name: /delete/i })).not.toBeInTheDocument();

    unmount();

    renderWithProviders(
      <AgentCard {...defaultProps} isAdmin={true} onDeleteClick={onDeleteClick} />
    );
    expect(screen.getByRole("button", { name: /delete/i })).toBeInTheDocument();
  });

  it("should call onDeleteClick with agent id and name when delete is clicked", async () => {
    const user = userEvent.setup();
    const onDeleteClick = vi.fn();
    renderWithProviders(
      <AgentCard {...defaultProps} isAdmin={true} onDeleteClick={onDeleteClick} />
    );

    await user.click(screen.getByRole("button", { name: /delete/i }));

    expect(onDeleteClick).toHaveBeenCalledWith("agent-123", "Test Agent");
  });
});
