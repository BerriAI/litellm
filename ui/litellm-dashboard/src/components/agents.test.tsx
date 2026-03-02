import React from "react";
import { render, screen, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import AgentsPanel from "./agents";

vi.mock("./networking", () => ({
  getAgentsList: vi.fn().mockResolvedValue({ agents: [] }),
  deleteAgentCall: vi.fn(),
  keyListCall: vi.fn().mockResolvedValue({ keys: [] }),
}));

vi.mock("./agents/add_agent_form", () => ({
  default: () => <div data-testid="add-agent-form" />,
}));

vi.mock("./agents/agent_card_grid", () => ({
  default: ({ isAdmin }: { isAdmin: boolean }) => (
    <div data-testid="agent-card-grid" data-is-admin={String(isAdmin)} />
  ),
}));

vi.mock("./agents/agent_info", () => ({
  default: () => <div data-testid="agent-info" />,
}));

describe("AgentsPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should render the Agents panel title", async () => {
    render(<AgentsPanel accessToken="test-token" userRole="Admin" />);
    expect(screen.getByText("Agents")).toBeInTheDocument();
  });

  it("should show Add New Agent button for admin users", async () => {
    render(<AgentsPanel accessToken="test-token" userRole="Admin" />);
    expect(screen.getByText("+ Add New Agent")).toBeInTheDocument();
  });

  it("should show Add New Agent button for proxy_admin users", async () => {
    render(<AgentsPanel accessToken="test-token" userRole="proxy_admin" />);
    expect(screen.getByText("+ Add New Agent")).toBeInTheDocument();
  });

  it("should not show Add New Agent button for internal_user role", async () => {
    render(<AgentsPanel accessToken="test-token" userRole="Internal User" />);
    expect(screen.queryByText("+ Add New Agent")).not.toBeInTheDocument();
  });

  it("should not show Add New Agent button for internal_user_viewer role", async () => {
    render(<AgentsPanel accessToken="test-token" userRole="Internal Viewer" />);
    expect(screen.queryByText("+ Add New Agent")).not.toBeInTheDocument();
  });

  it("should pass isAdmin=true to AgentCardGrid for admin role", async () => {
    render(<AgentsPanel accessToken="test-token" userRole="Admin" />);
    await waitFor(() => {
      const grid = screen.getByTestId("agent-card-grid");
      expect(grid).toHaveAttribute("data-is-admin", "true");
    });
  });

  it("should pass isAdmin=false to AgentCardGrid for internal user role", async () => {
    render(<AgentsPanel accessToken="test-token" userRole="Internal User" />);
    await waitFor(() => {
      const grid = screen.getByTestId("agent-card-grid");
      expect(grid).toHaveAttribute("data-is-admin", "false");
    });
  });
});
