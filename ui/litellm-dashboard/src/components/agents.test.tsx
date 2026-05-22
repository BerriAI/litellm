import React from "react";
import { render, screen, waitFor, act, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import AgentsPanel from "./agents";
import * as networking from "./networking";

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

// Note: agents.tsx no longer uses AgentCardGrid — it renders a Table directly.

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

  it("should show Actions column header for admin role", async () => {
    render(<AgentsPanel accessToken="test-token" userRole="Admin" />);
    await waitFor(() => {
      expect(screen.getByRole("columnheader", { name: /actions/i })).toBeInTheDocument();
    });
  });

  it("should not show Actions column header for internal user role", async () => {
    render(<AgentsPanel accessToken="test-token" userRole="Internal User" />);
    await waitFor(() => {
      expect(screen.queryByRole("columnheader", { name: /actions/i })).not.toBeInTheDocument();
      // confirm table is rendered (not still loading)
      expect(screen.getByRole("table")).toBeInTheDocument();
    });
  });

  it("should render the Health Check toggle", async () => {
    render(<AgentsPanel accessToken="test-token" userRole="Admin" />);
    expect(screen.getByText("Health Check")).toBeInTheDocument();
  });

  it("should render the Health Check toggle for non-admin users too", async () => {
    render(<AgentsPanel accessToken="test-token" userRole="Internal User" />);
    expect(screen.getByText("Health Check")).toBeInTheDocument();
  });

  it("should call getAgentsList with health_check=false on initial load", async () => {
    render(<AgentsPanel accessToken="test-token" userRole="Admin" />);
    await waitFor(() => {
      expect(networking.getAgentsList).toHaveBeenCalledWith("test-token", false);
    });
  });

  it("should fetch agent keys with the key list API max page size", async () => {
    vi.mocked(networking.getAgentsList).mockResolvedValueOnce({
      agents: [
        {
          agent_id: "agent-1",
          agent_name: "Test Agent",
          litellm_params: { model: "gpt-4o" },
        },
      ],
    });

    render(<AgentsPanel accessToken="test-token" userRole="Admin" />);

    await waitFor(() => {
      expect(networking.keyListCall).toHaveBeenCalledWith(
        "test-token",
        null,
        null,
        null,
        null,
        null,
        1,
        100,
      );
    });
  });

  it("should keep fetching key list pages until visible agent keys are found", async () => {
    vi.mocked(networking.getAgentsList).mockResolvedValueOnce({
      agents: [
        {
          agent_id: "agent-1",
          agent_name: "Test Agent",
          litellm_params: { model: "gpt-4o" },
        },
      ],
    });
    vi.mocked(networking.keyListCall)
      .mockResolvedValueOnce({ keys: [], total_pages: 2 })
      .mockResolvedValueOnce({
        keys: [{ agent_id: "agent-1", key_alias: "agent-key", token: "1234567890" }],
        total_pages: 2,
      });

    render(<AgentsPanel accessToken="test-token" userRole="Admin" />);

    await waitFor(() => {
      expect(networking.keyListCall).toHaveBeenCalledTimes(2);
      expect(networking.keyListCall).toHaveBeenLastCalledWith(
        "test-token",
        null,
        null,
        null,
        null,
        null,
        2,
        100,
      );
      expect(screen.getByText("Active")).toBeInTheDocument();
    });
  });

  it("should call getAgentsList with health_check=true when toggle is enabled", async () => {
    render(<AgentsPanel accessToken="test-token" userRole="Admin" />);
    await waitFor(() => {
      expect(networking.getAgentsList).toHaveBeenCalledWith("test-token", false);
    });

    const toggle = screen.getByRole("switch");
    await act(async () => {
      fireEvent.click(toggle);
    });

    await waitFor(() => {
      expect(networking.getAgentsList).toHaveBeenCalledWith("test-token", true);
    });
  });
});
