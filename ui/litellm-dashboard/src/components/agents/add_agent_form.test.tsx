import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen, waitFor, act, fireEvent } from "@testing-library/react";
import { renderWithProviders } from "../../../tests/test-utils";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: vi.fn(),
}));

vi.mock("../networking", () => ({
  getAgentCreateMetadata: vi.fn().mockResolvedValue([]),
  getAgentsList: vi.fn().mockResolvedValue({ agents: [] }),
  keyListCall: vi.fn().mockResolvedValue({ keys: [] }),
  modelAvailableCall: vi.fn().mockResolvedValue({ data: [] }),
  createAgentCall: vi.fn().mockResolvedValue({ agent_id: "a1", agent_name: "test" }),
  keyCreateForAgentCall: vi.fn().mockResolvedValue({ key: "sk-123" }),
  keyUpdateCall: vi.fn().mockResolvedValue({}),
}));

vi.mock("../mcp_server_management/MCPServerSelector", () => ({
  default: () => <div data-testid="mcp-server-selector" />,
}));

vi.mock("../mcp_server_management/MCPToolPermissions", () => ({
  default: () => <div data-testid="mcp-tool-permissions" />,
}));

vi.mock("../guardrails/GuardrailSelector", () => ({
  default: () => <div data-testid="guardrail-selector" />,
}));

vi.mock("../shared/CreatedKeyDisplay", () => ({
  default: () => <div data-testid="created-key-display" />,
}));

import AddAgentForm from "./add_agent_form";

const baseAuthorized = {
  token: "123",
  accessToken: "123",
  userId: "user-1",
  userEmail: "user@example.com",
  userRole: "Admin",
  premiumUser: false,
  disabledPersonalKeyCreation: null,
  showSSOBanner: false,
};

const defaultProps = {
  visible: true,
  onClose: vi.fn(),
  accessToken: "test-token",
  onSuccess: vi.fn(),
  teams: [],
};

const navigateToGovernanceStep = async () => {
  const agentNameInput = screen.getByLabelText(/agent name/i);
  await act(async () => {
    fireEvent.change(agentNameInput, { target: { value: "test-agent" } });
  });

  // Step 0 -> Step 1 (Entitlements)
  const nextButtons = screen.getAllByRole("button", { name: /next/i });
  await act(async () => {
    fireEvent.click(nextButtons[nextButtons.length - 1]);
  });

  await waitFor(() => {
    expect(screen.getByText("Allowed Models")).toBeInTheDocument();
  });

  // Step 1 -> Step 2 (Governance)
  const nextButtons2 = screen.getAllByRole("button", { name: /next/i });
  await act(async () => {
    fireEvent.click(nextButtons2[nextButtons2.length - 1]);
  });

  await waitFor(() => {
    expect(screen.getByText("Tracing")).toBeInTheDocument();
  });
};

describe("AddAgentForm tracing enforcement premium gate", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should show enterprise upgrade notice when user is not premium", async () => {
    vi.mocked(useAuthorized).mockReturnValue(baseAuthorized);

    renderWithProviders(<AddAgentForm {...defaultProps} />);
    await navigateToGovernanceStep();

    expect(
      screen.getByText(/enforcing trace-id requirements on agents is a litellm enterprise feature/i)
    ).toBeInTheDocument();

    // The switch labels should NOT be present (tracing section replaced by notice)
    const allSwitches = screen.queryAllByRole("switch");
    const tracingSectionSwitches = allSwitches.filter((el) => {
      const parent = el.closest(".space-y-4");
      return parent?.textContent?.includes("Require x-litellm-trace-id");
    });
    expect(tracingSectionSwitches).toHaveLength(0);
  });

  it("should show tracing switches when user is premium", async () => {
    vi.mocked(useAuthorized).mockReturnValue({
      ...baseAuthorized,
      premiumUser: true,
    });

    renderWithProviders(<AddAgentForm {...defaultProps} />);
    await navigateToGovernanceStep();

    expect(
      screen.queryByText(/enforcing trace-id requirements on agents is a litellm enterprise feature/i)
    ).not.toBeInTheDocument();

    // Both tracing switch labels should be visible
    expect(
      screen.getByText("Require x-litellm-trace-id on calls TO this agent")
    ).toBeInTheDocument();

    // The switches for tracing should be present
    const allSwitches = screen.getAllByRole("switch");
    expect(allSwitches.length).toBeGreaterThanOrEqual(2);
  });
});
