import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen, waitFor } from "@testing-library/react";
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
  const nextButtons = screen.getAllByRole("button", { name: /next/i });
  const nextButton = nextButtons[nextButtons.length - 1];

  // Step 0 -> 1: need agent_name to be filled
  // The form validation may block, so we fill agent name first
  const agentNameInput = screen.getByLabelText(/agent name/i);
  const { fireEvent } = await import("@testing-library/react");
  const { act } = await import("react");
  await act(async () => {
    fireEvent.change(agentNameInput, { target: { value: "test-agent" } });
  });

  // Click Next to go to step 1 (Entitlements)
  await act(async () => {
    fireEvent.click(nextButton);
  });

  // Click Next to go to step 2 (Governance)
  await waitFor(() => {
    expect(screen.getByText("Entitlements")).toBeInTheDocument();
  });

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

    expect(
      screen.queryByText(/require x-litellm-trace-id on calls to this agent/i)
    ).not.toBeInTheDocument();

    expect(
      screen.queryByText(/require x-litellm-trace-id on calls by this agent/i)
    ).not.toBeInTheDocument();
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

    expect(
      screen.getByText(/require x-litellm-trace-id on calls to this agent/i)
    ).toBeInTheDocument();

    expect(
      screen.getByText(/require x-litellm-trace-id on calls by this agent/i)
    ).toBeInTheDocument();
  });
});
