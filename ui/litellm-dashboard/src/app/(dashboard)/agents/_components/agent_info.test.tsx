import React from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import AgentInfoView from "./agent_info";
import * as networking from "@/components/networking";
import type { Agent } from "@/components/agents/types";

vi.mock("@/components/networking", () => ({
  getAgentInfo: vi.fn(),
  getAgentCreateMetadata: vi.fn(),
  patchAgentCall: vi.fn(),
}));

vi.mock("@/app/(dashboard)/hooks/keys/useKeys", () => ({
  useKeys: () => ({ data: { keys: [] }, isLoading: false, refetch: vi.fn() }),
}));

vi.mock("./agent_card_discovery", () => ({
  default: () => <div data-testid="agent-card-discovery" />,
}));

vi.mock("./agent_form_fields", () => ({
  default: () => <div data-testid="agent-form-fields" />,
  unmountedA2AFieldNames: () => [],
}));

const agent = {
  agent_id: "agent-1",
  agent_name: "support-agent",
  agent_card_params: {
    name: "Support Agent",
    description: "Answers support questions",
    url: "http://localhost:9999/",
    version: "1.0.0",
    protocolVersion: "1.0",
    capabilities: { streaming: false },
    skills: [],
  },
  tpm_limit: 100,
} as unknown as Agent;

describe("AgentInfoView settings", () => {
  beforeEach(() => {
    vi.mocked(networking.getAgentInfo).mockReset().mockResolvedValue(agent);
    vi.mocked(networking.getAgentCreateMetadata).mockReset().mockResolvedValue([]);
    vi.mocked(networking.patchAgentCall).mockReset().mockResolvedValue({});
  });

  it("submits the edited agent when Save Changes is pressed", async () => {
    render(<AgentInfoView agentId="agent-1" onClose={vi.fn()} accessToken="sk-test" isAdmin={true} />);

    fireEvent.click(await screen.findByRole("tab", { name: "Settings" }));
    fireEvent.click(screen.getByRole("button", { name: "Edit Settings" }));

    const tpmLimit = await screen.findByLabelText("TPM Limit");
    fireEvent.change(tpmLimit, { target: { value: "42" } });

    fireEvent.click(screen.getByRole("button", { name: /Save Changes/ }));

    await waitFor(() => expect(networking.patchAgentCall).toHaveBeenCalledTimes(1));
    const [token, agentId, payload] = vi.mocked(networking.patchAgentCall).mock.calls[0];
    expect(token).toBe("sk-test");
    expect(agentId).toBe("agent-1");
    expect(payload.tpm_limit).toBe(42);
  });
});
