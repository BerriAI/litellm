import React from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import AgentInfoView from "./agent_info";
import AgentFormFields from "./agent_form_fields";
import * as networking from "@/components/networking";
import type { Agent } from "@/components/agents/types";

vi.mock("@/components/networking", () => ({
  getAgentInfo: vi.fn(),
  getAgentCreateMetadata: vi.fn(),
  patchAgentCall: vi.fn(),
  triggerAgentKillSwitchCall: vi.fn(),
}));

vi.mock("@/app/(dashboard)/hooks/keys/useKeys", () => ({
  useKeys: () => ({ data: { keys: [] }, isLoading: false, refetch: vi.fn() }),
}));

vi.mock("./AgentIdentityDetails", () => ({
  AgentIdentityDetails: () => null,
}));

vi.mock("./agent_card_discovery", () => ({
  default: () => <div data-testid="agent-card-discovery" />,
}));

vi.mock("./agent_form_fields", () => ({
  default: vi.fn(() => <div data-testid="agent-form-fields" />),
  unmountedA2AFieldNames: () => [],
}));

vi.mock("@/app/(dashboard)/hooks/mcpServers/useMCPServers", () => ({
  useMCPServers: () => ({ data: [{ server_id: "srv-1", server_name: "github" }] }),
}));

vi.mock("@/app/(dashboard)/hooks/accessGroups/useAccessGroups", () => ({
  useAccessGroups: () => ({
    data: [{ access_group_id: "ag-1", access_group_name: "support-tools" }],
    isLoading: false,
    isError: false,
  }),
}));

vi.mock("@/components/common_components/AccessGroupSelector", () => ({
  default: ({ value, onChange }: { value?: string[]; onChange: (value: string[]) => void }) => (
    <div>
      <span data-testid="selected-access-groups">{(value ?? []).join(",")}</span>
      <button type="button" onClick={() => onChange(["ag-1"])}>
        Attach ag-1
      </button>
      <button type="button" onClick={() => onChange([])}>
        Detach all access groups
      </button>
    </div>
  ),
}));

vi.mock("@/components/mcp_server_management/MCPServerSelector", () => ({
  default: () => <div data-testid="mcp-server-selector" />,
}));

vi.mock("@/components/mcp_server_management/MCPToolPermissions", () => ({
  default: () => <div data-testid="mcp-tool-permissions" />,
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
    vi.restoreAllMocks();
    vi.mocked(AgentFormFields)
      .mockReset()
      .mockImplementation(() => <div data-testid="agent-form-fields" />);
    vi.mocked(networking.getAgentInfo).mockReset().mockResolvedValue(agent);
    vi.mocked(networking.getAgentCreateMetadata).mockReset().mockResolvedValue([]);
    vi.mocked(networking.patchAgentCall).mockReset().mockResolvedValue({});
    vi.mocked(networking.triggerAgentKillSwitchCall).mockReset();
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
    const clearedMcpGrants = { mcp_servers: [], mcp_access_groups: [], mcp_toolsets: [], mcp_tool_permissions: {} };
    expect(payload.object_permission).toEqual(clearedMcpGrants);
    expect(payload.access_group_ids).toEqual([]);
  });

  it("saves unrelated settings when the existing card has no description", async () => {
    const actual = await vi.importActual<typeof import("./agent_form_fields")>("./agent_form_fields");
    vi.mocked(AgentFormFields).mockImplementation(actual.default);
    const { description: _description, ...card } = agent.agent_card_params;
    vi.mocked(networking.getAgentInfo).mockResolvedValue({ ...agent, agent_card_params: card });
    render(<AgentInfoView agentId="agent-1" onClose={vi.fn()} accessToken="sk-test" isAdmin={true} />);
    fireEvent.click(await screen.findByRole("tab", { name: "Settings" }));
    fireEvent.click(screen.getByRole("button", { name: "Edit Settings" }));
    expect(await screen.findByLabelText("Description")).toHaveValue("");
    fireEvent.change(screen.getByLabelText("TPM Limit"), { target: { value: "42" } });
    fireEvent.click(screen.getByRole("button", { name: /Save Changes/ }));
    await waitFor(() => expect(networking.patchAgentCall).toHaveBeenCalledOnce());
    const [, , payload] = vi.mocked(networking.patchAgentCall).mock.calls[0];
    expect(payload.tpm_limit).toBe(42);
    expect(payload.agent_card_params.description).toBe("");
  });

  it("sends the newly attached access group in the update payload", async () => {
    render(<AgentInfoView agentId="agent-1" onClose={vi.fn()} accessToken="sk-test" isAdmin={true} />);

    fireEvent.click(await screen.findByRole("tab", { name: "Settings" }));
    fireEvent.click(screen.getByRole("button", { name: "Edit Settings" }));
    fireEvent.click(await screen.findByRole("button", { name: "Attach ag-1" }));
    expect(screen.getByTestId("selected-access-groups")).toHaveTextContent("ag-1");

    fireEvent.click(screen.getByRole("button", { name: /Save Changes/ }));

    await waitFor(() => expect(networking.patchAgentCall).toHaveBeenCalledTimes(1));
    const [, , payload] = vi.mocked(networking.patchAgentCall).mock.calls[0];
    expect(payload.access_group_ids).toEqual(["ag-1"]);
  });

  it("loads the attached access groups into the editor and sends an empty list once detached", async () => {
    vi.mocked(networking.getAgentInfo).mockResolvedValue({ ...agent, access_group_ids: ["ag-1"] });
    render(<AgentInfoView agentId="agent-1" onClose={vi.fn()} accessToken="sk-test" isAdmin={true} />);

    fireEvent.click(await screen.findByRole("tab", { name: "Settings" }));
    fireEvent.click(screen.getByRole("button", { name: "Edit Settings" }));
    expect(await screen.findByTestId("selected-access-groups")).toHaveTextContent("ag-1");

    fireEvent.click(screen.getByRole("button", { name: "Detach all access groups" }));
    fireEvent.click(screen.getByRole("button", { name: /Save Changes/ }));

    await waitFor(() => expect(networking.patchAgentCall).toHaveBeenCalledTimes(1));
    const [, , payload] = vi.mocked(networking.patchAgentCall).mock.calls[0];
    expect(payload.access_group_ids).toEqual([]);
  });

  it("shows MCP grants with server names on the overview tab", async () => {
    vi.mocked(networking.getAgentInfo).mockResolvedValue({
      ...agent,
      object_permission: { mcp_servers: ["srv-1"] },
    } as unknown as Agent);

    render(<AgentInfoView agentId="agent-1" onClose={vi.fn()} accessToken="sk-test" isAdmin={true} />);

    expect(await screen.findByText("github (srv-1)")).toBeInTheDocument();
  });

  it("shows attached access groups with their names on the overview tab", async () => {
    vi.mocked(networking.getAgentInfo).mockResolvedValue({ ...agent, access_group_ids: ["ag-1", "ag-unknown"] });

    render(<AgentInfoView agentId="agent-1" onClose={vi.fn()} accessToken="sk-test" isAdmin={true} />);

    expect(await screen.findByText("support-tools (ag-1)")).toBeInTheDocument();
    expect(screen.getByText("ag-unknown")).toBeInTheDocument();
  });

  it("shows None when the agent has no access groups attached", async () => {
    render(<AgentInfoView agentId="agent-1" onClose={vi.fn()} accessToken="sk-test" isAdmin={true} />);

    expect(await screen.findByText("Access Groups")).toBeInTheDocument();
    expect(screen.getByText("None")).toBeInTheDocument();
  });

  it("renders the kill switch Danger Zone for admins with the configured webhook", async () => {
    vi.mocked(networking.getAgentInfo).mockResolvedValue({
      ...agent,
      kill_switch: { url: "https://ops.example.com/kill", method: "DELETE" },
    });
    render(<AgentInfoView agentId="agent-1" onClose={vi.fn()} accessToken="sk-test" isAdmin={true} />);

    const dangerZone = await screen.findByRole("region", { name: "Danger Zone" });
    expect(dangerZone).toHaveTextContent("DELETE https://ops.example.com/kill");
    expect(screen.getByRole("button", { name: "Fire Kill Switch" })).toBeInTheDocument();
    expect(screen.queryByText("Kill Switch")).not.toBeInTheDocument();
  });

  it("hides the Danger Zone from non-admins", async () => {
    vi.mocked(networking.getAgentInfo).mockResolvedValue({
      ...agent,
      kill_switch: { url: "https://ops.example.com/kill", method: "POST" },
    });
    render(<AgentInfoView agentId="agent-1" onClose={vi.fn()} accessToken="sk-test" isAdmin={false} />);

    expect(await screen.findByRole("heading", { name: "support-agent" })).toBeInTheDocument();
    expect(screen.queryByRole("region", { name: "Danger Zone" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Fire Kill Switch" })).not.toBeInTheDocument();
  });
});
