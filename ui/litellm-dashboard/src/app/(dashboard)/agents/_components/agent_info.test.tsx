import React from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { NuqsTestingAdapter, type OnUrlUpdateFunction } from "nuqs/adapters/testing";
import type { ReactNode } from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderWithProviders } from "@/../tests/test-utils";
import AgentInfoView from "./agent_info";
import * as networking from "@/components/networking";
import type { Agent } from "@/components/agents/types";
import type { KeyResponse } from "@/components/key_team_helpers/key_list";

vi.mock("@/components/networking", () => ({
  getAgentInfo: vi.fn(),
  getAgentCreateMetadata: vi.fn(),
  patchAgentCall: vi.fn(),
}));

const agentKeysState = vi.hoisted(() => ({
  keys: [] as KeyResponse[],
  isLoading: false,
  refetch: vi.fn(),
}));

vi.mock("@/app/(dashboard)/hooks/keys/useKeys", () => ({
  useKeys: () => ({
    data: { keys: agentKeysState.keys },
    isLoading: agentKeysState.isLoading,
    refetch: agentKeysState.refetch,
  }),
}));

vi.mock("@/components/templates/key_info_view", () => ({
  default: ({
    keyData,
    onClose,
    onDelete,
    onKeyDataUpdate,
    backButtonText,
  }: {
    keyData: KeyResponse | undefined;
    onClose: () => void;
    onDelete: () => void;
    onKeyDataUpdate?: (updated: Partial<KeyResponse>) => void;
    backButtonText: string;
  }) => (
    <div>
      <p data-testid="key-info-view">{keyData ? keyData.key_alias : "Key not found"}</p>
      <button onClick={onClose}>{backButtonText}</button>
      <button onClick={onDelete}>Delete key</button>
      <button onClick={() => onKeyDataUpdate?.({ token: "hash-rotated", key_alias: "agent-primary-key" })}>
        Regenerate key
      </button>
      <button onClick={() => onKeyDataUpdate?.({ blocked: true })}>Block key</button>
      <button onClick={() => onKeyDataUpdate?.({ token: "hash-abcdef123456789", key_alias: "renamed" })}>
        Rename key
      </button>
    </div>
  ),
}));

vi.mock("./agent_card_discovery", () => ({
  default: () => <div data-testid="agent-card-discovery" />,
}));

vi.mock("./agent_form_fields", () => ({
  default: () => <div data-testid="agent-form-fields" />,
  unmountedA2AFieldNames: () => [],
}));

vi.mock("@/app/(dashboard)/hooks/mcpServers/useMCPServers", () => ({
  useMCPServers: () => ({ data: [{ server_id: "srv-1", server_name: "github" }] }),
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
    agentKeysState.keys = [];
    agentKeysState.isLoading = false;
    vi.mocked(networking.getAgentInfo).mockReset().mockResolvedValue(agent);
    vi.mocked(networking.getAgentCreateMetadata).mockReset().mockResolvedValue([]);
    vi.mocked(networking.patchAgentCall).mockReset().mockResolvedValue({});
  });

  it("submits the edited agent when Save Changes is pressed", async () => {
    renderWithProviders(<AgentInfoView agentId="agent-1" onClose={vi.fn()} accessToken="sk-test" isAdmin={true} />);

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
  });

  it("shows MCP grants with server names on the overview tab", async () => {
    vi.mocked(networking.getAgentInfo).mockResolvedValue({
      ...agent,
      object_permission: { mcp_servers: ["srv-1"] },
    } as unknown as Agent);

    renderWithProviders(<AgentInfoView agentId="agent-1" onClose={vi.fn()} accessToken="sk-test" isAdmin={true} />);

    expect(await screen.findByText("github (srv-1)")).toBeInTheDocument();
  });
});

describe("AgentInfoView URL state", () => {
  const lastUrlUpdate = (onUrlUpdate: ReturnType<typeof vi.fn<OnUrlUpdateFunction>>) =>
    onUrlUpdate.mock.calls.at(-1)?.[0];
  const agentKey = { token: "hash-abcdef123456789", key_alias: "agent-primary-key" } as KeyResponse;
  const renderAgent = (isAdmin: boolean, searchParams: string, onUrlUpdate?: OnUrlUpdateFunction) =>
    renderWithProviders(<AgentInfoView agentId="agent-1" onClose={vi.fn()} accessToken="sk-test" isAdmin={isAdmin} />, {
      searchParams,
      onUrlUpdate,
    });

  beforeEach(() => {
    vi.mocked(networking.getAgentInfo).mockReset().mockResolvedValue(agent);
    vi.mocked(networking.getAgentCreateMetadata).mockReset().mockResolvedValue([]);
    agentKeysState.keys = [agentKey];
    agentKeysState.isLoading = false;
    agentKeysState.refetch.mockReset();
  });

  it("opens the Settings tab named in the URL for an admin", async () => {
    renderAgent(true, "?agent=agent-1&tab=settings");

    expect(await screen.findByRole("tab", { name: "Settings" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("tab", { name: "Overview" })).toHaveAttribute("aria-selected", "false");
  });

  it("writes tab when Settings is picked and drops it on Overview", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderAgent(true, "?agent=agent-1", onUrlUpdate);

    await user.click(await screen.findByRole("tab", { name: "Settings" }));
    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("tab")).toBe("settings"));
    expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("agent")).toBe("agent-1");

    await user.click(screen.getByRole("tab", { name: "Overview" }));
    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.has("tab")).toBe(false));
    expect(screen.getByRole("tab", { name: "Overview" })).toHaveAttribute("aria-selected", "true");
  });

  it("keeps a non-admin on Overview and clears tab=settings from the URL", async () => {
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    render(<AgentInfoView agentId="agent-1" onClose={vi.fn()} accessToken="sk-test" isAdmin={false} />, {
      wrapper: ({ children }: { children: ReactNode }) => (
        <NuqsTestingAdapter
          searchParams="?agent=agent-1&tab=settings"
          onUrlUpdate={onUrlUpdate}
          hasMemory
          resetUrlUpdateQueueOnMount={false}
        >
          {children}
        </NuqsTestingAdapter>
      ),
    });

    expect(await screen.findByRole("tab", { name: "Overview" })).toHaveAttribute("aria-selected", "true");
    expect(screen.queryByRole("tab", { name: "Settings" })).not.toBeInTheDocument();
    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.has("tab")).toBe(false));
    expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("agent")).toBe("agent-1");
  });

  it("shows the key named in the URL", async () => {
    renderAgent(true, `?agent=agent-1&key=${agentKey.token}`);

    expect(await screen.findByTestId("key-info-view")).toHaveTextContent("agent-primary-key");
    expect(screen.queryByRole("tab", { name: "Overview" })).not.toBeInTheDocument();
  });

  it("waits for the agent's keys before resolving the key in the URL", async () => {
    agentKeysState.keys = [];
    agentKeysState.isLoading = true;
    const { rerender } = renderAgent(true, `?agent=agent-1&key=${agentKey.token}`);
    await waitFor(() => expect(networking.getAgentInfo).toHaveBeenCalled());
    await waitFor(() => expect(screen.queryByRole("tab", { name: "Overview" })).not.toBeInTheDocument());
    expect(screen.queryByTestId("key-info-view")).not.toBeInTheDocument();

    agentKeysState.keys = [agentKey];
    agentKeysState.isLoading = false;
    rerender(<AgentInfoView agentId="agent-1" onClose={vi.fn()} accessToken="sk-test" isAdmin={true} />);

    expect(await screen.findByTestId("key-info-view")).toHaveTextContent("agent-primary-key");
  });

  it("hands KeyInfoView no key data when the URL names a key the agent does not have", async () => {
    renderAgent(true, "?agent=agent-1&key=hash-unknown");

    expect(await screen.findByTestId("key-info-view")).toHaveTextContent("Key not found");
  });

  it("pushes key when a key is opened and clears it on back", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderAgent(true, "?agent=agent-1", onUrlUpdate);

    await user.click(await screen.findByRole("button", { name: /^hash-abcdef1/ }));

    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("key")).toBe(agentKey.token));
    expect(lastUrlUpdate(onUrlUpdate)?.options.history).toBe("push");
    expect(screen.getByTestId("key-info-view")).toHaveTextContent("agent-primary-key");

    await user.click(screen.getByRole("button", { name: "Back to Agent" }));

    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.has("key")).toBe(false));
    expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("agent")).toBe("agent-1");
    expect(await screen.findByRole("tab", { name: "Overview" })).toBeInTheDocument();
  });

  it("swaps key to the regenerated token in place and refetches the agent's keys", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderAgent(true, `?agent=agent-1&key=${agentKey.token}`, onUrlUpdate);

    await user.click(await screen.findByRole("button", { name: "Regenerate key" }));

    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("key")).toBe("hash-rotated"));
    expect(lastUrlUpdate(onUrlUpdate)?.options.history).toBe("replace");
    expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("agent")).toBe("agent-1");
    expect(agentKeysState.refetch).toHaveBeenCalledTimes(1);
  });

  it.each(["Block key", "Rename key"])(
    "keeps key and skips the refetch when %s leaves the token unchanged",
    async (buttonName) => {
      const user = userEvent.setup();
      renderAgent(true, `?agent=agent-1&key=${agentKey.token}`);

      await user.click(await screen.findByRole("button", { name: buttonName }));

      expect(agentKeysState.refetch).not.toHaveBeenCalled();
      expect(screen.getByTestId("key-info-view")).toHaveTextContent("agent-primary-key");
    },
  );

  it("clears key and refetches the agent's keys after the key is deleted", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderAgent(true, `?agent=agent-1&key=${agentKey.token}`, onUrlUpdate);

    await user.click(await screen.findByRole("button", { name: "Delete key" }));

    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.has("key")).toBe(false));
    expect(agentKeysState.refetch).toHaveBeenCalledTimes(1);
  });
});
