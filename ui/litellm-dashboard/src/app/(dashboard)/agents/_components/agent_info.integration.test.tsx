import React from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent, { PointerEventsCheckLevel } from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import AgentInfoView from "./agent_info";
import * as networking from "@/components/networking";
import type { AgentCreateInfo } from "@/components/networking";

vi.mock("@/components/networking", () => ({
  getAgentInfo: vi.fn(),
  patchAgentCall: vi.fn(),
  getAgentCreateMetadata: vi.fn(),
}));

vi.mock("@/app/(dashboard)/hooks/keys/useKeys", () => ({
  useKeys: () => ({ data: { keys: [] }, isLoading: false, refetch: vi.fn() }),
}));

vi.mock("./agent_card_discovery", () => ({ default: () => <div data-testid="agent-card-discovery" /> }));

const A2A_AGENT = {
  agent_id: "agent-1",
  agent_name: "my-agent",
  agent_card_params: {
    protocolVersion: "1.0",
    name: "My Agent",
    description: "does things",
    url: "http://localhost:9999/",
    version: "2.3.4",
    defaultInputModes: ["text"],
    defaultOutputModes: ["text"],
    capabilities: { streaming: true, pushNotifications: true },
    skills: [{ id: "chat", name: "Chat", description: "chat skill", tags: ["a"], examples: ["ex1"] }],
    iconUrl: "https://example.com/icon.png",
    documentationUrl: "https://example.com/docs",
  },
  litellm_params: { model: "gpt-4o", make_public: true, cost_per_query: 0.5 },
  tpm_limit: 111,
  rpm_limit: 222,
  session_tpm_limit: 333,
  session_rpm_limit: 444,
  static_headers: { Authorization: "Bearer x" },
  extra_headers: ["x-api-key"],
};

const LANGGRAPH_AGENT = {
  agent_id: "agent-2",
  agent_name: "lg-agent",
  agent_card_params: { name: "LG", description: "graph agent", url: "", version: "1.0.0", skills: [] },
  litellm_params: {
    custom_llm_provider: "langgraph",
    model: "langgraph/asst_1",
    api_base: "https://lg.example.com",
    cost_per_query: 0.25,
  },
};

const langgraphInfo: AgentCreateInfo = {
  agent_type: "langgraph",
  agent_type_display_name: "LangGraph",
  description: "LangGraph platform",
  logo_url: "/l.png",
  use_a2a_form_fields: false,
  litellm_params_template: { custom_llm_provider: "langgraph" },
  model_template: "langgraph/{assistant_id}",
  credential_fields: [
    { key: "api_base", label: "API Base", field_type: "text", required: true },
    {
      key: "assistant_id",
      label: "Assistant ID",
      field_type: "text",
      required: false,
      include_in_litellm_params: false,
    },
  ],
};

const setup = () => userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });

const renderView = () => render(<AgentInfoView agentId="agent-1" onClose={vi.fn()} accessToken="tok" isAdmin={true} />);

const openEditor = async (user: ReturnType<typeof setup>) => {
  await user.click(await screen.findByRole("tab", { name: "Settings" }));
  await user.click(await screen.findByRole("button", { name: "Edit Settings" }));
  await screen.findByLabelText("Agent Name");
};

const patchedPayload = () => vi.mocked(networking.patchAgentCall).mock.calls[0][2] as Record<string, unknown>;

const save = async (user: ReturnType<typeof setup>) => {
  await user.click(screen.getByRole("button", { name: "Save Changes" }));
  await waitFor(() => expect(networking.patchAgentCall).toHaveBeenCalledTimes(1));
};

describe("AgentInfoView update payload", () => {
  beforeEach(() => {
    vi.mocked(networking.getAgentCreateMetadata).mockReset().mockResolvedValue([langgraphInfo]);
    vi.mocked(networking.getAgentInfo)
      .mockReset()
      .mockResolvedValue(A2A_AGENT as never);
    vi.mocked(networking.patchAgentCall)
      .mockReset()
      .mockResolvedValue({} as never);
  });

  it("sends only the fields whose panel has been opened, dropping the rest", async () => {
    const user = setup();
    renderView();
    await openEditor(user);

    await save(user);

    expect(patchedPayload()).toEqual({
      agent_name: "my-agent",
      agent_card_params: {
        protocolVersion: "1.0",
        name: "My Agent",
        description: "does things",
        url: "http://localhost:9999/",
        version: "2.3.4",
        defaultInputModes: ["text"],
        defaultOutputModes: ["text"],
        capabilities: { streaming: false },
        skills: [],
      },
      tpm_limit: 111,
      rpm_limit: 222,
      session_tpm_limit: 333,
      session_rpm_limit: 444,
    });
  });

  it("sends the loaded values of every panel the user opens", async () => {
    const user = setup();
    renderView();
    await openEditor(user);

    await user.click(screen.getByRole("button", { name: /Skills/ }));
    await user.click(screen.getByRole("button", { name: /Capabilities/ }));
    await user.click(screen.getByRole("button", { name: /Optional Settings/ }));
    await user.click(screen.getByRole("button", { name: /Cost Configuration/ }));
    await user.click(screen.getByRole("button", { name: /LiteLLM Parameters/ }));
    await user.click(screen.getByRole("button", { name: /Authentication Headers/ }));
    await screen.findByLabelText("Forward Client Headers");

    await save(user);

    expect(patchedPayload()).toEqual({
      agent_name: "my-agent",
      agent_card_params: {
        protocolVersion: "1.0",
        name: "My Agent",
        description: "does things",
        url: "http://localhost:9999/",
        version: "2.3.4",
        defaultInputModes: ["text"],
        defaultOutputModes: ["text"],
        capabilities: { streaming: true, pushNotifications: true, stateTransitionHistory: undefined },
        skills: [{ id: "chat", name: "Chat", description: "chat skill", tags: ["a"], examples: ["ex1"] }],
        iconUrl: "https://example.com/icon.png",
        documentationUrl: "https://example.com/docs",
      },
      litellm_params: { model: "gpt-4o", make_public: true, cost_per_query: 0.5 },
      static_headers: { Authorization: "Bearer x" },
      extra_headers: ["x-api-key"],
      tpm_limit: 111,
      rpm_limit: 222,
      session_tpm_limit: 333,
      session_rpm_limit: 444,
    });
  });

  it("clamps a rate limit typed below its minimum up to that minimum", async () => {
    const user = setup();
    renderView();
    await openEditor(user);

    await user.clear(screen.getByLabelText("TPM Limit"));
    fireEvent.change(screen.getByLabelText("TPM Limit"), { target: { value: "-5" } });
    await save(user);

    expect(patchedPayload().tpm_limit).toBe(0);
  });

  it("drops a rate limit the user cleared", async () => {
    const user = setup();
    renderView();
    await openEditor(user);

    await user.clear(screen.getByLabelText("RPM Limit"));
    await save(user);

    expect(patchedPayload()).not.toHaveProperty("rpm_limit");
  });

  it("submits when Enter is pressed in a text field", async () => {
    const user = setup();
    renderView();
    await openEditor(user);

    await user.type(screen.getByLabelText("Agent Name"), "{Enter}");

    await waitFor(() => expect(networking.patchAgentCall).toHaveBeenCalledTimes(1));
  });

  it("blocks the save while a required field is empty", async () => {
    const user = setup();
    renderView();
    await openEditor(user);

    await user.clear(screen.getByLabelText("Agent Name"));
    await user.click(screen.getByRole("button", { name: "Save Changes" }));

    expect(await screen.findByText("Please enter a unique agent name")).toBeInTheDocument();
    expect(networking.patchAgentCall).not.toHaveBeenCalled();
  });

  it("sends credential fields and the rebuilt model for a dynamic agent type", async () => {
    vi.mocked(networking.getAgentInfo).mockResolvedValue(LANGGRAPH_AGENT as never);
    const user = setup();
    renderView();
    await openEditor(user);

    await user.clear(screen.getByLabelText("API Base"));
    fireEvent.change(screen.getByLabelText("API Base"), { target: { value: "https://other.example.com" } });

    await save(user);

    expect(patchedPayload()).toEqual({
      agent_name: "lg-agent",
      agent_card_params: {
        protocolVersion: "1.0",
        name: "lg-agent",
        description: "graph agent",
        url: "https://other.example.com",
        version: "1.0.0",
        defaultInputModes: ["text"],
        defaultOutputModes: ["text"],
        capabilities: { streaming: true },
        skills: [{ id: "chat", name: "Chat", description: "General chat capability", tags: ["chat", "conversation"] }],
      },
      litellm_params: {
        custom_llm_provider: "langgraph",
        api_base: "https://other.example.com",
        model: "langgraph/asst_1",
      },
    });
  });

  it("reloads the agent and leaves edit mode when the edit is cancelled", async () => {
    const user = setup();
    renderView();
    await openEditor(user);
    expect(networking.getAgentInfo).toHaveBeenCalledTimes(1);

    await user.click(screen.getByRole("button", { name: "Cancel" }));

    expect(await screen.findByRole("button", { name: "Edit Settings" })).toBeInTheDocument();
    expect(networking.getAgentInfo).toHaveBeenCalledTimes(2);
    expect(networking.patchAgentCall).not.toHaveBeenCalled();
  });
});
