import React from "react";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent, { PointerEventsCheckLevel } from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import AddAgentForm from "./add_agent_form";
import * as networking from "@/components/networking";
import type { AgentCreateInfo } from "@/components/networking";

vi.mock("@/components/networking", () => ({
  createAgentCall: vi.fn(),
  getAgentCreateMetadata: vi.fn(),
  getAgentsList: vi.fn(),
  keyCreateForAgentCall: vi.fn(),
  keyListCall: vi.fn(),
  keyUpdateCall: vi.fn(),
  modelAvailableCall: vi.fn(),
}));

vi.mock("./agent_card_discovery", () => ({ default: () => <div data-testid="agent-card-discovery" /> }));
vi.mock("@/components/mcp_server_management/MCPServerSelector", () => ({ default: () => <div /> }));
vi.mock("@/components/mcp_server_management/MCPToolPermissions", () => ({ default: () => <div /> }));
vi.mock("@/components/guardrails/GuardrailSelector", () => ({ default: () => <div /> }));
vi.mock("@/components/common_components/team_dropdown", () => ({ default: () => <div /> }));

const a2aInfo: AgentCreateInfo = {
  agent_type: "a2a",
  agent_type_display_name: "A2A Agent",
  description: "Agent-to-agent protocol",
  logo_url: "/ui/assets/logos/a2a_agent.png",
  credential_fields: [],
  use_a2a_form_fields: true,
};

const langgraphInfo: AgentCreateInfo = {
  agent_type: "langgraph",
  agent_type_display_name: "LangGraph",
  description: "LangGraph platform",
  logo_url: "/ui/assets/logos/langgraph.png",
  use_a2a_form_fields: false,
  litellm_params_template: { custom_llm_provider: "langgraph" },
  model_template: "langgraph/{assistant_id}",
  credential_fields: [
    { key: "api_base", label: "API Base", field_type: "text", required: true, placeholder: "https://host" },
    { key: "assistant_id", label: "Assistant ID", field_type: "text", required: true, default_value: "" },
    { key: "api_key", label: "API Key", field_type: "password", required: false },
  ],
};

const renderForm = () =>
  render(<AddAgentForm visible={true} onClose={vi.fn()} accessToken="tok" onSuccess={vi.fn()} />);

const panel = (name: RegExp) => screen.findByRole("button", { name });

const openAgentTypeMenu = async (user: ReturnType<typeof userEvent.setup>) => {
  await user.click(screen.getAllByRole("combobox")[0]);
};

const createdPayload = () => vi.mocked(networking.createAgentCall).mock.calls[0][1] as Record<string, unknown>;

const goToLastStepAndCreate = async (user: ReturnType<typeof userEvent.setup>) => {
  await user.click(await screen.findByRole("button", { name: /^Next/ }));
  await user.click(await screen.findByRole("button", { name: /^Next/ }));
  await user.click(await screen.findByRole("button", { name: /^Next/ }));
  await user.click(await screen.findByRole("button", { name: /Create Agent/ }));
  await waitFor(() => expect(networking.createAgentCall).toHaveBeenCalledTimes(1));
};

const selectAgentType = async (user: ReturnType<typeof userEvent.setup>, label: string) => {
  await openAgentTypeMenu(user);
  await user.click(await screen.findByText(label));
};

describe("AddAgentForm submit payload", () => {
  beforeEach(() => {
    vi.mocked(networking.getAgentCreateMetadata).mockReset().mockResolvedValue([a2aInfo, langgraphInfo]);
    vi.mocked(networking.getAgentsList)
      .mockReset()
      .mockResolvedValue({ agents: [{ agent_id: "sub-1", agent_name: "Sub Agent One" }] });
    vi.mocked(networking.keyListCall).mockReset().mockResolvedValue({ keys: [] });
    vi.mocked(networking.modelAvailableCall)
      .mockReset()
      .mockResolvedValue({ data: [{ id: "gpt-4o" }] });
    vi.mocked(networking.createAgentCall)
      .mockReset()
      .mockResolvedValue({ agent_id: "agent-1", agent_name: "created-agent" } as never);
    vi.mocked(networking.keyCreateForAgentCall)
      .mockReset()
      .mockResolvedValue({ key: "sk-new" } as never);
    vi.mocked(networking.keyUpdateCall)
      .mockReset()
      .mockResolvedValue({} as never);
  });

  it("sends every a2a field the user filled across all collapsible panels", async () => {
    const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
    renderForm();

    await user.type(await screen.findByLabelText("Agent Name"), "support-agent");
    await user.type(screen.getByLabelText("Display Name"), "Support Agent");
    await user.type(screen.getByPlaceholderText("Describe what this agent does..."), "answers questions");
    await user.type(screen.getByLabelText("URL"), "http://localhost:9999/");
    await user.clear(screen.getByLabelText("Version"));
    await user.type(screen.getByLabelText("Version"), "2.0.0");

    await user.click(await panel(/Skills/));
    await user.click(screen.getByRole("button", { name: /Add Skill/ }));
    await user.type(await screen.findByLabelText("Skill ID"), "hello");
    await user.type(screen.getByLabelText("Skill Name"), "Hello");
    await user.type(screen.getByPlaceholderText("What this skill does"), "greets");
    await user.type(screen.getByLabelText("Tags"), "greeting,polite");
    await user.type(screen.getByLabelText("Examples"), "say hi");
    await user.click(screen.getByLabelText("Agent Name"));

    await user.click(await panel(/Capabilities/));
    await user.click(await screen.findByRole("switch", { name: "Streaming" }));
    await user.click(screen.getByRole("switch", { name: "Push Notifications" }));

    await user.click(await panel(/Optional Settings/));
    await user.type(await screen.findByLabelText("Icon URL"), "https://example.com/icon.png");

    await user.click(await panel(/Cost Configuration/));
    await user.type(await screen.findByLabelText("Cost Per Query ($)"), "0.25");
    await user.type(screen.getByLabelText("Input Cost Per Token ($)"), "0.000002");

    await user.click(await panel(/LiteLLM Parameters/));
    await user.type(await screen.findByLabelText("Model (Optional)"), "gpt-4o");
    await user.click(screen.getByRole("switch", { name: "Make Public" }));

    await user.click(await panel(/Authentication Headers/));
    await user.click(await screen.findByRole("button", { name: /Add Static Header/ }));
    await user.type(await screen.findByPlaceholderText("Header name (e.g. Authorization)"), "X-Tenant");
    await user.type(screen.getByPlaceholderText("Value (e.g. Bearer token123)"), "acme");
    await user.type(screen.getByLabelText("Forward Client Headers"), "x-api-key,");
    await user.click(screen.getByLabelText("Agent Name"));

    await goToLastStepAndCreate(user);

    expect(createdPayload()).toEqual({
      agent_name: "support-agent",
      agent_card_params: {
        protocolVersion: "1.0",
        name: "Support Agent",
        description: "answers questions",
        url: "http://localhost:9999/",
        version: "2.0.0",
        defaultInputModes: ["text"],
        defaultOutputModes: ["text"],
        capabilities: { streaming: true, pushNotifications: true },
        skills: [
          {
            id: "hello",
            name: "Hello",
            description: "greets",
            tags: ["greeting", "polite"],
            examples: ["say hi"],
          },
        ],
        iconUrl: "https://example.com/icon.png",
      },
      litellm_params: {
        model: "gpt-4o",
        make_public: true,
        cost_per_query: 0.25,
        input_cost_per_token: 0.000002,
      },
      static_headers: { "X-Tenant": "acme" },
      extra_headers: ["x-api-key"],
    });
  });

  it("keeps a value typed into a panel the user collapsed again", async () => {
    const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
    renderForm();

    await user.type(await screen.findByLabelText("Agent Name"), "collapsed-agent");
    await user.type(screen.getByLabelText("Display Name"), "Collapsed");
    await user.type(screen.getByPlaceholderText("Describe what this agent does..."), "d");

    await user.click(await panel(/Cost Configuration/));
    await user.type(await screen.findByLabelText("Cost Per Query ($)"), "0.75");
    await user.click(await panel(/Cost Configuration/));

    await goToLastStepAndCreate(user);

    expect(createdPayload().litellm_params).toEqual({ cost_per_query: 0.75 });
  });

  it("restores what was typed when a collapsed panel is expanded again", async () => {
    const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
    renderForm();

    await user.click(await panel(/Cost Configuration/));
    await user.type(await screen.findByLabelText("Cost Per Query ($)"), "0.75");
    await user.click(await panel(/Cost Configuration/));
    await user.click(await panel(/Cost Configuration/));

    expect(await screen.findByLabelText("Cost Per Query ($)")).toHaveValue(0.75);
  });

  it("blocks the first step until the required agent name is filled", async () => {
    const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
    renderForm();

    await screen.findByLabelText("Agent Name");
    await user.click(screen.getByRole("button", { name: /^Next/ }));

    expect(await screen.findByText("Please enter a unique agent name")).toBeInTheDocument();
    expect(screen.getByLabelText("Agent Name")).toBeInTheDocument();
  });

  it("sends the custom agent shape when the custom type is picked", async () => {
    const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
    renderForm();

    await screen.findByLabelText("Agent Name");
    await openAgentTypeMenu(user);
    await user.click(await screen.findByText("Custom / Other"));

    await user.type(await screen.findByLabelText("Agent Name"), "my-custom-agent");
    await user.type(screen.getByPlaceholderText("Describe what this agent does\u2026"), "custom thing");

    await goToLastStepAndCreate(user);

    expect(createdPayload()).toEqual({
      agent_name: "my-custom-agent",
      agent_card_params: {
        protocolVersion: "1.0",
        name: "my-custom-agent",
        description: "custom thing",
        url: "",
        version: "1.0.0",
        defaultInputModes: ["text"],
        defaultOutputModes: ["text"],
        capabilities: { streaming: false },
        skills: [],
      },
    });
  });

  it("sends credential fields and the model template for a dynamic agent type", async () => {
    const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
    renderForm();

    await screen.findByLabelText("Agent Name");
    await selectAgentType(user, "LangGraph");

    await user.type(await screen.findByLabelText("Agent Name"), "lg-agent");
    await user.type(screen.getByPlaceholderText("Describe what this agent does..."), "graph agent");
    await user.type(screen.getByLabelText("API Base"), "https://lg.example.com");
    await user.type(screen.getByLabelText("Assistant ID"), "asst_1");
    await user.type(screen.getByLabelText("API Key"), "secret-value");

    await goToLastStepAndCreate(user);

    expect(createdPayload()).toEqual({
      agent_name: "lg-agent",
      agent_card_params: {
        protocolVersion: "1.0",
        name: "lg-agent",
        description: "graph agent",
        url: "https://lg.example.com",
        version: "1.0.0",
        defaultInputModes: ["text"],
        defaultOutputModes: ["text"],
        capabilities: { streaming: true },
        skills: [
          {
            id: "chat",
            name: "Chat",
            description: "General chat capability",
            tags: ["chat", "conversation"],
          },
        ],
      },
      litellm_params: {
        custom_llm_provider: "langgraph",
        api_base: "https://lg.example.com",
        assistant_id: "asst_1",
        api_key: "secret-value",
        model: "langgraph/asst_1",
      },
    });
  });

  it("resets to the agent type that was selected before the switch", async () => {
    const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
    renderForm();

    await screen.findByLabelText("Agent Name");
    expect(screen.getByLabelText("Version")).toHaveValue("1.0.0");

    await openAgentTypeMenu(user);
    await user.click(await screen.findByText("Custom / Other"));
    await screen.findByPlaceholderText("e.g. my-custom-agent");

    await selectAgentType(user, "A2A Agent");

    expect(await screen.findByLabelText("Version")).toHaveValue("");
  });

  it("sends entitlements and rate limits gathered on the later steps", async () => {
    const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
    renderForm();

    await user.type(await screen.findByLabelText("Agent Name"), "entitled-agent");
    await user.type(screen.getByLabelText("Display Name"), "Entitled");
    await user.type(screen.getByPlaceholderText("Describe what this agent does..."), "d");
    await user.click(screen.getByRole("button", { name: /^Next/ }));

    await user.type(await screen.findByLabelText("Allowed Models"), "gpt-4o,");
    await user.keyboard("{Escape}");
    await user.click(screen.getByLabelText("Allowed Agents (Sub-Agents)"));
    await user.click(await screen.findByTitle("Sub Agent One"));
    await user.keyboard("{Escape}");
    await user.click(screen.getByText(/Configure which models, agents, and MCP tools/));
    await user.click(screen.getByRole("button", { name: /^Next/ }));

    await user.click((await screen.findAllByRole("switch"))[1]);
    await user.type(screen.getByLabelText("TPM Limit"), "1000");
    await user.type(screen.getByLabelText("Session RPM Limit"), "20");
    await user.click(screen.getByRole("button", { name: /^Next/ }));

    await user.click(await screen.findByRole("button", { name: /Create Agent/ }));
    await waitFor(() => expect(networking.createAgentCall).toHaveBeenCalledTimes(1));

    const payload = createdPayload();
    expect(payload.tpm_limit).toBe(1000);
    expect(payload.session_rpm_limit).toBe(20);
    expect(payload.object_permission).toEqual({ models: ["gpt-4o"], agents: ["sub-1"] });
    expect(payload.litellm_params).toEqual({ require_trace_id_on_calls_by_agent: true });
  });

  it("creates a key named after the agent once the agent is created", async () => {
    const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
    renderForm();

    await user.type(await screen.findByLabelText("Agent Name"), "keyed-agent");
    await user.type(screen.getByLabelText("Display Name"), "Keyed");
    await user.type(screen.getByPlaceholderText("Describe what this agent does..."), "d");

    await goToLastStepAndCreate(user);

    expect(networking.keyCreateForAgentCall).toHaveBeenCalledWith(
      "tok",
      "agent-1",
      "keyed-agent-key",
      [],
      undefined,
      null,
    );
    expect(await screen.findByText("Agent Created!")).toBeInTheDocument();
    expect(within(screen.getByText("Agent Created!").parentElement!).getByText("created-agent")).toBeInTheDocument();
  });
});
