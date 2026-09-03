import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import AgentBuilderView from "./AgentBuilderView";
import type { AgentModel } from "../../llm_calls/fetch_agents";

const modelCreateCall = vi.fn().mockResolvedValue({ model_id: "id-new" });
const modelPatchUpdateCall = vi.fn().mockResolvedValue({});
const modelDeleteCall = vi.fn().mockResolvedValue({});
const keyCreateCall = vi.fn().mockResolvedValue({ key: "sk-agent-key" });
const fetchMCPServers = vi.fn().mockResolvedValue([]);
const fetchAvailableAgentModels = vi.fn();
const fetchAvailableModels = vi.fn().mockResolvedValue([{ model_group: "gpt-4o" }, { model_group: "claude-sonnet-4" }]);

vi.mock("@/components/networking", () => ({
  proxyBaseUrl: "https://proxy.example.com",
  modelCreateCall: (...args: unknown[]) => modelCreateCall(...args),
  modelPatchUpdateCall: (...args: unknown[]) => modelPatchUpdateCall(...args),
  modelDeleteCall: (...args: unknown[]) => modelDeleteCall(...args),
  keyCreateCall: (...args: unknown[]) => keyCreateCall(...args),
  fetchMCPServers: (...args: unknown[]) => fetchMCPServers(...args),
}));

vi.mock("../../llm_calls/fetch_agents", () => ({
  fetchAvailableAgentModels: (...args: unknown[]) => fetchAvailableAgentModels(...args),
}));

vi.mock("@/components/llm_calls/fetch_models", () => ({
  fetchAvailableModels: (...args: unknown[]) => fetchAvailableModels(...args),
}));

vi.mock("@/components/CodeBlock", () => ({
  default: ({ code }: { code: string }) => <pre data-testid="code-block">{code}</pre>,
}));

const StatefulPanel = ({ label }: { label: string }) => {
  const [draft, setDraft] = useState("");
  return <input aria-label={label} value={draft} onChange={(event) => setDraft(event.target.value)} />;
};

vi.mock("./ChatUI", () => ({
  default: () => <StatefulPanel label="chat scratch" />,
}));

vi.mock("../complianceUI/ComplianceUI", () => ({
  default: () => <StatefulPanel label="batch scratch" />,
}));

const AGENTS: AgentModel[] = [
  {
    model_name: "support-agent",
    litellm_params: { model: "litellm_agent/gpt-4o", litellm_system_prompt: "Be helpful.", temperature: 0.3 },
    model_info: { id: "agent-1" },
  },
  {
    model_name: "research-agent",
    litellm_params: { model: "litellm_agent/claude-sonnet-4" },
    model_info: { id: "agent-2" },
  },
];

const props = {
  accessToken: "sk-access",
  token: "tok",
  userID: "u1",
  userRole: "Admin",
};

const controlUnder = (label: string): HTMLElement =>
  within(screen.getByText(label).parentElement!).getByRole("combobox");

const renderView = () => render(<AgentBuilderView {...props} />);

const waitForRoster = async () => {
  await screen.findByRole("button", { name: "support-agent litellm_agent" });
};

beforeEach(() => {
  vi.clearAllMocks();
  fetchAvailableAgentModels.mockResolvedValue(AGENTS);
  fetchAvailableModels.mockResolvedValue([{ model_group: "gpt-4o" }, { model_group: "claude-sonnet-4" }]);
  fetchMCPServers.mockResolvedValue([]);
  modelCreateCall.mockResolvedValue({ model_id: "id-new" });
  Object.defineProperty(window, "matchMedia", {
    writable: true,
    value: (query: string) => ({
      matches: false,
      media: query,
      onchange: null,
      addListener: () => {},
      removeListener: () => {},
      addEventListener: () => {},
      removeEventListener: () => {},
      dispatchEvent: () => false,
    }),
  });
});

describe("AgentBuilderView", () => {
  it("asks the visitor to sign in when there is no session", () => {
    render(<AgentBuilderView accessToken={null} token={null} userID={null} userRole={null} />);

    expect(screen.getByText("Sign in to use Agent Builder.")).toBeInTheDocument();
    expect(fetchAvailableAgentModels).not.toHaveBeenCalled();
  });

  it("marks itself busy while the roster loads", async () => {
    let release: (agents: AgentModel[]) => void = () => {};
    fetchAvailableAgentModels.mockReturnValue(
      new Promise<AgentModel[]>((resolve) => {
        release = resolve;
      }),
    );

    renderView();

    expect(document.querySelector('[aria-busy="true"]')).toBeInTheDocument();

    release(AGENTS);
    await waitForRoster();
    expect(document.querySelector('[aria-busy="true"]')).not.toBeInTheDocument();
  });

  it("lists every agent and opens the first one's configuration", async () => {
    renderView();
    await waitForRoster();

    expect(screen.getByRole("button", { name: "research-agent litellm_agent" })).toBeInTheDocument();
    expect(screen.getByText("Agent Builder")).toBeInTheDocument();
    expect(await screen.findByDisplayValue("support-agent")).toBeInTheDocument();
    expect(screen.getByDisplayValue("Be helpful.")).toBeInTheDocument();
    expect(screen.getByDisplayValue("0.3")).toBeInTheDocument();
  });

  it("loads the configuration of whichever agent is picked", async () => {
    const user = userEvent.setup();
    renderView();
    await waitForRoster();

    await user.click(screen.getByRole("button", { name: "research-agent litellm_agent" }));

    expect(await screen.findByDisplayValue("research-agent")).toBeInTheDocument();
  });

  it("offers a blank draft and a save control for a new agent", async () => {
    const user = userEvent.setup();
    renderView();
    await waitForRoster();

    await user.click(screen.getByRole("button", { name: /New agent/i }));

    expect(screen.getByRole("button", { name: /Save Agent/i })).toBeInTheDocument();
    expect(screen.getByDisplayValue("You are a helpful assistant.")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Update Agent/i })).not.toBeInTheDocument();
  });

  it("creates the agent under the litellm_agent prefix", async () => {
    const user = userEvent.setup();
    renderView();
    await waitForRoster();

    await user.click(screen.getByRole("button", { name: /New agent/i }));
    fireEvent.change(screen.getByPlaceholderText("My Agent"), { target: { value: "billing-agent" } });
    await user.click(screen.getByRole("button", { name: /Save Agent/i }));

    await waitFor(() => expect(modelCreateCall).toHaveBeenCalled());
    const payload = modelCreateCall.mock.calls[0][1];
    expect(payload.model_name).toBe("billing-agent");
    expect(payload.litellm_params.model).toBe("litellm_agent/gpt-4o");
  });

  it("will not save a draft without a name", async () => {
    const user = userEvent.setup();
    renderView();
    await waitForRoster();

    await user.click(screen.getByRole("button", { name: /New agent/i }));
    await user.click(screen.getByRole("button", { name: /Save Agent/i }));

    expect(modelCreateCall).not.toHaveBeenCalled();
  });

  it("updates the selected agent through its model id", async () => {
    const user = userEvent.setup();
    renderView();
    await waitForRoster();
    await screen.findByDisplayValue("support-agent");

    await user.click(screen.getByRole("button", { name: /Update Agent/i }));

    await waitFor(() => expect(modelPatchUpdateCall).toHaveBeenCalled());
    expect(modelPatchUpdateCall.mock.calls[0][2]).toBe("agent-1");
  });

  it("deletes only after the warning is confirmed", async () => {
    const user = userEvent.setup();
    renderView();
    await waitForRoster();
    await screen.findByDisplayValue("support-agent");

    await user.click(screen.getAllByRole("button", { name: /Delete$/ })[0]);

    expect(await screen.findByText(/Are you sure you want to delete "support-agent"/)).toBeInTheDocument();
    expect(modelDeleteCall).not.toHaveBeenCalled();

    const confirmations = screen.getAllByRole("button", { name: /Delete$/ });
    await user.click(confirmations[confirmations.length - 1]);

    await waitFor(() => expect(modelDeleteCall).toHaveBeenCalledWith("sk-access", "agent-1"));
  });

  it("abandons the delete when the warning is dismissed", async () => {
    const user = userEvent.setup();
    renderView();
    await waitForRoster();
    await screen.findByDisplayValue("support-agent");

    await user.click(screen.getAllByRole("button", { name: /Delete$/ })[0]);
    await screen.findByText(/Are you sure you want to delete "support-agent"/);
    await user.click(screen.getAllByRole("button", { name: /Cancel$/ }).at(-1)!);

    await waitFor(() =>
      expect(screen.queryByText(/Are you sure you want to delete "support-agent"/)).not.toBeInTheDocument(),
    );
    expect(modelDeleteCall).not.toHaveBeenCalled();
  });

  it("shows a ready-to-run curl example on the Connect tab", async () => {
    const user = userEvent.setup();
    renderView();
    await waitForRoster();

    await user.click(screen.getByRole("tab", { name: /Connect/i }));

    const snippet = await screen.findByTestId("code-block");
    expect(snippet).toHaveTextContent("https://proxy.example.com/v1/chat/completions");
    expect(snippet).toHaveTextContent('"model": "support-agent"');
  });

  it("mints a key scoped to the selected agent", async () => {
    const user = userEvent.setup();
    renderView();
    await waitForRoster();

    await user.click(screen.getByRole("tab", { name: /Connect/i }));
    await user.click(await screen.findByRole("button", { name: /Create key for this agent/i }));

    await waitFor(() => expect(keyCreateCall).toHaveBeenCalled());
    expect(keyCreateCall.mock.calls[0][2].models).toEqual(["support-agent"]);
    expect(await screen.findByTestId("code-block")).toHaveTextContent("Bearer sk-agent-key");
  });

  it("keeps a tab's own state alive while the user works in another tab", async () => {
    const user = userEvent.setup();
    renderView();
    await waitForRoster();

    await user.click(screen.getByRole("tab", { name: /Chat/i }));
    const scratch = await screen.findByLabelText("chat scratch");
    fireEvent.change(scratch, { target: { value: "half a thought" } });
    expect(scratch).toHaveValue("half a thought");

    await user.click(screen.getByRole("tab", { name: /Configure/i }));
    await screen.findByDisplayValue("support-agent");

    await user.click(screen.getByRole("tab", { name: /Chat/i }));
    expect(await screen.findByLabelText("chat scratch")).toHaveValue("half a thought");
  });

  it("keeps the batch tab's state alive across a round trip too", async () => {
    const user = userEvent.setup();
    renderView();
    await waitForRoster();

    await user.click(screen.getByRole("tab", { name: /Batch Test/i }));
    fireEvent.change(await screen.findByLabelText("batch scratch"), { target: { value: "seven cases" } });

    await user.click(screen.getByRole("tab", { name: /Connect/i }));
    await screen.findByTestId("code-block");

    await user.click(screen.getByRole("tab", { name: /Batch Test/i }));
    expect(await screen.findByLabelText("batch scratch")).toHaveValue("seven cases");
  });

  it("attaches the MCP servers the agent should reach", async () => {
    const user = userEvent.setup();
    fetchMCPServers.mockResolvedValue([{ server_id: "srv-1", alias: "github", server_name: "github-mcp" }]);
    renderView();
    await waitForRoster();
    await screen.findByDisplayValue("support-agent");

    await user.click(controlUnder("MCP servers"));
    const options = await screen.findAllByText("github");
    await user.click(options[options.length - 1]);
    await user.keyboard("{Escape}");

    await user.click(screen.getByRole("button", { name: /Update Agent/i }));

    await waitFor(() => expect(modelPatchUpdateCall).toHaveBeenCalled());
    expect(modelPatchUpdateCall.mock.calls[0][1].litellm_params.tools).toEqual([
      { type: "mcp", server_label: "litellm", server_url: "litellm_proxy/mcp/github", require_approval: "never" },
    ]);
  });

  it("warns that the builder is experimental", async () => {
    renderView();
    await waitForRoster();

    expect(screen.getByText(/Agent Builder is experimental/)).toBeInTheDocument();
    expect(within(screen.getByText(/Agent Builder is experimental/)).getByRole("link")).toHaveAttribute(
      "href",
      "mailto:product@berri.ai",
    );
  });
});
