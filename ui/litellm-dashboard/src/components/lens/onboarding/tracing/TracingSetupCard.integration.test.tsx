import { act, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { chooseSelectOption, renderWithProviders, testQueryClient } from "@/../tests/test-utils";
import { copyToClipboard } from "@/utils/dataUtils";
import { agentTraceCall, apiClient } from "../../../networking";
import {
  codingAgentCommand,
  codingAgentPrompt,
  maskSecret,
  TRACING_KEY_REQUEST,
  tracingEnvSnippet,
  TracingSetupCard,
} from "./TracingSetupCard";
import { FRAMEWORKS } from "./tracingSetupGuides";
import type { Trace } from "../../traces/types";

vi.mock("../../../networking", () => ({
  getProxyBaseUrl: () => "http://proxy.test/",
  agentTraceCall: vi.fn(),
  apiClient: { post: vi.fn(), get: vi.fn() },
}));
vi.mock("@/utils/dataUtils", () => ({ copyToClipboard: vi.fn().mockResolvedValue(true) }));

const SECRET = "sk-abcdefghijklmnopWXYZ";

const renderCard = async (
  props: {
    detail?: string | null;
    connected?: boolean;
    onCheck?: () => void;
    readOnly?: boolean;
    canMintTracingKey?: boolean;
  } = {},
) => {
  const onOpenTrace = vi.fn();
  renderWithProviders(
    <TracingSetupCard
      detail={props.detail ?? null}
      connected={props.connected}
      onCheck={props.onCheck}
      readOnly={props.readOnly}
      canMintTracingKey={props.canMintTracingKey ?? true}
      accessToken="sk-admin"
      onOpenTrace={onOpenTrace}
    />,
  );
  if (!props.detail) await screen.findByRole("combobox", { name: "Your agent framework" });
  return { onOpenTrace, card: screen.getByTestId("tracing-setup-card") };
};

const network = vi.fn<typeof fetch>();
beforeEach(() => {
  testQueryClient.clear();
  vi.clearAllMocks();
  vi.stubGlobal("fetch", network);
  network.mockResolvedValue(Response.json({}));
  const readyService = {
    url: "https://traces.test",
    configured: true,
    release: "v1.2.3",
    connected: true,
    status: { storage_ready: true, credentials_ready: true },
  };
  vi.mocked(apiClient.get).mockResolvedValue(readyService);
  vi.mocked(apiClient.post).mockResolvedValue({ key: SECRET, active: true });
});

describe("TracingSetupCard", () => {
  it("guides agent connection while waiting for the first trace", async () => {
    const user = userEvent.setup();
    const { card } = await renderCard();
    expect(screen.getByRole("heading", { name: "Connect your agent" })).toBeVisible();
    expect(screen.getByText("Tracing enabled")).toBeVisible();
    expect(screen.getByText("Waiting for your first trace")).toBeVisible();
    expect(screen.queryByRole("button", { name: "Preview sample" })).not.toBeInTheDocument();
    expect(network).not.toHaveBeenCalled();
    expect(card).not.toHaveTextContent("store: clickhouse");
    expect(screen.getByText(/^export LITELLM_TRACING_KEY=/)).toBeVisible();
    expect(card).not.toHaveTextContent(/langsmith/i);
  });

  it("keeps connection details visible and copies the full trace endpoint", async () => {
    const user = userEvent.setup();
    await renderCard();
    expect(screen.getByRole("combobox", { name: "Your agent framework" })).toBeVisible();
    expect(screen.getByRole("button", { name: "Copy https://traces.test/v1/traces" })).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Copy https://traces.test/v1/traces" }));
    expect(copyToClipboard).toHaveBeenLastCalledWith("https://traces.test/v1/traces");
  });

  it("shows connection guidance for another agent without a demo", async () => {
    await renderCard({ connected: true });
    expect(screen.getByRole("heading", { name: "Connect another agent" })).toBeVisible();
    expect(screen.queryByRole("button", { name: "Preview sample" })).not.toBeInTheDocument();
  });

  it("builds the coding agent command for the selected framework and keeps both manual installers", async () => {
    const user = userEvent.setup();
    await renderCard();
    await chooseSelectOption(user, screen.getByRole("combobox", { name: "Your agent framework" }), "CrewAI");
    const prompt = codingAgentPrompt(
      "http://proxy.test",
      "https://traces.test",
      FRAMEWORKS.find((guide) => guide.id === "crewai")!,
      "openai/gpt-6.1-sol",
    );
    expect(screen.getByText(/^claude /)).not.toBeVisible();
    await user.click(screen.getByRole("button", { name: "Copy setup command" }));
    expect(copyToClipboard).toHaveBeenLastCalledWith(codingAgentCommand("Claude Code", prompt));
    await user.click(screen.getByRole("tab", { name: "Codex" }));
    await user.click(screen.getByRole("button", { name: "Copy setup command" }));
    expect(copyToClipboard).toHaveBeenLastCalledWith(codingAgentCommand("Codex", prompt));
    await user.click(screen.getByText("View command"));
    expect(screen.getByText(/^codex /)).toBeVisible();
    expect(screen.getByText(/^codex /)).toHaveTextContent(codingAgentCommand("Codex", prompt), {
      normalizeWhitespace: false,
    });

    expect(screen.getByText(/^pip install opentelemetry-distro/)).toHaveTextContent(
      "crewai openinference-instrumentation-crewai",
    );
    await user.click(screen.getByRole("tab", { name: "uv" }));
    expect(screen.getByText(/^uv add opentelemetry-distro/)).toHaveTextContent(
      "crewai openinference-instrumentation-crewai",
    );
  });

  it("uses the selected framework's tracing and agent name without asking for a model", async () => {
    const user = userEvent.setup();
    const { card } = await renderCard();
    await chooseSelectOption(user, screen.getByRole("combobox", { name: "Your agent framework" }), "Vercel AI SDK");
    expect(screen.queryByRole("combobox", { name: "Model" })).not.toBeInTheDocument();
    expect(card).toHaveTextContent("npm install ai @ai-sdk/otel");
    expect(card).toHaveTextContent('const AGENT_NAME = "research_agent"');
    expect(card).toHaveTextContent("functionId: AGENT_NAME");
    expect(card).toHaveTextContent("Use a model configured on this proxy.");
    expect(screen.getByText(/^import \{ createOpenAICompatible/)).toHaveTextContent(
      'const model = litellm("openai/gpt-6.1-sol")',
    );
    expect(card).toHaveTextContent('baseURL: "http://proxy.test/v1"');
  });

  it("keeps plugin model settings and uses a generated tracing key only for tracing", async () => {
    const user = userEvent.setup();
    vi.mocked(apiClient.post).mockResolvedValue({ key: SECRET });
    const { card } = await renderCard();
    await chooseSelectOption(user, screen.getByRole("combobox", { name: "Your agent framework" }), "Hermes");
    expect(screen.queryByRole("combobox", { name: "Model" })).not.toBeInTheDocument();
    expect(card).toHaveTextContent("Keep your existing model settings");
    await user.click(screen.getByRole("button", { name: "Generate tracing key" }));
    await screen.findByText("Your tracing key");
    expect(card).toHaveTextContent("gen_ai.agent.name: research_agent");
    expect(card).toHaveTextContent("endpoint: https://traces.test/v1/traces");
    expect(card).toHaveTextContent('Authorization: "Bearer ${LITELLM_TRACING_KEY}"');
    expect(card).not.toHaveTextContent(SECRET);
  });

  it("hides the actions a read-only viewer cannot perform", async () => {
    const { card } = await renderCard({ readOnly: true });
    expect(screen.queryByRole("button", { name: "Send a test trace" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Generate tracing key" })).not.toBeInTheDocument();
    expect(card).toHaveTextContent("Ask your proxy admin for a dedicated Lens tracing key.");
    expect(card).toHaveTextContent("Connection details");
  });

  it("offers a scoped tracing key only to callers allowed to create tracing keys", async () => {
    const { card } = await renderCard({ canMintTracingKey: false });
    expect(screen.queryByRole("button", { name: "Generate tracing key" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Send a test trace" })).toBeVisible();
    expect(card).toHaveTextContent("Ask your proxy admin for a dedicated Lens tracing key.");
  });

  it("generates a tracing key that stays masked on screen but copies in full", async () => {
    const user = userEvent.setup();
    vi.mocked(apiClient.post).mockResolvedValue({ key: SECRET });
    const { card } = await renderCard();
    await user.click(screen.getByRole("button", { name: "Generate tracing key" }));

    expect(await screen.findByText("Your tracing key")).toBeVisible();
    expect(apiClient.post).toHaveBeenCalledWith("/lens/tracing/keys", {
      accessToken: "sk-admin",
      body: TRACING_KEY_REQUEST,
    });
    expect(card).not.toHaveTextContent(SECRET);
    expect(card).toHaveTextContent(maskSecret(SECRET));
    await user.click(screen.getAllByRole("button", { name: "Copy" })[0]);
    expect(copyToClipboard).toHaveBeenLastCalledWith(SECRET);
  });

  it("sends a test trace, waits for it to land, then opens it", async () => {
    const user = userEvent.setup();
    const summary = { trace_id: "abc", name: "weather_agent" } as Trace["summary"];
    network.mockResolvedValue(Response.json({}));
    vi.mocked(agentTraceCall).mockResolvedValue({ summary, agents: [], spans: [] } as unknown as Trace);
    const { onOpenTrace } = await renderCard();

    await user.click(screen.getByRole("button", { name: "Generate tracing key" }));
    await screen.findByText("Your tracing key");
    await user.click(screen.getByRole("button", { name: "Send a test trace" }));
    await user.click(await screen.findByRole("button", { name: /View trace/ }));

    const uploadOptions = {
      method: "POST",
      credentials: "omit",
      redirect: "error",
      headers: { Accept: "application/json", "Content-Type": "application/json", Authorization: `Bearer ${SECRET}` },
    };
    expect(network).toHaveBeenCalledWith("https://traces.test/v1/traces", expect.objectContaining(uploadOptions));
    expect(vi.mocked(agentTraceCall).mock.calls[0][1]).toMatch(/^[0-9a-f]{32}$/);
    expect(onOpenTrace).toHaveBeenCalledWith(summary);
  });

  it("reports a failed send instead of claiming success", async () => {
    const user = userEvent.setup();
    network.mockRejectedValue(new Error("boom"));
    await renderCard();

    await user.click(screen.getByRole("button", { name: "Generate tracing key" }));
    await screen.findByText("Your tracing key");
    await user.click(screen.getByRole("button", { name: "Send a test trace" }));

    expect(await screen.findByText("Could not send the test trace.")).toBeVisible();
    expect(agentTraceCall).not.toHaveBeenCalled();
    expect(screen.queryByRole("button", { name: /View trace/ })).not.toBeInTheDocument();
  });

  it("shows matching-version installation instructions without changing the landing design", async () => {
    const user = userEvent.setup();
    const onCheck = vi.fn();
    const missingService = {
      configured: false,
      connected: false,
      release: "v1.2.3",
      url: "",
      status: {},
    };
    vi.mocked(apiClient.get).mockResolvedValue(missingService);
    const { card } = await renderCard({ detail: "Agent tracing is not enabled", onCheck });
    expect(await screen.findByText("Install Lens")).toBeVisible();
    expect(card).toHaveTextContent("Use Lens v1.2.3 to match this LiteLLM deployment");
    expect(screen.getByRole("link", { name: "Helm setup" })).toHaveAttribute(
      "href",
      "https://docs.litellm.ai/docs/proxy/lens/deployment#using-helm",
    );
    expect(screen.getByRole("link", { name: "Docker setup" })).toHaveAttribute(
      "href",
      "https://docs.litellm.ai/docs/proxy/lens/deployment#using-docker",
    );
    expect(screen.queryByRole("combobox", { name: "Your agent framework" })).not.toBeInTheDocument();
    const readyService = {
      configured: true,
      connected: true,
      url: "https://traces.test",
      status: { storage_ready: true },
    };
    vi.mocked(apiClient.get).mockResolvedValue(readyService);
    await user.click(screen.getByRole("button", { name: "Check setup" }));
    expect(onCheck).toHaveBeenCalledOnce();
    expect(await screen.findByRole("combobox", { name: "Your agent framework" })).toBeVisible();
  });

  it.each([
    [false, false, "Lens is configured, but LiteLLM cannot reach it"],
    [true, false, "Lens is connected, but its trace storage is unavailable"],
  ])(
    "explains a configured service failure without recommending reinstallation",
    async (connected, storageReady, message) => {
      const service = {
        configured: true,
        connected,
        url: "https://traces.test",
        status: { storage_ready: storageReady },
      };
      vi.mocked(apiClient.get).mockResolvedValue(service);
      await renderCard({ detail: "Service unavailable" });
      expect(await screen.findByText(message, { exact: false })).toBeVisible();
      expect(screen.queryByText("Install Lens")).not.toBeInTheDocument();
      expect(screen.queryByRole("button", { name: "Generate tracing key" })).not.toBeInTheDocument();
    },
  );

  it("preserves the framework and uncopied key across a background service outage", async () => {
    const user = userEvent.setup();
    await renderCard();
    await chooseSelectOption(user, screen.getByRole("combobox", { name: "Your agent framework" }), "LangGraph");
    await user.click(screen.getByRole("button", { name: "Generate tracing key" }));
    await screen.findByText("Your tracing key");
    const ready = testQueryClient.getQueryData(["lens-service", "sk-admin"]);
    const unavailable = {
      configured: true,
      connected: false,
      url: "https://traces.test",
      status: { storage_ready: false },
    };
    act(() => testQueryClient.setQueryData(["lens-service", "sk-admin"], unavailable));
    expect(await screen.findByText(/Lens is configured, but LiteLLM cannot reach it/)).toBeVisible();
    act(() => testQueryClient.setQueryData(["lens-service", "sk-admin"], ready));
    expect(await screen.findByRole("combobox", { name: "Your agent framework" })).toHaveTextContent("LangGraph");
    expect(screen.getByText("Your tracing key")).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Copy tracing configuration" }));
    expect(copyToClipboard).toHaveBeenLastCalledWith(tracingEnvSnippet("https://traces.test", SECRET));
    expect(apiClient.post).toHaveBeenCalledOnce();
  });

  it("shows the copyable configuration without another disclosure and includes the generated key", async () => {
    const user = userEvent.setup();
    await renderCard();
    expect(screen.getByText(/^export LITELLM_TRACING_KEY=/)).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Generate tracing key" }));
    await screen.findByText("Your tracing key");
    await user.click(screen.getByRole("button", { name: "Copy tracing configuration" }));
    expect(copyToClipboard).toHaveBeenLastCalledWith(tracingEnvSnippet("https://traces.test", SECRET));
  });
});

describe("setup snippets", () => {
  it("uses the instance trace endpoint and keeps tracing and inference keys separate", () => {
    const env = tracingEnvSnippet("https://traces.test");
    expect(env).toContain('OTEL_EXPORTER_OTLP_TRACES_ENDPOINT="https://traces.test/v1/traces"');
    expect(env).toContain('OTEL_EXPORTER_OTLP_PROTOCOL="http/protobuf"');
    expect(env).not.toContain("export LITELLM_API_KEY=");
    expect(env).toContain("Bearer $LITELLM_TRACING_KEY");
    const withKey = tracingEnvSnippet("https://traces.test", SECRET);
    expect(withKey).toContain(`export LITELLM_TRACING_KEY="${SECRET}"\n`);
    expect(withKey).toContain('OTEL_EXPORTER_OTLP_TRACES_HEADERS="Authorization=Bearer $LITELLM_TRACING_KEY"');
    expect(withKey).not.toContain("export LITELLM_API_KEY=");

    const prompt = codingAgentPrompt("http://proxy.test", "https://traces.test", FRAMEWORKS[0], "openai/gpt-6.1-sol");
    expect(prompt).toContain('OTEL_EXPORTER_OTLP_TRACES_ENDPOINT="https://traces.test/v1/traces"');
    expect(prompt).toContain("Keep the existing model configuration");
    expect(prompt).toContain('AGENT_NAME = "research_agent"');
    expect(prompt).toContain("name=AGENT_NAME");
    expect(prompt).toContain("Lens > Traces");
  });

  it("builds a shell-safe command for each coding agent", () => {
    expect(codingAgentCommand("Claude Code", "it's")).toBe("claude 'it'\\''s'");
    expect(codingAgentCommand("Codex", "go")).toBe("codex 'go'");
  });

  it("masks secrets but keeps a recognisable prefix and suffix", () => {
    expect(maskSecret(SECRET)).toBe(`sk-ab${"•".repeat(16)}WXYZ`);
    expect(maskSecret("short")).toBe("•••••");
  });
});
