import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { chooseSelectOption, renderWithProviders } from "@/../tests/test-utils";
import { copyToClipboard } from "@/utils/dataUtils";
import { LensPreviewContext } from "@/components/lens/LensPreviewButton";
import { agentTraceCall, apiClient, sendOtlpTraceCall } from "../../networking";
import {
  codingAgentCommand,
  codingAgentPrompt,
  maskSecret,
  TRACING_KEY_REQUEST,
  tracingEnvSnippet,
  TracingSetupCard,
} from "./TracingSetupCard";
import { FRAMEWORKS } from "./tracingSetupGuides";
import type { Trace } from "./traceTypes";

vi.mock("../../networking", () => ({
  getProxyBaseUrl: () => "http://proxy.test/",
  sendOtlpTraceCall: vi.fn(),
  agentTraceCall: vi.fn(),
  apiClient: { post: vi.fn() },
}));
vi.mock("@/utils/dataUtils", () => ({ copyToClipboard: vi.fn().mockResolvedValue(true) }));

const SECRET = "sk-abcdefghijklmnopWXYZ";

const renderCard = (
  props: {
    detail?: string | null;
    connected?: boolean;
    onCheck?: () => void;
    readOnly?: boolean;
    canMintTracingKey?: boolean;
    onPreview?: () => void;
  } = {},
) => {
  const onOpenTrace = vi.fn();
  renderWithProviders(
    <LensPreviewContext.Provider value={{ target: document.body, open: props.onPreview }}>
      <TracingSetupCard
        detail={props.detail ?? null}
        connected={props.connected}
        onCheck={props.onCheck}
        readOnly={props.readOnly}
        canMintTracingKey={props.canMintTracingKey ?? true}
        accessToken="sk-admin"
        onOpenTrace={onOpenTrace}
      />
    </LensPreviewContext.Provider>,
  );
  return { onOpenTrace, card: screen.getByTestId("tracing-setup-card") };
};

beforeEach(() => vi.clearAllMocks());

describe("TracingSetupCard", () => {
  it("offers the interactive demo while waiting for the first trace", async () => {
    const user = userEvent.setup();
    const onPreview = vi.fn();
    const { card } = renderCard({ onPreview });
    expect(screen.getByRole("heading", { name: "Connect your agent" })).toBeVisible();
    expect(screen.getByText("Tracing enabled")).toBeVisible();
    expect(screen.getByText("Waiting for your first trace")).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Preview sample" }));
    expect(onPreview).toHaveBeenCalledOnce();
    expect(sendOtlpTraceCall).not.toHaveBeenCalled();
    expect(card).not.toHaveTextContent("store: clickhouse");
    await user.click(screen.getByText("Set up manually"));
    expect(screen.getByText(/^export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=/)).toBeVisible();
    expect(card).not.toHaveTextContent(/langsmith/i);
  });

  it("keeps connection details visible and copies the full trace endpoint", async () => {
    const user = userEvent.setup();
    renderCard();
    expect(screen.getByRole("combobox", { name: "Your agent framework" })).toBeVisible();
    expect(screen.getByRole("button", { name: "Copy http://proxy.test/v1/traces" })).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Copy http://proxy.test/v1/traces" }));
    expect(copyToClipboard).toHaveBeenLastCalledWith("http://proxy.test/v1/traces");
  });

  it("shows connection guidance for another agent without a demo", () => {
    renderCard({ connected: true });
    expect(screen.getByRole("heading", { name: "Connect another agent" })).toBeVisible();
    expect(screen.queryByRole("button", { name: "Preview sample" })).not.toBeInTheDocument();
  });

  it("builds the coding agent command for the selected framework and keeps both manual installers", async () => {
    const user = userEvent.setup();
    renderCard();
    await chooseSelectOption(user, screen.getByRole("combobox", { name: "Your agent framework" }), "CrewAI");
    const prompt = codingAgentPrompt(
      "http://proxy.test",
      FRAMEWORKS.find((guide) => guide.id === "crewai")!,
      "openai/gpt-6-sol",
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

    await user.click(screen.getByText("Set up manually"));
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
    const { card } = renderCard();
    await chooseSelectOption(user, screen.getByRole("combobox", { name: "Your agent framework" }), "Vercel AI SDK");
    await user.click(screen.getByText("Set up manually"));
    expect(screen.queryByRole("combobox", { name: "Model" })).not.toBeInTheDocument();
    expect(card).toHaveTextContent("npm install ai @ai-sdk/otel");
    expect(card).toHaveTextContent('const AGENT_NAME = "research_agent"');
    expect(card).toHaveTextContent("functionId: AGENT_NAME");
    expect(card).toHaveTextContent("Use a model configured on this proxy.");
    expect(screen.getByText(/^import \{ createOpenAICompatible/)).toHaveTextContent(
      'const model = litellm("openai/gpt-6-sol")',
    );
    expect(card).toHaveTextContent('baseURL: "http://proxy.test/v1"');
  });

  it("keeps plugin model settings and uses a generated tracing key only for tracing", async () => {
    const user = userEvent.setup();
    vi.mocked(apiClient.post).mockResolvedValue({ key: SECRET });
    const { card } = renderCard();
    await chooseSelectOption(user, screen.getByRole("combobox", { name: "Your agent framework" }), "Hermes");
    await user.click(screen.getByText("Set up manually"));
    expect(screen.queryByRole("combobox", { name: "Model" })).not.toBeInTheDocument();
    expect(card).toHaveTextContent("Keep your existing model settings");
    await user.click(screen.getByRole("button", { name: "Generate tracing key" }));
    await screen.findByText("Your tracing key");
    expect(card).toHaveTextContent("gen_ai.agent.name: research_agent");
    expect(card).toHaveTextContent("endpoint: http://proxy.test/v1/traces");
    expect(card).toHaveTextContent('Authorization: "Bearer ${LITELLM_TRACING_KEY}"');
    expect(card).not.toHaveTextContent(SECRET);
  });

  it("hides the actions a read-only viewer cannot perform", () => {
    const { card } = renderCard({ readOnly: true });
    expect(screen.queryByRole("button", { name: "Send a test trace" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Generate tracing key" })).not.toBeInTheDocument();
    expect(card).toHaveTextContent("ask a proxy admin for one");
    expect(card).toHaveTextContent("Connection details");
  });

  it("offers a scoped tracing key only to callers allowed to set key routes", () => {
    const { card } = renderCard({ canMintTracingKey: false });
    expect(screen.queryByRole("button", { name: "Generate tracing key" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Send a test trace" })).toBeVisible();
    expect(card).toHaveTextContent("Use any LiteLLM virtual key you already have");
  });

  it("generates a tracing key that stays masked on screen but copies in full", async () => {
    const user = userEvent.setup();
    vi.mocked(apiClient.post).mockResolvedValue({ key: SECRET });
    const { card } = renderCard();
    await user.click(screen.getByText("Set up manually"));
    await user.click(screen.getByRole("button", { name: "Generate tracing key" }));

    expect(await screen.findByText("Your tracing key")).toBeVisible();
    expect(apiClient.post).toHaveBeenCalledWith("/key/generate", {
      accessToken: "sk-admin",
      body: TRACING_KEY_REQUEST,
    });
    expect(TRACING_KEY_REQUEST.allowed_routes).toEqual(["/v1/traces"]);
    expect(card).not.toHaveTextContent(SECRET);
    expect(card).toHaveTextContent(maskSecret(SECRET));
    await user.click(screen.getAllByRole("button", { name: "Copy" })[0]);
    expect(copyToClipboard).toHaveBeenLastCalledWith(SECRET);
  });

  it("sends a test trace, waits for it to land, then opens it", async () => {
    const user = userEvent.setup();
    const summary = { trace_id: "abc", name: "weather_agent" } as Trace["summary"];
    vi.mocked(sendOtlpTraceCall).mockResolvedValue(undefined);
    vi.mocked(agentTraceCall).mockResolvedValue({ summary, agents: [], spans: [] } as unknown as Trace);
    const { onOpenTrace } = renderCard();

    await user.click(screen.getByRole("button", { name: "Send a test trace" }));
    await user.click(await screen.findByRole("button", { name: /View trace/ }));

    expect(sendOtlpTraceCall).toHaveBeenCalledOnce();
    expect(vi.mocked(agentTraceCall).mock.calls[0][1]).toMatch(/^[0-9a-f]{32}$/);
    expect(onOpenTrace).toHaveBeenCalledWith(summary);
  });

  it("reports a failed send instead of claiming success", async () => {
    const user = userEvent.setup();
    vi.mocked(sendOtlpTraceCall).mockRejectedValue(new Error("boom"));
    renderCard();

    await user.click(screen.getByRole("button", { name: "Send a test trace" }));

    expect(await screen.findByText("Could not send the test trace.")).toBeVisible();
    expect(agentTraceCall).not.toHaveBeenCalled();
    expect(screen.queryByRole("button", { name: /View trace/ })).not.toBeInTheDocument();
  });

  it("guides proxy setup before agent setup and allows checking readiness", async () => {
    const user = userEvent.setup();
    const onCheck = vi.fn();
    const { card } = renderCard({ detail: "Agent tracing is not enabled", onCheck });
    expect(screen.getByRole("heading", { name: "Enable tracing" })).toBeVisible();
    expect(card).toHaveTextContent("type: clickhouse");
    expect(card).toHaveTextContent("url: os.environ/CLICKHOUSE_URL");
    expect(screen.queryByRole("combobox", { name: "Your agent framework" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Send a test trace" })).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Check setup" }));
    expect(onCheck).toHaveBeenCalledOnce();
    expect(screen.getByText(/Tracing is still unavailable/)).toBeVisible();
  });
});

describe("setup snippets", () => {
  it("uses the instance trace endpoint and keeps tracing and inference keys separate", () => {
    const env = tracingEnvSnippet("http://proxy.test");
    expect(env).toContain('OTEL_EXPORTER_OTLP_TRACES_ENDPOINT="http://proxy.test/v1/traces"');
    expect(env).toContain('OTEL_EXPORTER_OTLP_PROTOCOL="http/protobuf"');
    expect(env).not.toContain("export LITELLM_API_KEY=");
    expect(env).toContain("Bearer $LITELLM_API_KEY");
    const withKey = tracingEnvSnippet("http://proxy.test", SECRET);
    expect(withKey).toContain(`export LITELLM_TRACING_KEY=${SECRET}\n`);
    expect(withKey).toContain('OTEL_EXPORTER_OTLP_TRACES_HEADERS="Authorization=Bearer $LITELLM_TRACING_KEY"');
    expect(withKey).not.toContain("export LITELLM_API_KEY=");

    const prompt = codingAgentPrompt("http://proxy.test", FRAMEWORKS[0], "openai/gpt-6-sol");
    expect(prompt).toContain('OTEL_EXPORTER_OTLP_TRACES_ENDPOINT="http://proxy.test/v1/traces"');
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
