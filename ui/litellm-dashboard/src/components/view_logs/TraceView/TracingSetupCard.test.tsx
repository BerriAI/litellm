import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { chooseSelectOption, renderWithProviders } from "@/../tests/test-utils";
import { copyToClipboard } from "@/utils/dataUtils";
import { agentTraceCall, apiClient, sendOtlpTraceCall } from "../../networking";
import {
  codingAgentCommand,
  codingAgentPrompt,
  maskSecret,
  TRACING_KEY_REQUEST,
  tracingEnvSnippet,
  TracingSetupCard,
} from "./TracingSetupCard";
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
  return { onOpenTrace, card: screen.getByTestId("tracing-setup-card") };
};

beforeEach(() => vi.clearAllMocks());

describe("TracingSetupCard", () => {
  it("shows agent connection guidance and labels the example run as sample data", async () => {
    const user = userEvent.setup();
    const { card } = renderCard();
    expect(screen.getByRole("heading", { name: "Connect your agent" })).toBeVisible();
    expect(screen.getByText("Tracing enabled")).toBeVisible();
    expect(screen.getByText("Waiting for your first trace")).toBeVisible();
    expect(screen.getByTestId("trace-preview")).toBeVisible();
    expect(card).toHaveTextContent("sample data, not your runs");
    expect(card).not.toHaveTextContent("store: clickhouse");
    await user.click(screen.getByText("Set up manually"));
    expect(screen.getByText(/^export OTEL_EXPORTER_OTLP_ENDPOINT=http:\/\/proxy.test/)).toBeVisible();
    expect(card).not.toHaveTextContent(/langsmith/i);
  });

  it("lists the OTEL endpoints before the framework picker, each copyable", async () => {
    const user = userEvent.setup();
    const { card } = renderCard();
    const text = card.textContent ?? "";
    expect(text.indexOf("OpenTelemetry (OTEL) endpoints")).toBeLessThan(text.indexOf("Your agent framework"));
    await user.click(screen.getByRole("button", { name: "Copy http://proxy.test/v1/traces" }));
    expect(copyToClipboard).toHaveBeenLastCalledWith("http://proxy.test/v1/traces");
  });

  it("hides the sample preview once traces are arriving", () => {
    renderCard({ connected: true });
    expect(screen.getByRole("heading", { name: "Connect another agent" })).toBeVisible();
    expect(screen.queryByTestId("trace-preview")).not.toBeInTheDocument();
  });

  it("builds the coding agent command for the selected framework and keeps both manual installers", async () => {
    const user = userEvent.setup();
    renderCard();
    await chooseSelectOption(user, screen.getByRole("combobox", { name: "Your agent framework" }), "CrewAI");
    const prompt = codingAgentPrompt("http://proxy.test", {
      label: "CrewAI",
      packages: "crewai openinference-instrumentation-crewai",
    });
    const commandText = () => screen.getByText(/^claude |^codex /).textContent;
    expect(commandText()).toBe(codingAgentCommand("Claude Code", prompt));
    await user.click(screen.getByRole("tab", { name: "Codex" }));
    expect(commandText()).toBe(codingAgentCommand("Codex", prompt));

    await user.click(screen.getByText("Set up manually"));
    expect(screen.getByText(/pip install -U opentelemetry-distro/)).toHaveTextContent(
      "crewai openinference-instrumentation-crewai",
    );
    await user.click(screen.getByRole("tab", { name: "uv" }));
    expect(screen.getByText(/uv add opentelemetry-distro/)).toHaveTextContent(
      "crewai openinference-instrumentation-crewai",
    );
  });

  it("uses npm and a CommonJS-safe TypeScript entrypoint for the Vercel AI SDK", async () => {
    const user = userEvent.setup();
    const { card } = renderCard();
    await chooseSelectOption(user, screen.getByRole("combobox", { name: "Your agent framework" }), "Vercel AI SDK");
    await user.click(screen.getByText("Set up manually"));
    expect(card).toHaveTextContent("npm install ai @ai-sdk/openai-compatible @vercel/otel");
    expect(card).toHaveTextContent("my_agent.ts");
    expect(card).not.toHaveTextContent("opentelemetry-instrument python");
    const quickstart = screen.getByText(/registerOTel\(\{/).textContent ?? "";
    expect(quickstart).toContain("async function main()");
    expect(quickstart.split("\n").filter((line) => /^(const|let) .*= await /.test(line))).toEqual([]);
  });

  it("hides the actions a read-only viewer cannot perform", () => {
    const { card } = renderCard({ readOnly: true });
    expect(screen.queryByRole("button", { name: "Send a test trace" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Generate tracing key" })).not.toBeInTheDocument();
    expect(card).toHaveTextContent("ask a proxy admin for one");
    expect(card).toHaveTextContent("OpenTelemetry (OTEL) endpoints");
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
    expect(card).toHaveTextContent("store: clickhouse");
    expect(screen.queryByRole("combobox", { name: "Your agent framework" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Send a test trace" })).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Check setup" }));
    expect(onCheck).toHaveBeenCalledOnce();
    expect(screen.getByText(/Tracing is still unavailable/)).toBeVisible();
  });
});

describe("setup snippets", () => {
  it("uses OTLP HTTP with the proxy base URL and reads the key from the environment", () => {
    const env = tracingEnvSnippet("http://proxy.test");
    expect(env).toContain("OTEL_EXPORTER_OTLP_ENDPOINT=http://proxy.test\n");
    expect(env).toContain("OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf");
    expect(env).not.toContain("/v1/traces");
    expect(env).not.toContain("export LITELLM_API_KEY=");
    expect(env).toContain("Bearer $LITELLM_API_KEY");
    const withKey = tracingEnvSnippet("http://proxy.test", SECRET);
    expect(withKey).toContain(`export LITELLM_TRACING_KEY=${SECRET}\n`);
    expect(withKey).toContain('OTEL_EXPORTER_OTLP_HEADERS="Authorization=Bearer $LITELLM_TRACING_KEY"');
    expect(withKey).not.toContain("LITELLM_API_KEY");

    const prompt = codingAgentPrompt("http://proxy.test", { label: "LangChain", packages: "langchain" });
    expect(prompt).toContain("base_url=http://proxy.test/v1");
    expect(prompt).toContain("opentelemetry-distro opentelemetry-exporter-otlp-proto-http langchain");
    expect(prompt).toContain("OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf");
    expect(prompt).toContain("Lens > Traces");
    expect(prompt).not.toMatch(/langsmith/i);
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
