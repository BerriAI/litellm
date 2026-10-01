import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import {
  codingAgentPrompt,
  inCodeSnippet,
  otlpEndpointRows,
  tracingEnvSnippet,
  TracingSetupCard,
} from "./TracingSetupCard";

vi.mock("../../networking", () => ({ getProxyBaseUrl: () => "http://proxy.test/" }));
vi.mock("@/utils/dataUtils", () => ({ copyToClipboard: vi.fn().mockResolvedValue(true) }));

const IN_CODE = "In code";
const ENV_TAB = "Environment variables (no code)";

const openInCode = async (user: ReturnType<typeof userEvent.setup>) => {
  await user.click(screen.getByRole("tab", { name: IN_CODE }));
};

describe("TracingSetupCard", () => {
  it("shows the waiting state and OTEL-only setup when tracing is on", () => {
    render(<TracingSetupCard detail={null} />);

    const card = screen.getByTestId("tracing-setup-card");
    expect(card).toHaveTextContent("Waiting for traces");
    expect(card).toHaveTextContent("No traces detected yet. Follow our guide to start tracing your application.");
    expect(card).not.toHaveTextContent(/langsmith/i);
    expect(card).not.toHaveTextContent("store: clickhouse");
  });

  it("documents the OTLP endpoint: base URL, ingest route, protocol, auth scoping and read APIs", () => {
    render(<TracingSetupCard detail={null} />);

    const endpoint = screen.getByTestId("otlp-endpoint");
    expect(endpoint).toHaveTextContent("http://proxy.test");
    expect(endpoint).toHaveTextContent("POST http://proxy.test/v1/traces");
    expect(endpoint).toHaveTextContent("protobuf");
    expect(endpoint).toHaveTextContent("Traces are scoped to that key and its team");
    expect(endpoint).toHaveTextContent("GET http://proxy.test/v1/traces/{trace_id}?format=md");
  });

  it("instruments Deep Agents by default with env vars, including the protocol and the HTTP-noise fix", () => {
    render(<TracingSetupCard detail={null} />);

    expect(screen.getByRole("radio", { name: "Deep Agents" })).toBeChecked();
    expect(screen.getByRole("tab", { name: ENV_TAB })).toHaveAttribute("aria-selected", "true");
    const instrument = screen.getByRole("region", { name: "Instrument your agent" });
    expect(instrument).toHaveTextContent("deepagents langchain-openai openinference-instrumentation-langchain");
    expect(instrument).toHaveTextContent("OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf");
    expect(instrument).toHaveTextContent("OTEL_PYTHON_DISABLED_INSTRUMENTATIONS=httpx,requests");
    expect(instrument).toHaveTextContent("create_deep_agent");
    expect(instrument).toHaveTextContent("opentelemetry-instrument python my_agent.py");
  });

  it("walks Deep Agents through in-code setup with the full ingest URL and a flush", async () => {
    const user = userEvent.setup();
    render(<TracingSetupCard detail={null} />);
    await openInCode(user);

    const instrument = screen.getByRole("region", { name: "Instrument your agent" });
    expect(instrument).toHaveTextContent("LangChainInstrumentor().instrument(tracer_provider=provider)");
    expect(instrument).toHaveTextContent('endpoint=f"{LITELLM_PROXY}/v1/traces"');
    expect(instrument).toHaveTextContent("create_deep_agent(model=llm");
    expect(instrument).toHaveTextContent("provider.shutdown()");
  });

  it("uses create_agent for LangChain / LangGraph in code", async () => {
    const user = userEvent.setup();
    render(<TracingSetupCard detail={null} />);
    await user.click(screen.getByRole("radio", { name: "LangChain / LangGraph" }));
    await openInCode(user);

    const instrument = screen.getByRole("region", { name: "Instrument your agent" });
    expect(instrument).toHaveTextContent("from langchain.agents import create_agent");
    expect(instrument).not.toHaveTextContent("create_deep_agent");
  });

  it("offers only the env-var route for frameworks whose in-code instrumentor isn't documented", async () => {
    const user = userEvent.setup();
    render(<TracingSetupCard detail={null} />);
    await openInCode(user);
    await user.click(screen.getByRole("radio", { name: "CrewAI" }));

    expect(screen.queryByRole("tab", { name: IN_CODE })).not.toBeInTheDocument();
    const instrument = screen.getByRole("region", { name: "Instrument your agent" });
    expect(instrument).toHaveTextContent("crewai openinference-instrumentation-crewai");
    expect(instrument).not.toHaveTextContent("LangChainInstrumentor");
  });

  it("gives coding agents a prompt that sets tracing up and verifies it", () => {
    render(<TracingSetupCard detail={null} />);

    const prompt = within(screen.getByRole("region", { name: "Let Claude Code or Codex set it up" }));
    expect(prompt.getByText(/Send this Deep Agents project/)).toBeInTheDocument();
    expect(screen.getByRole("region", { name: "Verify" })).toHaveTextContent("http://proxy.test/v1/traces?start_ms=");
  });

  it("shows the proxy config step only when tracing is not enabled", () => {
    render(<TracingSetupCard detail="Agent tracing is not enabled" />);
    const card = screen.getByTestId("tracing-setup-card");
    expect(card).toHaveTextContent("Tracing is not enabled");
    expect(card).toHaveTextContent("store: clickhouse");
  });
});

describe("setup snippets", () => {
  it("env endpoint is the base URL while the in-code exporter uses the full ingest route", () => {
    const env = tracingEnvSnippet("http://proxy.test");
    expect(env).toContain("OTEL_EXPORTER_OTLP_ENDPOINT=http://proxy.test\n");
    expect(env).not.toContain("/v1/traces");
    expect(env).toContain("Bearer $LITELLM_API_KEY");

    const code = inCodeSnippet("http://proxy.test", "agent = ...");
    expect(code).toContain('endpoint=f"{LITELLM_PROXY}/v1/traces"');
    expect(code).toContain('os.environ["LITELLM_API_KEY"]');
  });

  it("exposes endpoint rows another panel can render", () => {
    const rows = otlpEndpointRows("http://proxy.test");
    expect(rows.map((row) => row.label)).toContain("Ingest route");
    expect(rows.find((row) => row.label === "OTLP endpoint")?.value).toBe("http://proxy.test");
  });

  it("coding-agent prompt covers deps, env, model routing, names, flush and verification without a real key", () => {
    const prompt = codingAgentPrompt("http://proxy.test", { label: "LangChain", packages: "langchain" });
    expect(prompt).toContain("opentelemetry-distro opentelemetry-exporter-otlp-proto-http langchain");
    expect(prompt).toContain("OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf");
    expect(prompt).toContain("OTEL_PYTHON_DISABLED_INSTRUMENTATIONS=");
    expect(prompt).toContain("base_url=http://proxy.test/v1");
    expect(prompt).toContain("name=");
    expect(prompt).toContain("provider.shutdown()");
    expect(prompt).toContain("Verify: run the agent once, then GET http://proxy.test/v1/traces");
    expect(prompt).not.toMatch(/sk-[A-Za-z0-9]/);
    expect(prompt).not.toMatch(/langsmith/i);
  });
});
