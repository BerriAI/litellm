import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { codingAgentPrompt, tracingEnvSnippet, TracingSetupCard } from "./TracingSetupCard";

vi.mock("../../networking", () => ({ getProxyBaseUrl: () => "http://proxy.test/" }));
vi.mock("@/utils/dataUtils", () => ({ copyToClipboard: vi.fn().mockResolvedValue(true) }));

describe("TracingSetupCard", () => {
  it("shows the waiting state and OTEL-only setup when tracing is on", () => {
    render(<TracingSetupCard detail={null} />);

    const card = screen.getByTestId("tracing-setup-card");
    expect(card).toHaveTextContent("Waiting for traces");
    expect(card).toHaveTextContent("No traces detected yet. Follow our guide to start tracing your application.");
    expect(card).toHaveTextContent("OTEL_EXPORTER_OTLP_ENDPOINT=http://proxy.test");
    expect(card).not.toHaveTextContent(/langsmith/i);
    expect(card).not.toHaveTextContent("store: clickhouse");
  });

  it("swaps the install command and prompt when another framework is picked", async () => {
    const user = userEvent.setup();
    render(<TracingSetupCard detail={null} />);

    expect(screen.getByTestId("tracing-setup-card")).toHaveTextContent("openinference-instrumentation-langchain");
    await user.click(screen.getByRole("radio", { name: "CrewAI" }));

    const card = screen.getByTestId("tracing-setup-card");
    expect(screen.getByRole("radio", { name: "CrewAI" })).toBeChecked();
    expect(card).toHaveTextContent("pip install -U opentelemetry-distro");
    expect(card).toHaveTextContent("crewai openinference-instrumentation-crewai");
    expect(card).toHaveTextContent("Send this CrewAI project's OpenTelemetry traces to LiteLLM.");
    expect(card).not.toHaveTextContent("openinference-instrumentation-langchain");
  });

  it("shows the proxy config step only when tracing is not enabled", () => {
    render(<TracingSetupCard detail="Agent tracing is not enabled" />);
    const card = screen.getByTestId("tracing-setup-card");
    expect(card).toHaveTextContent("Tracing is not enabled");
    expect(card).toHaveTextContent("store: clickhouse");
  });
});

describe("setup snippets", () => {
  it("points OTLP at the proxy base URL and reads the key from the environment", () => {
    const env = tracingEnvSnippet("http://proxy.test");
    expect(env).toContain("OTEL_EXPORTER_OTLP_ENDPOINT=http://proxy.test\n");
    expect(env).not.toContain("/v1/traces");
    expect(env).toContain("Bearer $LITELLM_API_KEY");

    const prompt = codingAgentPrompt("http://proxy.test", { label: "LangChain", packages: "langchain" });
    expect(prompt).toContain("base_url=http://proxy.test/v1");
    expect(prompt).toContain("opentelemetry-distro opentelemetry-exporter-otlp-proto-http langchain");
    expect(prompt).not.toMatch(/langsmith/i);
  });
});
