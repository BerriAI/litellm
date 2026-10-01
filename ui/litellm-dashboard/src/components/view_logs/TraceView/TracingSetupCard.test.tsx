import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { chooseSelectOption } from "@/../tests/test-utils";
import { copyToClipboard } from "@/utils/dataUtils";
import { codingAgentPrompt, tracingEnvSnippet, TracingSetupCard } from "./TracingSetupCard";

vi.mock("../../networking", () => ({ getProxyBaseUrl: () => "http://proxy.test/" }));
vi.mock("@/utils/dataUtils", () => ({ copyToClipboard: vi.fn().mockResolvedValue(true) }));

describe("TracingSetupCard", () => {
  it("shows agent connection guidance once tracing is enabled without presenting example data as live", async () => {
    const user = userEvent.setup();
    render(<TracingSetupCard detail={null} />);
    expect(screen.getByRole("heading", { name: "Connect your agent" })).toBeVisible();
    expect(screen.getByText("Tracing enabled")).toBeVisible();
    expect(screen.getByText("Waiting for your first trace")).toBeVisible();
    expect(screen.getByRole("img", { hidden: true, name: /Example agent trace/ })).not.toBeVisible();
    expect(screen.getByTestId("tracing-setup-card")).not.toHaveTextContent("store: clickhouse");
    await user.click(screen.getByText("Set up manually"));
    expect(screen.getByText(/OTEL_EXPORTER_OTLP_ENDPOINT=http:\/\/proxy.test/)).toBeVisible();
    expect(screen.getByTestId("tracing-setup-card")).not.toHaveTextContent(/langsmith/i);
  });

  it("copies instructions for the selected framework and preserves both manual installers", async () => {
    const user = userEvent.setup();
    render(<TracingSetupCard detail={null} />);
    await chooseSelectOption(user, screen.getByRole("combobox", { name: "Your agent framework" }), "CrewAI");
    await user.click(screen.getByRole("button", { name: "Copy setup prompt" }));
    expect(copyToClipboard).toHaveBeenLastCalledWith(
      codingAgentPrompt("http://proxy.test", {
        label: "CrewAI",
        packages: "crewai openinference-instrumentation-crewai",
      }),
    );
    expect(screen.getByRole("button", { name: "Prompt copied" })).toBeVisible();
    await user.click(screen.getByText("Set up manually"));
    expect(screen.getByText(/pip install -U opentelemetry-distro/)).toHaveTextContent(
      "crewai openinference-instrumentation-crewai",
    );
    await user.click(screen.getByRole("tab", { name: "uv" }));
    expect(screen.getByText(/uv add opentelemetry-distro/)).toHaveTextContent(
      "crewai openinference-instrumentation-crewai",
    );
  });

  it("guides proxy setup before agent setup and allows checking readiness", async () => {
    const user = userEvent.setup();
    const onCheck = vi.fn();
    render(<TracingSetupCard detail="Agent tracing is not enabled" onCheck={onCheck} />);
    expect(screen.getByRole("heading", { name: "Enable tracing" })).toBeVisible();
    expect(screen.getByTestId("tracing-setup-card")).toHaveTextContent("store: clickhouse");
    expect(screen.queryByRole("combobox", { name: "Your agent framework" })).not.toBeInTheDocument();
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
    expect(env).toContain("Bearer $LITELLM_API_KEY");
    const prompt = codingAgentPrompt("http://proxy.test", { label: "LangChain", packages: "langchain" });
    expect(prompt).toContain("base_url=http://proxy.test/v1");
    expect(prompt).toContain("opentelemetry-distro opentelemetry-exporter-otlp-proto-http langchain");
    expect(prompt).toContain("OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf");
    expect(prompt).toContain("Lens > Traces");
    expect(prompt).not.toMatch(/langsmith/i);
  });
});
