import { screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ApiError } from "@/lib/http/client";

import { renderWithProviders, testQueryClient } from "../../../../tests/test-utils";
import traceList from "./__fixtures__/trace_list.json";
import { AgentTracesSection } from "./AgentTracesSection";
import type { TracePage } from "./traceTypes";

vi.mock("../../networking", () => ({
  agentTraceListCall: vi.fn(),
  agentTraceCall: vi.fn(),
  agentTraceSpanCall: vi.fn(),
}));

import { agentTraceListCall } from "../../networking";

const renderSection = (view: "all" | "traces" | "llm") =>
  renderWithProviders(
    <AgentTracesSection
      view={view}
      accessToken="sk-test"
      isActive
      startTime="2026-09-29T00:00"
      endTime="2026-09-30T00:00"
      isCustomDate={false}
      isLiveTail={false}
      onOpenRequestLog={vi.fn()}
    />,
  );

describe("AgentTracesSection", () => {
  beforeEach(() => {
    testQueryClient.clear();
    vi.mocked(agentTraceListCall).mockReset();
  });

  it("renders the setup snippet when the proxy answers 501", async () => {
    vi.mocked(agentTraceListCall).mockRejectedValue(
      new ApiError("Agent tracing is not enabled", 501, { detail: "Agent tracing is not enabled" }),
    );
    renderSection("traces");

    const card = await screen.findByTestId("tracing-setup-card");
    expect(card).toHaveTextContent("Agent tracing is not enabled");
    expect(card).toHaveTextContent("store: clickhouse");
    expect(card).toHaveTextContent("CLICKHOUSE_URL=<ClickHouse writer URL>");
    expect(card).toHaveTextContent("CLICKHOUSE_READER_URL=<ClickHouse read-only URL>");
    expect(card).toHaveTextContent("LANGSMITH_TRACING_MODE=otel");
    expect(card).toHaveTextContent('OTEL_EXPORTER_OTLP_HEADERS="Authorization=Bearer <litellm key>"');
    expect(agentTraceListCall).toHaveBeenCalledTimes(1);
  });

  it("shows the tracing setup when the proxy has no traces route", async () => {
    vi.mocked(agentTraceListCall).mockRejectedValue(new ApiError("Not Found", 404, { detail: "Not Found" }));
    renderSection("traces");

    const card = await screen.findByTestId("tracing-setup-card");
    expect(card).toHaveTextContent("This proxy does not expose /v1/traces");
    expect(screen.queryByRole("table", { name: "Agent traces" })).not.toBeInTheDocument();
  });

  it("renders nothing in All when tracing is off, so request logs stay front and centre", async () => {
    vi.mocked(agentTraceListCall).mockRejectedValue(new ApiError("off", 501, { detail: "off" }));
    renderSection("all");
    await vi.waitFor(() => expect(agentTraceListCall).toHaveBeenCalled());
    expect(screen.queryByTestId("tracing-setup-card")).not.toBeInTheDocument();
    expect(screen.queryByRole("table", { name: "Agent traces" })).not.toBeInTheDocument();
  });

  it("lists traces with the agent badge, red status for runs with errors and cost", async () => {
    vi.mocked(agentTraceListCall).mockResolvedValue(traceList as TracePage);
    renderSection("traces");

    const rows = await screen.findAllByTestId("agent-trace-row");
    expect(rows).toHaveLength((traceList as TracePage).data.length);
    const lead = rows.find((row) => row.textContent?.includes("research_lead"));
    expect(lead).toHaveTextContent("◆ 6 agents · 21 LLM · 25 tool");
    expect(lead).toHaveTextContent("Success");
    expect(lead).toHaveTextContent("$0.1283");
    expect(lead).toHaveTextContent("Should we store OTEL agent spans in ClickHouse or Postgres at 50k spans/sec?");
    const failed = rows.find((row) => row.textContent?.includes("acme-404"));
    expect(failed).toHaveTextContent("Failure");
    expect(failed).toHaveTextContent("2 err");
  });

  it("does not query traces in the LLM requests view", () => {
    renderSection("llm");
    expect(agentTraceListCall).not.toHaveBeenCalled();
  });
});
