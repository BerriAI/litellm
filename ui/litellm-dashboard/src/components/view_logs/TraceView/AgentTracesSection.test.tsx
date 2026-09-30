import { fireEvent, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ApiError } from "@/lib/http/client";

import { renderWithProviders, testQueryClient } from "../../../../tests/test-utils";
import traceList from "./__fixtures__/trace_list.json";
import AgentTracesPage from "./AgentTracesPage";
import { AgentTracesSection, filterRuns } from "./AgentTracesSection";
import type { TracePage, TraceSummary } from "./traceTypes";

vi.mock("../../networking", () => ({
  agentTraceListCall: vi.fn(),
  agentTraceCall: vi.fn(),
  agentTraceSpanCall: vi.fn(),
  getProxyBaseUrl: () => "http://localhost:4000",
}));

vi.mock("./TraceDrawer", () => ({
  RunView: ({ traceId, onBack }: { traceId: string; onBack: () => void }) => (
    <div data-testid="run-view">
      run {traceId}
      <button onClick={onBack}>back</button>
    </div>
  ),
}));

import { agentTraceListCall } from "../../networking";

const runs = (traceList as TracePage).data as TraceSummary[];

const renderSection = () =>
  renderWithProviders(
    <AgentTracesSection
      accessToken="sk-test"
      isActive
      startTime="2026-09-29T00:00"
      endTime="2026-09-30T00:00"
      isCustomDate={false}
      isLiveTail={false}
    />,
  );

// A window wide enough to hold the fixture runs (2026-09-30 ~06:43 UTC) in any local timezone.
const renderWindowed = () =>
  renderWithProviders(
    <AgentTracesSection
      accessToken="sk-test"
      isActive
      startTime="2026-09-29T00:00"
      endTime="2026-10-01T00:00"
      isCustomDate
      isLiveTail={false}
    />,
  );

const bucketRunCounts = () =>
  screen.getAllByTestId("timeline-bucket").map((bucket) => Number(bucket.getAttribute("data-runs")));

describe("AgentTracesSection", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  beforeEach(() => {
    vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockReturnValue({
      left: 0,
      width: 600,
      top: 0,
      height: 56,
      right: 600,
      bottom: 56,
      x: 0,
      y: 0,
      toJSON: () => ({}),
    } as DOMRect);
    testQueryClient.clear();
    vi.mocked(agentTraceListCall).mockReset();
  });

  it("renders the setup snippet when the proxy answers 501", async () => {
    vi.mocked(agentTraceListCall).mockRejectedValue(
      new ApiError("Agent tracing is not enabled", 501, { detail: "Agent tracing is not enabled" }),
    );
    renderSection();

    const card = await screen.findByTestId("tracing-setup-card");
    expect(card).toHaveTextContent("Tracing is not enabled");
    expect(card).toHaveTextContent("store: clickhouse");
    expect(card).toHaveTextContent("OTEL_EXPORTER_OTLP_ENDPOINT=");
    expect(card).not.toHaveTextContent(/langsmith/i);
    expect(card).toHaveTextContent('OTEL_EXPORTER_OTLP_HEADERS="Authorization=Bearer $LITELLM_API_KEY"');
    expect(card).toHaveTextContent("Let Claude Code or Codex set it up");
  });

  it("shows the waiting guide when tracing is on but no runs have arrived", async () => {
    vi.mocked(agentTraceListCall).mockResolvedValue({ ...(traceList as TracePage), data: [] });
    renderSection();

    const card = await screen.findByTestId("tracing-setup-card");
    expect(card).toHaveTextContent("Waiting for traces");
    expect(card).toHaveTextContent("No traces detected yet");
    expect(card).not.toHaveTextContent("store: clickhouse");
  });

  it("treats a proxy without the trace routes (404) like tracing being off", async () => {
    vi.mocked(agentTraceListCall).mockRejectedValue(new ApiError("Not Found", 404, { detail: "Not Found" }));
    renderSection();

    const card = await screen.findByTestId("tracing-setup-card");
    expect(card).toHaveTextContent("Tracing is not enabled");
    expect(card).toHaveTextContent("CLICKHOUSE_READER_URL");
  });

  it("lists every run with its input, counts and failed column", async () => {
    vi.mocked(agentTraceListCall).mockResolvedValue(traceList as TracePage);
    renderSection();

    const rows = await screen.findAllByTestId("agent-trace-row");
    expect(rows).toHaveLength(runs.length);
    const lead = rows.find((row) => row.textContent?.includes("Should we store OTEL agent spans"));
    expect(lead).toBeDefined();
    const failed = rows.find((row) => row.textContent?.includes("acme-404")) as HTMLElement;
    expect(within(failed).getByLabelText("2 errors")).toBeInTheDocument();
    expect(screen.getByText(`${runs.length} runs`)).toBeInTheDocument();
    expect(screen.queryByRole("columnheader", { name: "Cost" })).not.toBeInTheDocument();
  });

  it("filters by input text and by trace id", async () => {
    vi.mocked(agentTraceListCall).mockResolvedValue(traceList as TracePage);
    renderSection();
    await screen.findAllByTestId("agent-trace-row");

    const search = screen.getByLabelText("Search runs");
    fireEvent.change(search, { target: { value: "acme-404" } });
    expect(screen.getAllByTestId("agent-trace-row")).toHaveLength(1);

    const lead = runs.find((r) => r.name === "research_lead") as TraceSummary;
    fireEvent.change(search, { target: { value: lead.trace_id.slice(0, 10) } });
    const rows = screen.getAllByTestId("agent-trace-row");
    expect(rows).toHaveLength(1);
    expect(rows[0]).toHaveTextContent("Should we store OTEL agent spans");
  });

  it("status filter 'Failed' keeps only runs with errors", () => {
    const failed = filterRuns(runs, "", "all", "error");
    expect(failed.length).toBeGreaterThan(0);
    expect(failed.every((r) => r.error_count > 0)).toBe(true);
    const ok = filterRuns(runs, "", "all", "ok");
    expect(ok.every((r) => r.error_count === 0)).toBe(true);
    expect(failed.length + ok.length).toBe(runs.length);
  });

  it("opens the run in place and goes back to the list", async () => {
    vi.mocked(agentTraceListCall).mockResolvedValue(traceList as TracePage);
    renderSection();
    const rows = await screen.findAllByTestId("agent-trace-row");

    fireEvent.click(rows[0]);
    expect(screen.getByTestId("run-view")).toHaveTextContent(`run ${runs[0].trace_id}`);
    expect(screen.queryByTestId("runs-table")).not.toBeInTheDocument();

    fireEvent.click(screen.getByText("back"));
    expect(screen.getByTestId("runs-table")).toBeInTheDocument();
  });

  it("plots every loaded run on the timeline", async () => {
    vi.mocked(agentTraceListCall).mockResolvedValue(traceList as TracePage);
    renderWindowed();
    await screen.findAllByTestId("agent-trace-row");

    expect(screen.getByTestId("traces-timeline")).toBeInTheDocument();
    const counts = bucketRunCounts();
    expect(counts).toHaveLength(60);
    expect(counts.reduce((a, b) => a + b, 0)).toBe(runs.length);
  });

  it("zooms by dragging, resizes and pans the bracket, and clears with Esc", async () => {
    vi.mocked(agentTraceListCall).mockResolvedValue(traceList as TracePage);
    renderWindowed();
    await screen.findAllByTestId("agent-trace-row");
    const area = screen.getByTestId("timeline-area");
    const x = (bucket: number) => bucket * 10 + 5;
    const drag = (target: HTMLElement, from: number, to: number) => {
      fireEvent.pointerDown(target, { clientX: x(from), pointerId: 1 });
      fireEvent.pointerMove(area, { clientX: x(to), pointerId: 1 });
      fireEvent.pointerUp(area, { clientX: x(to), pointerId: 1 });
    };
    const rowCount = () => screen.queryAllByTestId("agent-trace-row").length;
    const withRuns = bucketRunCounts().flatMap((count, i) => (count > 0 ? [i] : []));
    const first = withRuns[0];

    drag(area, 0, 1);
    expect(screen.getByTestId("timeline-selection")).toBeInTheDocument();
    expect(rowCount()).toBe(0);

    drag(screen.getByTestId("timeline-handle-hi"), 1, first);
    expect(rowCount()).toBeGreaterThan(0);

    drag(screen.getByTestId("timeline-selection"), 1, 1 - first);
    expect(rowCount()).toBeGreaterThan(0);
    drag(screen.getByTestId("timeline-selection"), 0, 59);
    expect(rowCount()).toBe(0);

    fireEvent.keyDown(screen.getByTestId("traces-timeline"), { key: "Escape" });
    expect(screen.queryByTestId("timeline-selection")).not.toBeInTheDocument();
    expect(rowCount()).toBe(runs.length);
  });
});

describe("AgentTracesPage", () => {
  beforeEach(() => {
    testQueryClient.clear();
    vi.mocked(agentTraceListCall).mockReset();
  });

  it("shows the actual range, switches presets from the popover, and toggles Live", async () => {
    vi.mocked(agentTraceListCall).mockResolvedValue(traceList as TracePage);
    renderWithProviders(<AgentTracesPage accessToken="sk-test" />);
    await screen.findByTestId("runs-table");

    const trigger = screen.getByRole("button", { name: "Time range" });
    expect(trigger).toHaveTextContent(/ to /);
    expect(screen.getByTestId("traces-timeline")).toHaveTextContent("Total 1d");

    fireEvent.click(trigger);
    fireEvent.click(await screen.findByRole("menuitemradio", { name: "Last 7 days" }));
    expect(await screen.findByText("Total 7d")).toBeInTheDocument();
    const last = vi.mocked(agentTraceListCall).mock.calls.at(-1)?.[0];
    expect((last?.endMs ?? 0) - (last?.startMs ?? 0)).toBeGreaterThanOrEqual(7 * 24 * 3600 * 1000 - 60_000);

    const live = screen.getByRole("button", { name: "Live" });
    expect(live).toHaveAttribute("aria-pressed", "true");
    fireEvent.click(live);
    expect(live).toHaveAttribute("aria-pressed", "false");
    expect(screen.getByRole("button", { name: "Reset zoom" })).toBeDisabled();
  });

  it("keeps the time controls on an empty range the user picked, instead of showing onboarding", async () => {
    vi.mocked(agentTraceListCall).mockResolvedValue(traceList as TracePage);
    renderWithProviders(<AgentTracesPage accessToken="sk-test" />);
    await screen.findByTestId("runs-table");

    vi.mocked(agentTraceListCall).mockResolvedValue({ ...(traceList as TracePage), data: [] });
    fireEvent.click(screen.getByRole("button", { name: "Time range" }));
    fireEvent.click(await screen.findByRole("menuitemradio", { name: "Last hour" }));

    expect(await screen.findByText("No runs match these filters.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Time range" })).toBeInTheDocument();
    expect(screen.queryByTestId("tracing-setup-card")).not.toBeInTheDocument();
  });

  it("asks the proxy for the last 24 hours by default", async () => {
    vi.mocked(agentTraceListCall).mockResolvedValue(traceList as TracePage);
    renderWithProviders(<AgentTracesPage accessToken="sk-test" />);
    await screen.findByTestId("runs-table");

    const { startMs, endMs } = vi.mocked(agentTraceListCall).mock.calls[0][0];
    expect(endMs - startMs).toBeGreaterThanOrEqual(24 * 3600 * 1000 - 60_000);
    expect(endMs - startMs).toBeLessThan(24 * 3600 * 1000 + 120_000);
  });
});
