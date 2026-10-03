import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ApiError } from "@/lib/http/client";

import { chooseSelectOption, renderWithProviders, testQueryClient } from "../../../../tests/test-utils";
import traceList from "./__fixtures__/trace_list.json";
import AgentTracesPage from "./AgentTracesPage";
import { AgentTracesSection, type TimeControls } from "./AgentTracesSection";
import type { TracePage, TraceSummary } from "./traceTypes";

vi.mock("../../networking", () => ({
  apiClient: { get: vi.fn(), post: vi.fn() },
  agentTraceListCall: vi.fn(),
  sendOtlpTraceCall: vi.fn(),
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

import { agentTraceListCall, apiClient } from "../../networking";

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
      onDemo={vi.fn()}
    />,
  );

// A UTC-pinned day around the fixture runs (2026-09-30 ~06:43 UTC), so they land in the same bucket in any timezone.
const renderWindowed = (timeControls?: TimeControls) =>
  renderWithProviders(
    <AgentTracesSection
      accessToken="sk-test"
      isActive
      startTime="2026-09-30T00:00Z"
      endTime="2026-10-01T00:00Z"
      isCustomDate
      isLiveTail={false}
      timeControls={timeControls}
    />,
  );

const bucketRunCounts = () =>
  screen.getAllByTestId("timeline-bucket").map((bucket) => Number(bucket.getAttribute("data-runs")));

describe("AgentTracesSection", () => {
  afterEach(() => {
    vi.useRealTimers();
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
    vi.mocked(apiClient.get).mockResolvedValue({ data: [] });
  });

  it.each([
    [401, "Your session is no longer valid. Sign out and sign in again."],
    [403, "Your account does not have access to these traces."],
  ])("stops live polling after HTTP %s and explains how to recover", async (status, message) => {
    vi.useFakeTimers();
    vi.mocked(agentTraceListCall).mockRejectedValue(new ApiError("Private token details", Number(status), {}));
    renderWithProviders(
      <AgentTracesSection
        accessToken="sk-test"
        isActive
        startTime="2026-09-29T00:00"
        endTime="2026-09-30T00:00"
        isCustomDate={false}
        isLiveTail
      />,
    );
    await act(async () => {
      await vi.advanceTimersByTimeAsync(60_000);
    });
    expect(screen.getByText(`Could not load runs: ${message}`)).toBeVisible();
    expect(screen.queryByText(/Private token details/)).not.toBeInTheDocument();
    expect(agentTraceListCall).toHaveBeenCalledOnce();
    expect(screen.getByTestId("runs-footer")).toHaveTextContent("Update failed");
  });

  it("renders the setup snippet when the proxy answers 501", async () => {
    vi.mocked(agentTraceListCall).mockRejectedValue(
      new ApiError("Agent tracing is not enabled", 501, { detail: "Agent tracing is not enabled" }),
    );
    renderSection();

    const card = await screen.findByTestId("tracing-setup-card");
    expect(card).toHaveTextContent("Tracing is not enabled");
    expect(screen.getByRole("button", { name: "Preview sample" })).toBeVisible();
    expect(card).toHaveTextContent("type: clickhouse");
    expect(card).toHaveTextContent("url: os.environ/CLICKHOUSE_URL");
    expect(screen.getByRole("button", { name: "Check setup" })).toBeEnabled();
    expect(card).not.toHaveTextContent(/langsmith/i);
    expect(card).toHaveTextContent("ClickHouse and proxy setup");
    expect(card).toHaveTextContent("Ask your proxy administrator");
  });

  it("shows the waiting guide when tracing is on but no runs have arrived", async () => {
    vi.mocked(agentTraceListCall).mockResolvedValue({ ...(traceList as TracePage), data: [] });
    renderSection();

    const card = await screen.findByTestId("tracing-setup-card");
    expect(card).toHaveTextContent("Connect your agent");
    expect(card).toHaveTextContent("Waiting for your first trace");
    expect(screen.getByRole("button", { name: "Preview sample" })).toBeVisible();
    expect(card).not.toHaveTextContent("store: clickhouse");
  });

  it("keeps the trace list available when traces exist outside the current time window", async () => {
    vi.mocked(agentTraceListCall).mockResolvedValue({ ...(traceList as TracePage), data: [] });
    vi.mocked(apiClient.get).mockResolvedValue(traceList);
    renderSection();
    expect(await screen.findByText("No runs match these filters.")).toBeVisible();
    expect(screen.queryByTestId("tracing-setup-card")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Preview sample" })).not.toBeInTheDocument();
    expect(apiClient.get).toHaveBeenCalledWith("/v1/traces", { accessToken: "sk-test", query: { start_ms: 0 } });
  });

  it("checks proxy readiness, waits for an agent, and confirms receipt using actual query results", async () => {
    vi.mocked(agentTraceListCall).mockRejectedValue(new ApiError("Tracing is not enabled", 501, {}));
    renderSection();
    const checkSetup = await screen.findByRole("button", { name: "Check setup" });
    vi.mocked(agentTraceListCall).mockResolvedValue({ ...(traceList as TracePage), data: [] });
    fireEvent.click(checkSetup);
    expect(await screen.findByRole("heading", { name: "Connect your agent" })).toBeVisible();
    expect(screen.getByText("Waiting for your first trace")).toBeVisible();
    vi.mocked(agentTraceListCall).mockResolvedValue(traceList as TracePage);
    fireEvent.click(screen.getByRole("button", { name: "Check for traces" }));
    expect(await screen.findAllByTestId("agent-trace-row")).toHaveLength(runs.length);
    expect(screen.getByText("Traces received. Select a run to inspect it.")).toBeVisible();
    expect(screen.queryByRole("button", { name: "Preview sample" })).not.toBeInTheDocument();
    expect(screen.queryByTestId("tracing-setup-card")).not.toBeInTheDocument();
  });

  it("keeps setup visible while checking and explains when tracing is still disabled", async () => {
    const failure = new ApiError("Tracing is not enabled", 501, {});
    vi.mocked(agentTraceListCall).mockRejectedValue(failure);
    renderSection();
    const checkSetup = await screen.findByRole("button", { name: "Check setup" });
    let rejectCheck: (error: Error) => void = () => {};
    vi.mocked(agentTraceListCall).mockImplementationOnce(
      () =>
        new Promise((_, reject) => {
          rejectCheck = reject;
        }),
    );
    fireEvent.click(checkSetup);
    expect(await screen.findByRole("button", { name: "Checking…" })).toBeDisabled();
    expect(screen.getByRole("heading", { name: "Enable tracing" })).toBeVisible();
    await act(async () => rejectCheck(failure));
    expect(await screen.findByText(/Tracing is still unavailable/)).toBeVisible();
    expect(screen.getByRole("button", { name: "Check setup" })).toBeEnabled();
  });

  it("keeps received traces and the open drawer visible during subsequent fetches", async () => {
    vi.mocked(agentTraceListCall).mockRejectedValue(new ApiError("Tracing is not enabled", 501, {}));
    renderSection();
    const checkSetup = await screen.findByRole("button", { name: "Check setup" });
    vi.mocked(agentTraceListCall).mockResolvedValue(traceList as TracePage);
    fireEvent.click(checkSetup);
    const rows = await screen.findAllByTestId("agent-trace-row");
    fireEvent.click(rows[0]);
    const drawer = screen.getByRole("complementary", { name: "Trace details" });

    let finishRefresh: (page: TracePage) => void = () => {};
    vi.mocked(agentTraceListCall).mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          finishRefresh = resolve;
        }),
    );
    await act(async () => {
      void testQueryClient.invalidateQueries({ queryKey: ["agentTraces"] });
    });
    expect(await screen.findByText("Updating…")).toBeVisible();
    expect(screen.getAllByTestId("agent-trace-row")).toHaveLength(runs.length);
    expect(screen.getByRole("complementary", { name: "Trace details" })).toBe(drawer);
    expect(screen.queryByTestId("tracing-setup-card")).not.toBeInTheDocument();
    await act(async () => finishRefresh(traceList as TracePage));
  });

  it("separates a failed history check from the empty list and retries that check", async () => {
    vi.mocked(agentTraceListCall).mockResolvedValue({ ...(traceList as TracePage), data: [] });
    vi.mocked(apiClient.get).mockRejectedValue(new ApiError("History unavailable", 503, {}));
    renderSection();
    expect(await screen.findByRole("alert")).toHaveTextContent("Could not check earlier traces. History unavailable");
    expect(screen.getByText("No runs match these filters.")).toBeVisible();
    expect(screen.queryByText(/Could not load runs/)).not.toBeInTheDocument();

    vi.mocked(apiClient.get).mockResolvedValue(traceList);
    fireEvent.click(screen.getByRole("button", { name: "Retry trace check" }));
    await waitFor(() => expect(screen.queryByRole("alert")).not.toBeInTheDocument());
    expect(screen.getByText("No runs match these filters.")).toBeVisible();
    expect(agentTraceListCall).toHaveBeenCalledTimes(1);
  });

  it("treats a proxy without the trace routes (404) like tracing being off", async () => {
    vi.mocked(agentTraceListCall).mockRejectedValue(new ApiError("Not Found", 404, { detail: "Not Found" }));
    renderSection();

    const card = await screen.findByTestId("tracing-setup-card");
    expect(card).toHaveTextContent("Tracing is not enabled");
    expect(card).toHaveTextContent("url: os.environ/CLICKHOUSE_URL");
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
    expect(screen.getByRole("columnheader", { name: "Cost" })).toBeInTheDocument();
    expect(within(failed).getByText("—")).toBeInTheDocument();
  });

  it("shows the spend returned for a run", async () => {
    vi.mocked(agentTraceListCall).mockResolvedValue({
      ...(traceList as TracePage),
      data: [{ ...runs[0], spend: 0.025 }],
    });
    renderSection();

    const row = await screen.findByTestId("agent-trace-row");
    expect(within(row).getByText("$0.03")).toBeInTheDocument();
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

  it("uses recorded agent names for the column and filter even when services are shared", async () => {
    vi.mocked(agentTraceListCall).mockResolvedValue({
      ...(traceList as TracePage),
      data: [
        ...runs.slice(1).map((run) => ({ ...run, service: "shared-app", agent_names: ["research-agent"] })),
        { ...runs[0], service: "shared-app", agent_names: ["billing-agent", "review-agent"] },
      ],
    });
    const user = userEvent.setup();
    renderSection();
    await screen.findAllByTestId("agent-trace-row");

    expect(screen.getByRole("columnheader", { name: "Agent" })).toBeInTheDocument();
    const agentFilter = screen.getByRole("combobox", { name: "Filter by agent" });
    expect(agentFilter).toHaveTextContent("All agents");

    await chooseSelectOption(user, agentFilter, "billing-agent");
    const rows = screen.getAllByTestId("agent-trace-row");
    expect(rows).toHaveLength(1);
    expect(rows[0]).toHaveTextContent("billing-agent");
    expect(rows[0]).not.toHaveTextContent("shared-app");

    await chooseSelectOption(user, agentFilter, "review-agent");
    expect(screen.getAllByTestId("agent-trace-row")).toHaveLength(1);

    await chooseSelectOption(user, agentFilter, "All agents");
    expect(screen.getAllByTestId("agent-trace-row")).toHaveLength(runs.length);
  });

  it("shows each run's agent name with the logo of the SDK that produced it", async () => {
    vi.mocked(agentTraceListCall).mockResolvedValue({
      ...(traceList as TracePage),
      data: [
        { ...runs[0], agent_names: ["research-bot"], frameworks: ["claude-agent-sdk", "claude-code"] },
        { ...runs[1], agent_names: [], frameworks: ["claude-code"] },
        { ...runs[2], frameworks: [] },
      ],
    });
    renderSection();
    const [sdkRun, cliRun, plainRun] = await screen.findAllByTestId("agent-trace-row");
    const agentCell = (row: HTMLElement) => within(row).getAllByRole("cell")[1];

    expect(agentCell(sdkRun)).toHaveTextContent(/^research-bot$/);
    expect(agentCell(sdkRun)).toHaveAttribute("title", "research-bot · Claude Agent SDK");
    expect(within(sdkRun).getByRole("img", { name: "Claude Agent SDK logo", hidden: true })).toHaveAttribute(
      "src",
      expect.stringContaining("anthropic.svg"),
    );
    expect(agentCell(cliRun)).toHaveTextContent(/^Claude Code$/);
    expect(within(plainRun).queryByRole("img", { hidden: true })).not.toBeInTheDocument();
    expect(agentCell(plainRun)).toHaveTextContent((runs[2].agent_names ?? [runs[2].service]).join(", "));
  });

  it("opens a run in a side drawer over the list and swaps runs without closing it", async () => {
    vi.mocked(agentTraceListCall).mockResolvedValue(traceList as TracePage);
    renderSection();
    const rows = await screen.findAllByTestId("agent-trace-row");

    fireEvent.click(rows[0]);
    const drawer = screen.getByRole("complementary", { name: "Trace details" });
    expect(within(drawer).getByTestId("run-view")).toHaveTextContent(`run ${runs[0].trace_id}`);
    expect(screen.getByTestId("runs-table")).toBeInTheDocument();
    expect(rows[0]).toHaveAttribute("aria-selected", "true");

    fireEvent.click(rows[1]);
    expect(screen.getByRole("complementary", { name: "Trace details" })).toBe(drawer);
    expect(within(drawer).getByTestId("run-view")).toHaveTextContent(`run ${runs[1].trace_id}`);
    expect(rows[1]).toHaveAttribute("aria-selected", "true");
    expect(rows[0]).toHaveAttribute("aria-selected", "false");
  });

  it("closes the drawer when the open row is clicked again or Escape is pressed", async () => {
    vi.mocked(agentTraceListCall).mockResolvedValue(traceList as TracePage);
    renderSection();
    const rows = await screen.findAllByTestId("agent-trace-row");

    fireEvent.click(rows[0]);
    fireEvent.click(rows[0]);
    expect(rows[0]).toHaveAttribute("aria-selected", "false");

    fireEvent.click(rows[1]);
    fireEvent.keyDown(window, { key: "Escape" });
    expect(rows[1]).toHaveAttribute("aria-selected", "false");
  });

  it("moves to the next and previous run with j / k and the header arrows", async () => {
    vi.mocked(agentTraceListCall).mockResolvedValue(traceList as TracePage);
    renderSection();
    const rows = await screen.findAllByTestId("agent-trace-row");

    fireEvent.click(rows[0]);
    fireEvent.keyDown(window, { key: "j" });
    expect(screen.getByTestId("run-view")).toHaveTextContent(`run ${runs[1].trace_id}`);
    fireEvent.keyDown(window, { key: "k" });
    expect(screen.getByTestId("run-view")).toHaveTextContent(`run ${runs[0].trace_id}`);
    expect(screen.getByRole("button", { name: "Previous trace (K)" })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "Next trace (J)" }));
    expect(screen.getByTestId("run-view")).toHaveTextContent(`run ${runs[1].trace_id}`);
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
    // The pan below moves a [0, first] bracket to the far right; it must end up clear of every run.
    expect(first).toBeGreaterThan(1);
    expect(first).toBeLessThan(30);

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

  it("clears timeline zoom when refreshed", async () => {
    vi.mocked(agentTraceListCall).mockResolvedValue(traceList as TracePage);
    renderWindowed({ rangeHours: 24, onRangeHoursChange: () => {}, onLiveChange: () => {} });
    await screen.findAllByTestId("agent-trace-row");
    const area = screen.getByTestId("timeline-area");
    const x = (bucket: number) => bucket * 10 + 5;

    fireEvent.pointerDown(area, { clientX: x(0), pointerId: 1 });
    fireEvent.pointerMove(area, { clientX: x(1), pointerId: 1 });
    fireEvent.pointerUp(area, { clientX: x(1), pointerId: 1 });
    expect(screen.getByTestId("timeline-selection")).toBeInTheDocument();
    expect(screen.queryAllByTestId("agent-trace-row")).toHaveLength(0);

    fireEvent.click(screen.getByRole("button", { name: "Refresh" }));

    expect(screen.queryByTestId("timeline-selection")).not.toBeInTheDocument();
    expect(screen.getAllByTestId("agent-trace-row")).toHaveLength(runs.length);
  });
});

describe("AgentTracesPage", () => {
  beforeEach(() => {
    testQueryClient.clear();
    vi.mocked(agentTraceListCall).mockReset();
    vi.mocked(apiClient.get).mockResolvedValue({ data: [] });
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
    expect(screen.getByRole("button", { name: "Refresh" })).toBeEnabled();
  });

  it("refreshes the trace list", async () => {
    vi.mocked(agentTraceListCall).mockResolvedValue(traceList as TracePage);
    renderWithProviders(<AgentTracesPage accessToken="sk-test" />);
    await screen.findByTestId("runs-table");

    const callsBeforeRefresh = vi.mocked(agentTraceListCall).mock.calls.length;
    fireEvent.click(screen.getByRole("button", { name: "Refresh" }));

    await waitFor(() => expect(vi.mocked(agentTraceListCall).mock.calls.length).toBeGreaterThan(callsBeforeRefresh));
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
