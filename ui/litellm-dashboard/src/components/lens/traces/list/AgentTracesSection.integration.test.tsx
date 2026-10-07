import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { mockAllIsIntersecting, setupIntersectionMocking } from "react-intersection-observer/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ApiError } from "@/lib/http/client";

import { renderWithProviders, testQueryClient } from "../../../../../tests/test-utils";
import traceList from "../__fixtures__/trace_list.json";
import AgentTracesPage from "./AgentTracesPage";
import { filterRuns } from "./runSearch/runQuery";
import type { RelativeRangeState } from "@/components/shared/timeRange/useRelativeRange";

import { AgentTracesSection } from "./AgentTracesSection";
import type { TraceFindingCount, TracePage, TraceSummary } from "../types";

vi.mock("../../../networking", () => ({
  apiClient: { get: vi.fn(), post: vi.fn() },
  agentTraceListCall: vi.fn(),
  sendOtlpTraceCall: vi.fn(),
  agentTraceCall: vi.fn(),
  agentTraceSpanCall: vi.fn(),
  getProxyBaseUrl: () => "http://localhost:4000",
}));

vi.mock("../detail/run/RunView", () => ({
  RunView: ({ traceId, onBack }: { traceId: string; onBack: () => void }) => (
    <div data-testid="run-view">
      run {traceId}
      <button onClick={onBack}>back</button>
    </div>
  ),
}));

import { agentTraceListCall, apiClient } from "../../../networking";

const runs = (traceList as TracePage).data as TraceSummary[];

const lastUrl = (onUrlUpdate: ReturnType<typeof vi.fn>) =>
  new URLSearchParams(String(onUrlUpdate.mock.lastCall?.[0].queryString ?? ""));

const ROLLING_DAY = { hours: 24, anchorMs: null };
const PINNED_DAY = { hours: 24, anchorMs: Date.parse("2026-10-01T00:00Z") };

const renderSection = (canViewFindings = true) =>
  renderWithProviders(
    <AgentTracesSection accessToken="sk-test" isActive range={ROLLING_DAY} canViewFindings={canViewFindings} />,
  );

// A UTC-pinned day around the fixture runs (2026-09-30 ~06:43 UTC), so they land in the same bucket in any timezone.
const renderWindowed = (timeControls?: RelativeRangeState) =>
  renderWithProviders(
    <AgentTracesSection accessToken="sk-test" isActive range={PINNED_DAY} timeControls={timeControls} />,
  );

const bucketRunCounts = () =>
  screen.getAllByTestId("timeline-bucket").map((bucket) => Number(bucket.getAttribute("data-total")));

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
    setupIntersectionMocking(vi.fn);
    testQueryClient.clear();
    vi.mocked(agentTraceListCall).mockReset();
    vi.mocked(apiClient.get).mockResolvedValue({ data: [] });
    vi.mocked(apiClient.post).mockImplementation(async (_path, options) => {
      const body = options?.body as { traces: { trace_id: string; trace_ref?: string }[] };
      return body.traces.map((trace) => ({ ...trace, finding_count: null }));
    });
  });

  it("loads the next page only once the list scrolls near its end, then stops at the last page", async () => {
    vi.mocked(agentTraceListCall)
      .mockResolvedValueOnce({ data: runs.slice(0, 1), next_cursor: "next" })
      .mockResolvedValueOnce({ data: runs.slice(1, 2), next_cursor: null });
    renderSection();
    expect(await screen.findByTestId("agent-trace-row")).toBeVisible();
    expect(screen.queryByRole("button", { name: "Load more" })).not.toBeInTheDocument();
    act(() => mockAllIsIntersecting(false));
    expect(agentTraceListCall).toHaveBeenCalledOnce();
    act(() => mockAllIsIntersecting(true));
    await waitFor(() => expect(screen.getAllByTestId("agent-trace-row")).toHaveLength(2));
    expect(vi.mocked(agentTraceListCall).mock.calls[1][0]).toMatchObject({ cursor: "next" });
    expect(screen.queryByTestId("runs-placeholder")).not.toBeInTheDocument();
    expect(agentTraceListCall).toHaveBeenCalledTimes(2);
  });

  it("waits for the user before paging when filters hide every loaded run", async () => {
    const user = userEvent.setup();
    vi.mocked(agentTraceListCall)
      .mockResolvedValueOnce({ data: runs.slice(0, 1), next_cursor: "next" })
      .mockResolvedValueOnce({ data: runs.slice(1, 2), next_cursor: "later" });
    renderSection();
    expect(await screen.findByTestId("agent-trace-row")).toBeVisible();
    await user.type(screen.getByRole("combobox", { name: "Search runs" }), "no-such-run");
    await waitFor(() => expect(screen.queryAllByTestId("agent-trace-row")).toHaveLength(0));
    act(() => mockAllIsIntersecting(true));
    expect(screen.queryByTestId("runs-placeholder")).not.toBeInTheDocument();
    expect(agentTraceListCall).toHaveBeenCalledOnce();
    await user.click(screen.getByRole("button", { name: "Load older runs" }));
    await waitFor(() => expect(agentTraceListCall).toHaveBeenCalledTimes(2));
    act(() => mockAllIsIntersecting(true));
    expect(agentTraceListCall).toHaveBeenCalledTimes(2);
  });

  it("keeps the original time window and loaded rows when another page fails", async () => {
    const user = userEvent.setup();
    const now = vi.spyOn(Date, "now").mockReturnValue(Date.parse("2026-10-01T00:00Z"));
    vi.mocked(agentTraceListCall).mockResolvedValueOnce({ data: runs.slice(0, 1), next_cursor: "next" });
    renderSection();
    expect(await screen.findByTestId("agent-trace-row")).toBeVisible();
    const first = vi.mocked(agentTraceListCall).mock.calls[0][0];
    now.mockReturnValue(Date.parse("2026-10-01T01:00Z"));
    vi.mocked(agentTraceListCall).mockRejectedValue(new ApiError("Please try again", 403, {}));
    act(() => mockAllIsIntersecting(true));
    expect(await screen.findByRole("alert")).toHaveTextContent("Could not load more runs");
    expect(screen.getAllByTestId("agent-trace-row")).toHaveLength(1);
    expect(screen.queryByTestId("runs-placeholder")).not.toBeInTheDocument();
    expect(agentTraceListCall).toHaveBeenCalledTimes(2);
    expect(vi.mocked(agentTraceListCall).mock.calls[1][0]).toEqual({ ...first, cursor: "next" });
    vi.mocked(agentTraceListCall).mockResolvedValueOnce({ data: runs.slice(1, 2), next_cursor: null });
    await user.click(screen.getByRole("button", { name: "Retry" }));
    await waitFor(() => expect(screen.getAllByTestId("agent-trace-row")).toHaveLength(2));
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it.each([
    [401, "Your session is no longer valid. Sign out and sign in again."],
    [403, "Your account does not have access to these traces."],
  ])("stops live polling after HTTP %s and explains how to recover", async (status, message) => {
    vi.useFakeTimers();
    vi.mocked(agentTraceListCall).mockRejectedValue(new ApiError("Private token details", Number(status), {}));
    renderWithProviders(<AgentTracesSection accessToken="sk-test" isActive range={ROLLING_DAY} />);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(60_000);
    });
    expect(screen.getByText(`Could not load runs: ${message}`)).toBeVisible();
    expect(screen.queryByText(/Private token details/)).not.toBeInTheDocument();
    expect(agentTraceListCall).toHaveBeenCalledOnce();
  });

  it("renders the setup snippet when the proxy answers 501", async () => {
    vi.mocked(agentTraceListCall).mockRejectedValue(
      new ApiError("Agent tracing is not enabled", 501, { detail: "Agent tracing is not enabled" }),
    );
    renderSection();

    const card = await screen.findByTestId("tracing-setup-card");
    expect(card).toHaveTextContent("Tracing is not enabled");
    expect(screen.queryByRole("button", { name: "Preview sample" })).not.toBeInTheDocument();
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
    expect(screen.queryByRole("button", { name: "Preview sample" })).not.toBeInTheDocument();
    expect(card).not.toHaveTextContent("store: clickhouse");
  });

  it("keeps the trace list available when traces exist outside the current time window", async () => {
    vi.mocked(agentTraceListCall).mockResolvedValue({ ...(traceList as TracePage), data: [] });
    vi.mocked(apiClient.get).mockResolvedValue(traceList);
    renderSection();
    expect(await screen.findByText("No runs in this time range")).toBeVisible();
    expect(screen.queryByTestId("tracing-setup-card")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Preview sample" })).not.toBeInTheDocument();
    expect(apiClient.get).toHaveBeenCalledWith("/v1/traces", { accessToken: "sk-test", query: { start_ms: 0 } });
    fireEvent.click(screen.getByRole("button", { name: "Set up tracing" }));
    expect(await screen.findByRole("heading", { name: "Connect another agent" })).toBeVisible();
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
    await waitFor(() => expect(screen.getByRole("table", { name: "Agent runs" })).toHaveAttribute("aria-busy", "true"));
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
    expect(screen.getByText("No runs in this time range")).toBeVisible();
    expect(screen.queryByText(/Could not load runs/)).not.toBeInTheDocument();

    vi.mocked(apiClient.get).mockResolvedValue(traceList);
    fireEvent.click(screen.getByRole("button", { name: "Retry trace check" }));
    await waitFor(() => expect(screen.queryByRole("alert")).not.toBeInTheDocument());
    expect(screen.getByText("No runs in this time range")).toBeVisible();
    expect(agentTraceListCall).toHaveBeenCalledTimes(1);
  });

  it("treats a proxy without the trace routes (404) like tracing being off", async () => {
    vi.mocked(agentTraceListCall).mockRejectedValue(new ApiError("Not Found", 404, { detail: "Not Found" }));
    renderSection();

    const card = await screen.findByTestId("tracing-setup-card");
    expect(card).toHaveTextContent("Tracing is not enabled");
    expect(card).toHaveTextContent("url: os.environ/CLICKHOUSE_URL");
  });

  it("lists uninvestigated runs without presenting tool errors as failures", async () => {
    vi.mocked(agentTraceListCall).mockResolvedValue(traceList as TracePage);
    renderSection();

    const rows = await screen.findAllByTestId("agent-trace-row");
    expect(rows).toHaveLength(runs.length);
    const lead = rows.find((row) => row.textContent?.includes("Should we store OTEL agent spans"));
    expect(lead).toBeDefined();
    const failed = rows.find((row) => row.textContent?.includes("acme-404")) as HTMLElement;
    expect(within(failed).queryByLabelText("2 errors")).not.toBeInTheDocument();
    expect(await within(failed).findByTitle("No conclusive investigation for this trace")).toHaveTextContent("-");
    expect(screen.getByRole("columnheader", { name: "Findings" })).toBeInTheDocument();
    expect(screen.queryByRole("columnheader", { name: "Failed" })).not.toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "Cost" })).toBeInTheDocument();
    expect(within(failed).getByText("—")).toBeInTheDocument();
  });

  it("distinguishes uninvestigated traces, completed clean investigations, and findings", async () => {
    vi.mocked(agentTraceListCall).mockResolvedValue({ data: runs, next_cursor: null });
    vi.mocked(apiClient.post).mockResolvedValue(
      runs.map((run, index) => ({
        trace_id: run.trace_id,
        trace_ref: run.trace_ref ?? "",
        finding_count: [null, 0, 3][index],
      })),
    );
    renderSection();
    expect(await screen.findByTitle("No conclusive investigation for this trace")).toHaveTextContent("-");
    expect(await screen.findByTitle("0 findings from completed investigations")).toHaveTextContent("0");
    expect(await screen.findByTitle("3 findings from completed investigations")).toHaveTextContent("3");
  });

  it("keeps failure labels out of the run totals above the findings table", async () => {
    vi.mocked(agentTraceListCall).mockResolvedValue({
      data: [{ ...runs[0], status: "error", error_count: 8 }],
      next_cursor: null,
    });
    renderSection();
    expect(await screen.findByTestId("agent-trace-row")).toBeVisible();
    expect(screen.getByText(/1 run from/)).toBeVisible();
    expect(screen.queryByText(/failed runs|with errors/)).not.toBeInTheDocument();
  });

  it("does not present a failed findings lookup as an uninvestigated or clean trace", async () => {
    vi.mocked(agentTraceListCall).mockResolvedValue({ data: runs.slice(0, 1), next_cursor: null });
    vi.mocked(apiClient.post).mockRejectedValue(new ApiError("Unavailable", 503, {}));
    renderSection();
    expect(await screen.findByTitle("Could not load findings")).toHaveTextContent("Unavailable");
    expect(screen.queryByTitle("No conclusive investigation for this trace")).not.toBeInTheDocument();
  });

  it("keeps the column picker open when findings finish loading", async () => {
    const user = userEvent.setup();
    const pending = Promise.withResolvers<TraceFindingCount[]>();
    vi.mocked(agentTraceListCall).mockResolvedValue({ data: runs.slice(0, 1), next_cursor: null });
    vi.mocked(apiClient.post).mockReturnValue(pending.promise);
    renderSection();
    expect(await screen.findByTestId("agent-trace-row")).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Columns" }));
    expect(await screen.findByTestId("view-option-cost")).toBeVisible();
    await act(async () => {
      pending.resolve([{ trace_id: runs[0].trace_id, trace_ref: runs[0].trace_ref ?? "", finding_count: 1 }]);
    });
    expect(await screen.findByTitle("1 findings from completed investigations")).toBeVisible();
    expect(screen.getByTestId("view-option-cost")).toBeVisible();
  });

  it("keeps traces available without requesting investigation data for users without access", async () => {
    vi.mocked(agentTraceListCall).mockResolvedValue({ data: runs, next_cursor: null });
    vi.mocked(apiClient.post).mockClear();
    renderSection(false);
    expect(await screen.findAllByTestId("agent-trace-row")).toHaveLength(runs.length);
    expect(screen.queryByRole("columnheader", { name: "Findings" })).not.toBeInTheDocument();
    expect(apiClient.post).not.toHaveBeenCalled();
  });

  it("shows the spend returned for a run", async () => {
    vi.mocked(agentTraceListCall).mockResolvedValue({
      ...(traceList as TracePage),
      data: [{ ...runs[0], spend: 0.025, priced_calls: runs[0].llm_calls }],
    });
    renderSection();

    const row = await screen.findByTestId("agent-trace-row");
    expect(within(row).getByText("$0.03")).toBeInTheDocument();
  });

  it("filters by input text and by trace id", async () => {
    const user = userEvent.setup();
    vi.mocked(agentTraceListCall).mockResolvedValue(traceList as TracePage);
    renderSection();
    await screen.findAllByTestId("agent-trace-row");

    const search = screen.getByRole("combobox", { name: "Search runs" });
    await user.type(search, "acme-404");
    await waitFor(() => expect(screen.getAllByTestId("agent-trace-row")).toHaveLength(1));

    const lead = runs.find((r) => r.name === "research_lead") as TraceSummary;
    await user.clear(search);
    await user.type(search, lead.trace_id.slice(0, 10));
    await waitFor(() =>
      expect(screen.getByTestId("agent-trace-row")).toHaveTextContent("Should we store OTEL agent spans"),
    );
  });

  it("uses recorded agent names for the column, suggestions and filter even when services are shared", async () => {
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
    await user.type(screen.getByRole("combobox", { name: "Search runs" }), "agent:");
    const suggestions = screen.getByRole("listbox", { name: "Search suggestions" });
    expect(
      within(suggestions)
        .getAllByRole("option")
        .map((option) => option.textContent),
    ).toEqual(["billing-agent", "research-agent", "review-agent"]);

    await user.click(within(suggestions).getByRole("option", { name: "review-agent" }));
    await waitFor(() => expect(screen.getAllByTestId("agent-trace-row")).toHaveLength(1));
    const row = screen.getByTestId("agent-trace-row");
    expect(row).toHaveTextContent("billing-agent");
    expect(row).not.toHaveTextContent("shared-app");
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
    expect(within(agentCell(sdkRun)).getByTitle("research-bot · Claude Agent SDK")).toBeInTheDocument();
    expect(within(sdkRun).getByRole("img", { name: "Claude Agent SDK logo", hidden: true })).toHaveAttribute(
      "src",
      expect.stringContaining("anthropic.svg"),
    );
    expect(agentCell(cliRun)).toHaveTextContent(/^Claude Code$/);
    expect(within(plainRun).queryByRole("img", { hidden: true })).not.toBeInTheDocument();
    expect(within(plainRun).getByTestId("span-icon")).toBeInTheDocument();
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
    fireEvent.keyDown(document.body, { key: "Escape" });
    expect(rows[1]).toHaveAttribute("aria-selected", "false");
  });

  it("moves to the next and previous run with j / k and the header arrows", async () => {
    vi.mocked(agentTraceListCall).mockResolvedValue(traceList as TracePage);
    renderSection();
    const rows = await screen.findAllByTestId("agent-trace-row");

    fireEvent.click(rows[0]);
    fireEvent.keyDown(document.body, { key: "j" });
    expect(screen.getByTestId("run-view")).toHaveTextContent(`run ${runs[1].trace_id}`);
    fireEvent.keyDown(document.body, { key: "k" });
    expect(screen.getByTestId("run-view")).toHaveTextContent(`run ${runs[0].trace_id}`);
    expect(screen.getByRole("button", { name: "Previous trace (K)" })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "Next trace (J)" }));
    expect(screen.getByTestId("run-view")).toHaveTextContent(`run ${runs[1].trace_id}`);
  });

  it("opens full screen from a shared link and drops it from the URL on close", async () => {
    vi.mocked(agentTraceListCall).mockResolvedValue(traceList as TracePage);
    const onUrlUpdate = vi.fn();
    renderWithProviders(<AgentTracesSection accessToken="sk-test" isActive range={ROLLING_DAY} />, {
      searchParams: `?trace=${runs[0].trace_id}&fullscreen=true`,
      onUrlUpdate,
    });
    const drawer = await screen.findByRole("complementary", { name: "Trace details" });
    expect(drawer).toHaveStyle({ width: "100%" });
    fireEvent.click(screen.getByRole("button", { name: "Close trace (Esc)" }));
    await waitFor(() => expect(lastUrl(onUrlUpdate).has("trace")).toBe(false));
    expect(lastUrl(onUrlUpdate).has("fullscreen")).toBe(false);
  });

  it("narrows the list to the zoom window named in the URL and clears it on request", async () => {
    vi.mocked(agentTraceListCall).mockResolvedValue(traceList as TracePage);
    const startMs = Date.parse(runs[0].start_time);
    const inWindow = runs.filter((run) => Math.abs(Date.parse(run.start_time) - startMs) <= 1);
    const onUrlUpdate = vi.fn();
    renderWithProviders(<AgentTracesSection accessToken="sk-test" isActive range={PINNED_DAY} />, {
      searchParams: `?from=${startMs - 1}&to=${startMs + 1}`,
      onUrlUpdate,
    });
    await waitFor(() => expect(screen.getAllByTestId("agent-trace-row")).toHaveLength(inWindow.length));
    expect(inWindow.length).toBeLessThan(runs.length);
    fireEvent.click(screen.getByRole("button", { name: "Clear time zoom" }));
    await waitFor(() => expect(screen.getAllByTestId("agent-trace-row")).toHaveLength(runs.length));
    expect(lastUrl(onUrlUpdate).has("from")).toBe(false);
  });

  it("opens the run named by ?trace= even when it is outside the loaded list, and clears it on close", async () => {
    vi.mocked(agentTraceListCall).mockResolvedValue(traceList as TracePage);
    const onUrlUpdate = vi.fn();
    renderWithProviders(<AgentTracesSection accessToken="sk-test" isActive range={ROLLING_DAY} />, {
      searchParams: "?trace=older-than-the-list&trace_ref=ref-9",
      onUrlUpdate,
    });
    const drawer = await screen.findByRole("complementary", { name: "Trace details" });
    expect(within(drawer).getByTestId("run-view")).toHaveTextContent("run older-than-the-list");
    expect(screen.getByRole("button", { name: "Next trace (J)" })).toBeDisabled();
    const rows = await screen.findAllByTestId("agent-trace-row");
    expect(rows.every((row) => row.getAttribute("aria-selected") === "false")).toBe(true);

    fireEvent.click(rows[1]);
    await waitFor(() => expect(lastUrl(onUrlUpdate).get("trace")).toBe(runs[1].trace_id));
    expect(lastUrl(onUrlUpdate).get("trace_ref")).toBe(runs[1].trace_ref ?? null);
    expect(onUrlUpdate.mock.lastCall?.[0].options.history).toBe("push");

    fireEvent.keyDown(document.body, { key: "Escape" });
    await waitFor(() => expect(lastUrl(onUrlUpdate).has("trace")).toBe(false));
    expect(rows[1]).toHaveAttribute("aria-selected", "false");
  });

  it("reads the query from the URL and writes typed changes back", async () => {
    const user = userEvent.setup();
    vi.mocked(agentTraceListCall).mockResolvedValue(traceList as TracePage);
    const onUrlUpdate = vi.fn();
    const mixed = runs.map((run, index) => (index === 1 ? { ...run, status: "error" as const } : run));
    vi.mocked(agentTraceListCall).mockResolvedValue({ data: mixed, next_cursor: null });
    const failed = filterRuns(mixed, "status:error");
    renderWithProviders(<AgentTracesSection accessToken="sk-test" isActive range={ROLLING_DAY} />, {
      searchParams: "?q=status:error",
      onUrlUpdate,
    });
    expect(await screen.findAllByTestId("agent-trace-row")).toHaveLength(failed.length);
    expect(failed.length).toBeLessThan(runs.length);
    const search = screen.getByRole("combobox", { name: "Search runs" });
    expect(search).toHaveTextContent("status:error");

    await user.clear(search);
    await waitFor(() => expect(screen.getAllByTestId("agent-trace-row")).toHaveLength(runs.length));
    await user.type(search, "-status:error");
    await waitFor(() => expect(screen.getAllByTestId("agent-trace-row")).toHaveLength(runs.length - failed.length));
    await waitFor(() => expect(lastUrl(onUrlUpdate).get("q")).toBe("-status:error"));
  });

  it("plots every loaded run on the timeline", async () => {
    vi.mocked(agentTraceListCall).mockResolvedValue(traceList as TracePage);
    renderWindowed();
    await screen.findAllByTestId("agent-trace-row");

    expect(screen.getByTestId("timeline")).toBeInTheDocument();
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

    fireEvent.keyDown(screen.getByTestId("timeline"), { key: "Escape" });
    expect(screen.queryByTestId("timeline-selection")).not.toBeInTheDocument();
    expect(rowCount()).toBe(runs.length);
  });
});

describe("AgentTracesPage", () => {
  beforeEach(() => {
    testQueryClient.clear();
    vi.mocked(agentTraceListCall).mockReset();
    vi.mocked(apiClient.get).mockResolvedValue({ data: [] });
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("names the rolling preset while Live and pins the actual range once paused", async () => {
    const pausedAt = Date.parse("2026-10-03T12:00:00.500");
    vi.useFakeTimers({ toFake: ["Date"], now: pausedAt });
    vi.mocked(agentTraceListCall).mockResolvedValue(traceList as TracePage);
    renderWithProviders(<AgentTracesPage accessToken="sk-test" />);
    await screen.findByTestId("runs-table");

    const trigger = screen.getByRole("button", { name: "Time range" });
    const live = screen.getByRole("button", { name: "Live" });
    expect(live).toHaveAttribute("aria-pressed", "true");
    expect(trigger).toHaveTextContent("Last 24 hours");

    fireEvent.click(trigger);
    fireEvent.click(await screen.findByRole("menuitemradio", { name: "Last 7 days" }));
    expect(trigger).toHaveTextContent("Last 7 days");
    await waitFor(() => {
      const last = vi.mocked(agentTraceListCall).mock.calls.at(-1)?.[0];
      expect((last?.endMs ?? 0) - (last?.startMs ?? 0)).toBeGreaterThanOrEqual(7 * 24 * 3600 * 1000 - 60_000);
    });

    fireEvent.click(live);
    expect(live).toHaveAttribute("aria-pressed", "false");
    expect(trigger).toHaveTextContent(/ to /);
    expect(trigger).not.toHaveTextContent("Last 7 days");

    await waitFor(() => expect(vi.mocked(agentTraceListCall).mock.calls.at(-1)?.[0].endMs).toBe(pausedAt));
  });

  it("keeps the time controls on an empty range the user picked, instead of showing onboarding", async () => {
    vi.mocked(agentTraceListCall).mockResolvedValue(traceList as TracePage);
    renderWithProviders(<AgentTracesPage accessToken="sk-test" />);
    await screen.findByTestId("runs-table");

    vi.mocked(agentTraceListCall).mockResolvedValue({ ...(traceList as TracePage), data: [] });
    fireEvent.click(screen.getByRole("button", { name: "Time range" }));
    fireEvent.click(await screen.findByRole("menuitemradio", { name: "Last hour" }));

    expect(await screen.findByText("No runs in this time range")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Time range" })).toBeInTheDocument();
    expect(screen.queryByTestId("tracing-setup-card")).not.toBeInTheDocument();
  });

  it("opens on the range named by ?hours= and writes a new preset back", async () => {
    vi.mocked(agentTraceListCall).mockResolvedValue(traceList as TracePage);
    const onUrlUpdate = vi.fn();
    renderWithProviders(<AgentTracesPage accessToken="sk-test" />, { searchParams: "?hours=168", onUrlUpdate });
    await screen.findByTestId("runs-table");
    const { startMs, endMs } = vi.mocked(agentTraceListCall).mock.calls[0][0];
    expect(endMs - startMs).toBeGreaterThanOrEqual(7 * 24 * 3600 * 1000 - 60_000);

    fireEvent.click(screen.getByRole("button", { name: "Time range" }));
    fireEvent.click(await screen.findByRole("menuitemradio", { name: "Last hour" }));
    await waitFor(() => {
      const last = vi.mocked(agentTraceListCall).mock.calls.at(-1)?.[0];
      expect((last?.endMs ?? 0) - (last?.startMs ?? 0)).toBeLessThanOrEqual(3600 * 1000 + 60_000);
    });
    await waitFor(() => expect(lastUrl(onUrlUpdate).get("hours")).toBe("1"));
  });

  it("keeps the shown runs behind a search spinner while a newly picked range loads", async () => {
    vi.mocked(agentTraceListCall).mockResolvedValue(traceList as TracePage);
    renderWithProviders(<AgentTracesPage accessToken="sk-test" />);
    expect(await screen.findAllByTestId("agent-trace-row")).toHaveLength(runs.length);
    expect(screen.queryByRole("status", { name: "Loading results" })).not.toBeInTheDocument();

    let resolveWeek: (page: TracePage) => void = () => {};
    vi.mocked(agentTraceListCall).mockImplementation(
      () =>
        new Promise<TracePage>((resolve) => {
          resolveWeek = resolve;
        }),
    );
    fireEvent.click(screen.getByRole("button", { name: "Time range" }));
    fireEvent.click(await screen.findByRole("menuitemradio", { name: "Last 7 days" }));

    expect(await screen.findByRole("status", { name: "Loading results" })).toBeInTheDocument();
    expect(screen.getAllByTestId("agent-trace-row")).toHaveLength(runs.length);
    expect(screen.queryByText("Loading runs…")).not.toBeInTheDocument();

    act(() => resolveWeek({ ...(traceList as TracePage), data: runs.slice(0, 1) }));
    await waitFor(() => expect(screen.queryByRole("status", { name: "Loading results" })).not.toBeInTheDocument());
    expect(screen.getAllByTestId("agent-trace-row")).toHaveLength(1);
  });

  it("keeps the timeline on the shown runs' window while a narrower range loads", async () => {
    vi.useFakeTimers({ toFake: ["Date"], now: Date.parse("2026-09-30T12:00Z") });
    vi.mocked(agentTraceListCall).mockResolvedValue(traceList as TracePage);
    renderWithProviders(<AgentTracesPage accessToken="sk-test" />, { searchParams: "?hours=24" });
    expect(await screen.findAllByTestId("agent-trace-row")).toHaveLength(runs.length);
    const sum = () => bucketRunCounts().reduce((a, b) => a + b, 0);
    expect(sum()).toBe(runs.length);

    vi.mocked(agentTraceListCall).mockImplementation(() => new Promise<TracePage>(() => {}));
    fireEvent.click(screen.getByRole("button", { name: "Time range" }));
    fireEvent.click(await screen.findByRole("menuitemradio", { name: "Last hour" }));

    expect(await screen.findByRole("status", { name: "Loading results" })).toBeInTheDocument();
    expect(screen.getAllByTestId("agent-trace-row")).toHaveLength(runs.length);
    expect(sum()).toBe(runs.length);
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
