"use client";
import { useLensDemo } from "@/components/lens/LensDemoContext";

import moment from "moment";
import { useMemo, useState } from "react";

import { Button } from "@/components/ui/button";

import { AgentTracesTable } from "./AgentTracesTable";
import { RunDrawer } from "./RunDrawer";
import { ALL_AGENTS, RunsToolbar, type RunStatusFilter } from "./RunsToolbar";
import type { TraceSummary } from "./traceTypes";
import { previewText, traceAgentNames } from "./traceUtils";
import { TimeRangeControls } from "./TimeRangeControls";
import { TracesTimeline, type TimeWindow } from "./TracesTimeline";
import { ActiveDot } from "./ActiveDot";
import { TracingSetupCard } from "./TracingSetupCard";
import { type AgentTracesResult, traceWindowStartMs, useAgentTraces, useTraceAvailability } from "./useAgentTraces";

/** Client-side search (input text or trace id) plus agent / status filters over the loaded runs. */
export function filterRuns(
  runs: TraceSummary[],
  query: string,
  agent: string,
  status: RunStatusFilter,
): TraceSummary[] {
  const q = query.trim().toLowerCase();
  return runs.filter((run) => {
    const haystack = [run.trace_id, previewText(run.input_preview), run.name].map((s) => s.toLowerCase());
    const matchesQuery = !q || haystack.some((text) => text.includes(q));
    const matchesAgent = agent === ALL_AGENTS || traceAgentNames(run).includes(agent);
    const failed = run.error_count > 0;
    const matchesStatus = status === "all" || (status === "error" ? failed : !failed);
    return matchesQuery && matchesAgent && matchesStatus;
  });
}

const runKey = (run: TraceSummary): string => run.trace_ref || run.trace_id;

const filterByWindow = (runs: TraceSummary[], range: TimeWindow): TraceSummary[] =>
  runs.filter((run) => {
    const t = moment(run.start_time).valueOf();
    return t >= range.startMs && t < range.endMs;
  });

export interface TimeControls {
  rangeHours: number;
  onRangeHoursChange: (hours: number) => void;
  onLiveChange: (live: boolean) => void;
}

interface AgentTracesSectionProps {
  accessToken: string;
  isActive: boolean;
  startTime: string;
  endTime: string;
  isCustomDate: boolean;
  isLiveTail: boolean;
  /** Page-owned time range + live state; when given, the toolbar shows the range / Live control group. */
  timeControls?: TimeControls;
  /** Called when a run opens / closes, so the page can hide its own header while a run fills the view. */
  onRunOpenChange?: (open: boolean) => void;
  readOnly?: boolean;
  canMintTracingKey?: boolean;
  onDemo?: () => void;
}

function useTracingSetup(traces: AgentTracesResult, isActive: boolean, rangeChanged: boolean) {
  const [setupResult, setSetupResult] = useState<{ detail: string | null } | null>(null);
  const waitingForFirstTrace = traces.traces.length === 0 && !rangeChanged;
  const settledResponse = isActive && !traces.isFetching && !traces.error;
  const rememberSetup = waitingForFirstTrace || setupResult !== null;
  if (settledResponse && rememberSetup && setupResult?.detail !== traces.notEnabledDetail) {
    setSetupResult({ detail: traces.notEnabledDetail });
  }

  const disabledDetail = traces.notEnabledDetail ?? (traces.isFetching ? setupResult?.detail : null);
  const loadingFirstPage = traces.isLoading && setupResult === null;
  const isEmpty = !loadingFirstPage && !traces.error && traces.traces.length === 0;
  return { disabledDetail, isEmpty, received: setupResult !== null && traces.traces.length > 0 };
}

function TraceHistoryError({ history }: { history: ReturnType<typeof useTraceAvailability> }) {
  if (!history.error) return null;
  return (
    <div role="alert" className="flex items-center justify-between gap-4 border-b px-3 py-3 text-sm">
      <p>Could not check earlier traces. {history.error.message}</p>
      <Button variant="outline" size="sm" disabled={history.isFetching} onClick={() => void history.refetch()}>
        Retry trace check
      </Button>
    </div>
  );
}

/** The Runs view: filters, the runs table and footer — or one run, in place, once a row is clicked. */
export function AgentTracesSection({
  accessToken,
  isActive,
  startTime,
  endTime,
  isCustomDate,
  isLiveTail,
  timeControls,
  onRunOpenChange,
  readOnly = false,
  canMintTracingKey = false,
  onDemo,
}: AgentTracesSectionProps) {
  const demo = useLensDemo();
  const [openTrace, setOpenTrace] = useState<TraceSummary | null>(null);
  const [query, setQuery] = useState("");
  const [agent, setAgent] = useState(ALL_AGENTS);
  const [status, setStatus] = useState<RunStatusFilter>("all");
  const [showSetup, setShowSetup] = useState(false);
  const [zoom, setZoom] = useState<TimeWindow | null>(null);
  const [rangeChanged, setRangeChanged] = useState(false);
  const traceQuery = { accessToken, startTime, endTime, isCustomDate, isLiveTail, enabled: isActive };
  const traces = useAgentTraces(traceQuery);
  const setup = useTracingSetup(traces, isActive, rangeChanged);
  const checkHistory = setup.isEmpty && !rangeChanged;
  const history = useTraceAvailability(accessToken, isActive && checkHistory && setup.disabledDetail == null);

  const checkTraces = () => {
    traces.refetch();
    if (setup.disabledDetail == null) void history.refetch();
  };

  const agents = useMemo(() => Array.from(new Set(traces.traces.flatMap(traceAgentNames))).sort(), [traces.traces]);
  // Relative ranges end "now" (the list query uses Date.now() too); round to the minute so the histogram is stable.
  const endMs = isCustomDate ? moment(endTime).valueOf() : moment().endOf("minute").valueOf();
  const range = useMemo(
    () => ({ startMs: traceWindowStartMs(startTime, endTime, isCustomDate, endMs), endMs }),
    [startTime, endTime, isCustomDate, endMs],
  );
  const filtered = useMemo(
    () => filterRuns(traces.traces, query, agent, status),
    [traces.traces, query, agent, status],
  );
  const runs = useMemo(() => (zoom ? filterByWindow(filtered, zoom) : filtered), [filtered, zoom]);

  const changeRange = (hours: number, apply: (hours: number) => void) => {
    setZoom(null);
    setRangeChanged(true);
    apply(hours);
  };

  const openRun = (trace: TraceSummary | null) => {
    setOpenTrace(trace);
    onRunOpenChange?.(trace !== null);
  };

  const openSentTrace = (trace: TraceSummary) => {
    setShowSetup(false);
    setRangeChanged(true);
    checkTraces();
    openRun(trace);
  };
  const setupProps = {
    accessToken,
    readOnly,
    canMintTracingKey,
    onOpenTrace: openSentTrace,
    onCheck: checkTraces,
    checking: traces.isFetching,
  };

  if (setup.disabledDetail != null)
    return <TracingSetupCard detail={setup.disabledDetail} onDemo={onDemo} {...setupProps} />;
  // Onboarding only on the first, default view; an empty range the user picked keeps its controls.
  if (checkHistory && !history.error && history.data === false)
    return <TracingSetupCard detail={null} onDemo={onDemo} {...setupProps} />;
  if (showSetup) {
    return (
      <div>
        <button
          type="button"
          onClick={() => setShowSetup(false)}
          className="mb-3 text-[13px] text-muted-foreground hover:text-foreground"
        >
          ← Back to traces
        </button>
        <TracingSetupCard detail={null} connected {...setupProps} />
      </div>
    );
  }

  const toggleRun = (trace: TraceSummary | null) =>
    openRun(trace !== null && openTrace !== null && runKey(trace) === runKey(openTrace) ? null : trace);

  return (
    <div className="flex min-h-[560px] flex-1 flex-col overflow-hidden border-y border-border bg-card">
      {checkHistory && <TraceHistoryError history={history} />}
      <TracesReceived received={setup.received} />
      <RunDrawer trace={openTrace} runs={runs} accessToken={accessToken} onSelect={openRun} />
      <RunsToolbar
        query={query}
        agent={agent}
        status={status}
        agents={agents}
        onQueryChange={setQuery}
        onAgentChange={setAgent}
        onStatusChange={setStatus}
      >
        {!demo && (
          <Button variant="outline" size="sm" onClick={() => setShowSetup(true)} className="shrink-0 gap-1.5">
            <ActiveDot />
            Set up tracing
          </Button>
        )}
        {timeControls && (
          <TimeRangeControls
            range={zoom ?? range}
            rangeHours={timeControls.rangeHours}
            onRangeHoursChange={(hours) => changeRange(hours, timeControls.onRangeHoursChange)}
            live={isLiveTail}
            showLive={!demo}
            onLiveChange={timeControls.onLiveChange}
            onRefresh={() => {
              setZoom(null);
              checkTraces();
            }}
            refreshing={traces.isFetching}
          />
        )}
      </RunsToolbar>
      <TracesTimeline runs={filtered} range={range} selection={zoom} onSelect={setZoom} />
      <AgentTracesTable
        traces={runs}
        isLoading={traces.isLoading || (checkHistory && history.isLoading)}
        error={traces.error}
        hasMore={traces.hasMore}
        isFetching={traces.isFetching}
        onRetry={traces.hasMore ? traces.loadMore : traces.refetch}
        onLoadMore={traces.loadMore}
        onOpenTrace={toggleRun}
        selectedKey={openTrace === null ? null : runKey(openTrace)}
      />
      <RunsFooter
        count={runs.length}
        zoom={zoom}
        isFetching={traces.isFetching}
        failed={!!traces.error}
        onResetZoom={() => setZoom(null)}
      />
    </div>
  );
}

function TracesReceived({ received }: { received: boolean }) {
  const demo = useLensDemo();
  if (!received || demo) return null;
  return (
    <p role="status" className="border-b px-3 py-3 text-sm text-emerald-700 dark:text-emerald-400">
      Traces received. Select a run to inspect it.
    </p>
  );
}

function RunsFooter({
  count,
  zoom,
  isFetching,
  failed,
  onResetZoom,
}: {
  count: number;
  zoom: TimeWindow | null;
  isFetching: boolean;
  failed: boolean;
  onResetZoom: () => void;
}) {
  const settled = failed ? "Update failed" : "Updated just now";
  const status = isFetching ? "Updating…" : settled;
  return (
    <footer
      data-testid="runs-footer"
      className="flex h-8 shrink-0 items-center border-t border-border bg-muted/40 px-3 font-mono text-[11px] text-muted-foreground"
    >
      {count} {count === 1 ? "run" : "runs"}
      {zoom && (
        <button
          type="button"
          onClick={() => onResetZoom()}
          aria-label="Clear time zoom"
          className="ml-3 rounded border border-info/40 bg-info/10 px-1.5 text-info hover:bg-info/20"
        >
          {moment(zoom.startMs).format("MMM DD, HH:mm")} to {moment(zoom.endMs).format("MMM DD, HH:mm")} ×
        </button>
      )}
      <span className="ml-auto">{status}</span>
    </footer>
  );
}
