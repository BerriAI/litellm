"use client";

import moment from "moment";
import { useMemo, useState } from "react";

import { AgentTracesTable } from "./AgentTracesTable";
import { ALL_SERVICES, RunsToolbar, type RunStatusFilter } from "./RunsToolbar";
import { RunView } from "./TraceDrawer";
import type { TraceSummary } from "./traceTypes";
import { previewText } from "./traceUtils";
import { TimeRangeControls } from "./TimeRangeControls";
import { TracesTimeline, type TimeWindow } from "./TracesTimeline";
import { TracingSetupCard } from "./TracingSetupCard";
import { traceWindowStartMs, useAgentTraces } from "./useAgentTraces";

/** Client-side search (input text or trace id) plus service / status filters over the loaded runs. */
export function filterRuns(
  runs: TraceSummary[],
  query: string,
  service: string,
  status: RunStatusFilter,
): TraceSummary[] {
  const q = query.trim().toLowerCase();
  return runs.filter((run) => {
    const haystack = [run.trace_id, previewText(run.input_preview), run.name].map((s) => s.toLowerCase());
    const matchesQuery = !q || haystack.some((text) => text.includes(q));
    const matchesService = service === ALL_SERVICES || run.service === service;
    const failed = run.error_count > 0;
    const matchesStatus = status === "all" || (status === "error" ? failed : !failed);
    return matchesQuery && matchesService && matchesStatus;
  });
}

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
}: AgentTracesSectionProps) {
  const [openTrace, setOpenTrace] = useState<TraceSummary | null>(null);
  const [query, setQuery] = useState("");
  const [service, setService] = useState(ALL_SERVICES);
  const [status, setStatus] = useState<RunStatusFilter>("all");
  const [showSetup, setShowSetup] = useState(false);
  const [zoom, setZoom] = useState<TimeWindow | null>(null);
  const [rangeChanged, setRangeChanged] = useState(false);
  const traceQuery = { accessToken, startTime, endTime, isCustomDate, isLiveTail, enabled: isActive };
  const traces = useAgentTraces(traceQuery);

  const services = useMemo(() => Array.from(new Set(traces.traces.map((t) => t.service))).sort(), [traces.traces]);
  // Relative ranges end "now" (the list query uses Date.now() too); round to the minute so the histogram is stable.
  const endMs = isCustomDate ? moment(endTime).valueOf() : moment().endOf("minute").valueOf();
  const range = useMemo(
    () => ({ startMs: traceWindowStartMs(startTime, endTime, isCustomDate, endMs), endMs }),
    [startTime, endTime, isCustomDate, endMs],
  );
  const filtered = useMemo(
    () => filterRuns(traces.traces, query, service, status),
    [traces.traces, query, service, status],
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

  if (traces.notEnabledDetail !== null) return <TracingSetupCard detail={traces.notEnabledDetail} />;
  // Onboarding only on the first, default view; an empty range the user picked keeps its controls.
  const isEmpty = !traces.isLoading && !traces.error && traces.traces.length === 0;
  if (isEmpty && !rangeChanged) return <TracingSetupCard detail={null} />;
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
        <TracingSetupCard detail={null} connected />
      </div>
    );
  }

  if (openTrace !== null) {
    return <RunView traceId={openTrace.trace_id} traceRef={openTrace.trace_ref} accessToken={accessToken} onBack={() => openRun(null)} />;
  }

  return (
    <div className="flex min-h-[560px] flex-1 flex-col overflow-hidden border-y border-border bg-card">
      <RunsToolbar
        query={query}
        service={service}
        status={status}
        services={services}
        onQueryChange={setQuery}
        onServiceChange={setService}
        onStatusChange={setStatus}
      >
        <button
          type="button"
          onClick={() => setShowSetup(true)}
          className="shrink-0 px-1 text-[11px] text-muted-foreground underline-offset-2 hover:text-info hover:underline"
        >
          Set up tracing
        </button>
        {timeControls && (
          <TimeRangeControls
            range={zoom ?? range}
            rangeHours={timeControls.rangeHours}
            onRangeHoursChange={(hours) => changeRange(hours, timeControls.onRangeHoursChange)}
            live={isLiveTail}
            onLiveChange={timeControls.onLiveChange}
            zoomed={zoom !== null}
            onResetZoom={() => setZoom(null)}
          />
        )}
      </RunsToolbar>
      <TracesTimeline runs={filtered} range={range} selection={zoom} onSelect={setZoom} />
      <AgentTracesTable
        traces={runs}
        isLoading={traces.isLoading}
        error={traces.error}
        hasMore={traces.hasMore}
        onLoadMore={traces.loadMore}
        onOpenTrace={openRun}
      />
      <footer
        data-testid="runs-footer"
        className="flex h-8 shrink-0 items-center border-t border-border bg-muted/40 px-3 font-mono text-[11px] text-muted-foreground"
      >
        {runs.length} {runs.length === 1 ? "run" : "runs"}
        {zoom && (
          <button
            type="button"
            onClick={() => setZoom(null)}
            aria-label="Clear time zoom"
            className="ml-3 rounded border border-info/40 bg-info/10 px-1.5 text-info hover:bg-info/20"
          >
            {moment(zoom.startMs).format("MMM DD, HH:mm")} to {moment(zoom.endMs).format("MMM DD, HH:mm")} ×
          </button>
        )}
        <span className="ml-auto">{traces.isFetching ? "Updating…" : "Updated just now"}</span>
      </footer>
    </div>
  );
}
