"use client";

import moment from "moment";
import { useMemo, useState } from "react";

import { filterRuns } from "./runSearch/runQuery";
import { RunsToolbar } from "./runSearch/RunsToolbar";
import { Inspector } from "@/components/shared/Inspector";
import { Button } from "@/components/ui/button";

import { AgentTracesTable } from "./AgentTracesTable";
import {
  type TraceRef,
  traceKey,
  traceRefOf,
  useOpenTraceRouting,
  useRunFilterRouting,
  useZoomRouting,
} from "../routing";
import type { TraceSummary } from "../types";
import { RunView } from "../detail/run/RunView";
import { TimeRangeControls } from "@/components/shared/timeline/TimeRangeControls";
import { TracesTimeline } from "./TracesTimeline";
import type { TimeWindow } from "@/components/shared/timeline/Timeline";
import { TracingSetupCard } from "../../onboarding/tracing/TracingSetupCard";
import { useTracesLive } from "../api";
import { type AgentTracesResult, traceWindowStartMs, useAgentTraces, useTraceAvailability } from "./useAgentTraces";

const DRAWER_WIDTH_KEY = "litellm.agentTraces.drawerWidth";

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
  readOnly?: boolean;
  canMintTracingKey?: boolean;
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

/** The Runs view: filters and the runs table — or one run, in place, once a row is clicked. */
export function AgentTracesSection({
  accessToken,
  isActive,
  startTime,
  endTime,
  isCustomDate,
  isLiveTail,
  timeControls,
  readOnly = false,
  canMintTracingKey = false,
}: AgentTracesSectionProps) {
  const live = useTracesLive();
  const { trace: openTrace, openTrace: openRun, selection, fullScreen, setFullScreen } = useOpenTraceRouting();
  const { query, setQuery } = useRunFilterRouting();
  const [showSetup, setShowSetup] = useState(false);
  const [zoom, setZoom] = useZoomRouting();
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

  // Relative ranges end "now" (the list query uses Date.now() too); round to the minute so the histogram is stable.
  const endMs = isCustomDate ? moment(endTime).valueOf() : moment().endOf("minute").valueOf();
  const range = useMemo(
    () => ({ startMs: traceWindowStartMs(startTime, endTime, isCustomDate, endMs), endMs }),
    [startTime, endTime, isCustomDate, endMs],
  );
  const filtered = useMemo(() => filterRuns(traces.traces, query), [traces.traces, query]);
  const runs = useMemo(() => (zoom ? filterByWindow(filtered, zoom) : filtered), [filtered, zoom]);
  const runRefs = useMemo(() => runs.map(traceRefOf), [runs]);

  const changeRange = (hours: number, apply: (hours: number) => void) => {
    setZoom(null);
    setRangeChanged(true);
    apply(hours);
  };

  const openSentTrace = (trace: TraceSummary) => {
    setShowSetup(false);
    setRangeChanged(true);
    checkTraces();
    openRun(traceRefOf(trace));
  };
  const setupProps = {
    accessToken,
    readOnly,
    canMintTracingKey,
    onOpenTrace: openSentTrace,
    onCheck: checkTraces,
    checking: traces.isFetching,
  };

  if (setup.disabledDetail != null) return <TracingSetupCard detail={setup.disabledDetail} {...setupProps} />;
  // Onboarding only on the first, default view; an empty range the user picked keeps its controls.
  if (checkHistory && !history.error && history.data === false)
    return <TracingSetupCard detail={null} {...setupProps} />;
  if (showSetup) {
    return (
      <div>
        <button
          type="button"
          onClick={() => setShowSetup(false)}
          className="mb-3 text-sm text-muted-foreground hover:text-foreground"
        >
          ← Back to traces
        </button>
        <TracingSetupCard detail={null} connected {...setupProps} />
      </div>
    );
  }

  return (
    <Inspector.Root
      items={runRefs}
      itemKey={traceKey}
      selected={openTrace}
      onSelectedChange={openRun}
      noun="trace"
      storageKey={DRAWER_WIDTH_KEY}
      fullScreen={fullScreen}
      onFullScreenChange={setFullScreen}
    >
      <div className="flex min-h-[560px] flex-1 flex-col overflow-hidden bg-card">
        {checkHistory && <TraceHistoryError history={history} />}
        <TracesReceived received={setup.received} />
        <Inspector.Panel label="Trace details" testId="run-drawer">
          {(shown: TraceRef) => (
            <RunView
              traceId={shown.traceId}
              traceRef={shown.traceRef}
              selection={selection}
              accessToken={accessToken}
              onBack={() => openRun(null)}
              embedded
            />
          )}
        </Inspector.Panel>
        <RunsToolbar query={query} onQueryChange={setQuery} runs={traces.traces} range={zoom ?? range}>
          {timeControls && (
            <TimeRangeControls
              fixedRange={zoom ?? (isLiveTail ? null : range)}
              rangeHours={timeControls.rangeHours}
              onRangeHoursChange={(hours) => changeRange(hours, timeControls.onRangeHoursChange)}
              live={isLiveTail}
              showLive={live}
              onLiveChange={timeControls.onLiveChange}
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
          rangeEmpty={traces.traces.length === 0}
          onSetUpTracing={() => setShowSetup(true)}
        />
      </div>
    </Inspector.Root>
  );
}

function TracesReceived({ received }: { received: boolean }) {
  if (!received) return null;
  return (
    <p role="status" className="border-b px-3 py-3 text-sm text-emerald-700 dark:text-emerald-400">
      Traces received. Select a run to inspect it.
    </p>
  );
}
