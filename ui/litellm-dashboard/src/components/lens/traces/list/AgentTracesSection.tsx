"use client";

import moment from "moment";
import { RefreshCw } from "lucide-react";
import { traceAgentNames } from "../utils";
import { useMemo, useState } from "react";

import { filterRuns } from "./runSearch/runQuery";
import { RunsToolbar } from "./runSearch/RunsToolbar";
import { Inspector } from "@/components/shared/Inspector";
import { Button } from "@/components/ui/button";

import { AgentTracesTable } from "./AgentTracesTable";
import { useTraceFindings } from "./useTraceFindings";
import { type TraceRef, traceKey, traceRefOf, useOpenTraceRouting, useRunFilterRouting } from "../routing";
import type { TraceSummary } from "../types";
import { RunView } from "../detail/run/RunView";
import { useZoomRouting } from "@/components/shared/timeRange/routing";
import { type RelativeRange, type TimeWindow, timeWindow } from "@/components/shared/timeRange/timeRange";
import { TimeRangeControls } from "@/components/shared/timeRange/TimeRangeControls";
import type { RelativeRangeState } from "@/components/shared/timeRange/useRelativeRange";
import { TracesTimeline } from "./TracesTimeline";
import { TracingSetupCard } from "../../onboarding/tracing/TracingSetupCard";
import { useTracesLive } from "../api";
import { type AgentTracesResult, useAgentTraces, useTraceAvailability } from "./useAgentTraces";

const DRAWER_WIDTH_KEY = "litellm.agentTraces.drawerWidth";

const filterByWindow = (runs: TraceSummary[], range: TimeWindow): TraceSummary[] =>
  runs.filter((run) => {
    const t = moment(run.start_time).valueOf();
    return t >= range.startMs && t < range.endMs;
  });

interface AgentTracesSectionProps {
  accessToken: string;
  isActive: boolean;
  range: RelativeRange;
  /** Page-owned time range + live state; when given, the toolbar shows the range / Live control group. */
  timeControls?: RelativeRangeState;
  readOnly?: boolean;
  canMintTracingKey?: boolean;
  canViewFindings?: boolean;
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
  range,
  timeControls,
  readOnly = false,
  canMintTracingKey = false,
  canViewFindings,
}: AgentTracesSectionProps) {
  const live = useTracesLive();
  const { trace: openTrace, openTrace: openRun, selection, fullScreen, setFullScreen } = useOpenTraceRouting();
  const { query, setQuery, agent, status } = useRunFilterRouting();
  const [showSetup, setShowSetup] = useState(false);
  const [zoom, setZoom] = useZoomRouting();
  const [rangeChanged, setRangeChanged] = useState(false);
  const traceQuery = { accessToken, range, enabled: isActive };
  const traces = useAgentTraces(traceQuery);
  const setup = useTracingSetup(traces, isActive, rangeChanged);
  const checkHistory = setup.isEmpty && !rangeChanged;
  const history = useTraceAvailability(accessToken, isActive && checkHistory && setup.disabledDetail == null);

  const checkTraces = () => {
    traces.refetch();
    if (setup.disabledDetail == null) void history.refetch();
  };

  // A live range ends "now" (the list query uses Date.now() too); round to the minute so the histogram is stable.
  const minuteEndMs = moment().endOf("minute").valueOf();
  const pickedWindow = useMemo(() => timeWindow(range, minuteEndMs), [range, minuteEndMs]);
  // While the previous range's rows stay on screen, describe them with their own window.
  const window = traces.isPlaceholder && traces.window ? traces.window : pickedWindow;
  const filtered = useMemo(
    () => filterRuns(traces.traces, query, { agent, status }),
    [traces.traces, query, agent, status],
  );
  const runs = useMemo(() => (zoom ? filterByWindow(filtered, zoom) : filtered), [filtered, zoom]);
  const runRefs = useMemo(() => runs.map(traceRefOf), [runs]);
  const findings = useTraceFindings(accessToken, runs, isActive, canViewFindings);

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
      <div className="flex min-h-0 flex-1 flex-col overflow-hidden bg-card">
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
        <RunsToolbar
          query={query}
          onQueryChange={setQuery}
          runs={traces.traces}
          range={zoom ?? window}
          busy={traces.isPlaceholder}
        >
          <TracingSetupAction available={traces.traces.length > 0} live={live} onSetup={() => setShowSetup(true)} />
          <Button
            variant="ghost"
            size="icon-sm"
            className="h-full w-10 shrink-0 rounded-none border-l"
            aria-label="Refresh traces"
            title="Refresh traces"
            disabled={traces.isFetching}
            onClick={checkTraces}
          >
            <RefreshCw className="size-3.5" />
          </Button>
          {timeControls && (
            <TimeRangeControls
              range={range}
              zoom={zoom}
              onHoursChange={(hours) => changeRange(hours, timeControls.setHours)}
              showLive={live}
              onLiveChange={timeControls.setLive}
            />
          )}
        </RunsToolbar>
        <TraceCounts runs={filtered} />
        <TracesTimeline runs={filtered} range={window} selection={zoom} onSelect={setZoom} />
        <AgentTracesTable
          traces={runs}
          findings={findings}
          canViewFindings={canViewFindings}
          isLoading={traces.isLoading || (checkHistory && history.isLoading)}
          error={traces.error}
          hasMore={traces.hasMore}
          isFetching={traces.isFetching}
          isPlaceholder={traces.isPlaceholder}
          onRetry={traces.hasMore ? traces.loadMore : traces.refetch}
          onLoadMore={traces.loadMore}
          rangeEmpty={traces.traces.length === 0}
          onSetUpTracing={() => setShowSetup(true)}
        />
        <TraceFooter runs={runs} hasMore={traces.hasMore} />
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

function TraceCounts({ runs }: { runs: readonly TraceSummary[] }) {
  return (
    <div className="flex shrink-0 flex-wrap items-center gap-x-3 gap-y-1 px-3 pt-2 text-xs text-muted-foreground">
      <span>
        {runs.length} {runs.length === 1 ? "run" : "runs"} from {new Set(runs.flatMap(traceAgentNames)).size} agents
      </span>
    </div>
  );
}

function TraceFooter({ runs, hasMore }: { runs: readonly TraceSummary[]; hasMore: boolean }) {
  return (
    <footer className="flex h-8 shrink-0 items-center border-t bg-muted/30 px-3 text-xs text-muted-foreground">
      {runs.length} {runs.length === 1 ? "run" : "runs"}
      {hasMore ? " loaded" : ""}
    </footer>
  );
}

function TracingSetupAction({ available, live, onSetup }: { available: boolean; live: boolean; onSetup: () => void }) {
  if (!available) return null;
  return (
    <Button
      variant="ghost"
      size="sm"
      className="h-full shrink-0 rounded-none border-l px-3 text-xs"
      disabled={!live}
      onClick={onSetup}
    >
      Set up tracing
    </Button>
  );
}
