"use client";

import { ArrowLeft, ScanSearch } from "lucide-react";
import moment from "moment";
import { type ComponentProps, useMemo, useState } from "react";

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
  useRunOrderRouting,
  useTracingSetupRoute,
} from "../routing";
import type { TraceSummary } from "../types";
import { RunView } from "../detail/run/RunView";
import { useZoomRouting } from "@/components/shared/timeRange/routing";
import { type RelativeRange, timeWindow } from "@/components/shared/timeRange/timeRange";
import { TimeRangeControls } from "@/components/shared/timeRange/TimeRangeControls";
import type { RelativeRangeState } from "@/components/shared/timeRange/useRelativeRange";
import { Timeline } from "@/components/shared/timeline/Timeline";
import { TracingSetupCard } from "../../onboarding/tracing/TracingSetupCard";
import { useTracesLive } from "../api";
import { type AgentTracesResult, useAgentTraces, useTraceAvailability } from "./useAgentTraces";
import { useTraceHistogram } from "./useTraceHistogram";
import type { InvestigateScope } from "../../route";

const DRAWER_WIDTH_KEY = "litellm.agentTraces.drawerWidth";
const RUN_NOUN = { singular: "run", plural: "runs" };

export type TimeControls = Pick<RelativeRangeState, "setHours" | "setLive">;

interface AgentTracesSectionProps {
  accessToken: string;
  isActive: boolean;
  range: RelativeRange;
  /** Page-owned range setters; when given, the toolbar shows the range / Live control group. */
  timeControls?: TimeControls;
  readOnly?: boolean;
  canMintTracingKey?: boolean;
  /** Offered once a search narrows the runs, to review exactly those runs in a new investigation. */
  onInvestigate?: (scope: InvestigateScope) => void;
}

const HOUR_MS = 3_600_000;

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
  onInvestigate,
}: AgentTracesSectionProps) {
  const live = useTracesLive();
  const { trace: openTrace, openTrace: openRun, selection, fullScreen, setFullScreen } = useOpenTraceRouting();
  const { query, setQuery } = useRunFilterRouting();
  const [connecting, setConnecting] = useTracingSetupRoute();
  const [order, setOrder] = useRunOrderRouting();
  const [zoom, setZoom] = useZoomRouting();
  const [rangeChanged, setRangeChanged] = useState(false);
  const traceQuery = { accessToken, range, enabled: isActive, q: query, zoom, order };
  const traces = useAgentTraces(traceQuery);
  const narrowed = rangeChanged || zoom !== null || query.trim() !== "";
  const setup = useTracingSetup(traces, isActive, narrowed);
  const checkHistory = setup.isEmpty && !narrowed;
  const history = useTraceAvailability(accessToken, isActive && checkHistory && setup.disabledDetail == null);

  const checkTraces = () => {
    traces.refetch();
    if (setup.disabledDetail == null) void history.refetch();
  };

  // A live range ends "now" (the list query uses Date.now() too); round to the minute so the histogram is stable.
  const minuteEndMs = moment().endOf("minute").valueOf();
  const window = useMemo(() => timeWindow(range, minuteEndMs), [range, minuteEndMs]);
  const histogram = useTraceHistogram(accessToken, { window, q: query }, isActive);
  const shownRange = zoom ?? window;
  const runs = traces.traces;
  const runRefs = useMemo(() => runs.map(traceRefOf), [runs]);

  const changeHours = (hours: number, apply: (hours: number) => void) => {
    setZoom(null);
    setRangeChanged(true);
    apply(hours);
  };

  const openSentTrace = (trace: TraceSummary) => {
    setConnecting(false);
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

  if (setup.disabledDetail != null) return <SetupPage detail={setup.disabledDetail} {...setupProps} />;
  // Onboarding only on the first, default view; an empty range the user picked keeps its controls.
  if (checkHistory && !history.error && history.data === false) return <SetupPage detail={null} {...setupProps} />;
  if (connecting) return <SetupPage detail={null} connected onBack={() => setConnecting(false)} {...setupProps} />;

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
        <RunsToolbar
          query={query}
          onQueryChange={setQuery}
          runs={traces.traces}
          range={shownRange}
          order={order}
          busy={traces.isPlaceholder}
        >
          {onInvestigate && (
            <InvestigateButton query={query} startMs={shownRange.startMs} onInvestigate={onInvestigate} />
          )}
          {timeControls && (
            <TimeRangeControls
              range={range}
              zoom={zoom}
              onHoursChange={(hours) => changeHours(hours, timeControls.setHours)}
              showLive={live}
              onLiveChange={timeControls.setLive}
            />
          )}
        </RunsToolbar>
        <Timeline
          buckets={histogram.buckets}
          loading={histogram.isLoading}
          range={window}
          selection={zoom}
          onSelect={setZoom}
          noun={RUN_NOUN}
        />
        <AgentTracesTable
          traces={runs}
          isLoading={traces.isLoading || (checkHistory && history.isLoading)}
          error={traces.error}
          hasMore={traces.hasMore}
          isFetching={traces.isFetching}
          isPlaceholder={traces.isPlaceholder}
          order={order}
          onOrderChange={setOrder}
          onRetry={traces.hasMore ? traces.loadMore : traces.refetch}
          onLoadMore={traces.loadMore}
          rangeEmpty={traces.traces.length === 0}
          onSetUpTracing={() => setConnecting(true)}
        />
      </div>
    </Inspector.Root>
  );
}

interface InvestigateButtonProps {
  readonly query: string;
  readonly startMs: number;
  readonly onInvestigate: (scope: InvestigateScope) => void;
}

function InvestigateButton({ query, startMs, onInvestigate }: InvestigateButtonProps) {
  const q = query.trim();
  if (!q) return null;
  const lookbackHours = () => Math.max(1, Math.ceil((Date.now() - startMs) / HOUR_MS));
  return (
    <button
      type="button"
      onClick={() => onInvestigate({ q, lookbackHours: lookbackHours() })}
      className="flex items-center gap-1.5 border-l border-border px-3 text-sm font-medium whitespace-nowrap text-foreground transition-colors outline-none hover:bg-muted/60 focus-visible:bg-muted/60"
    >
      <ScanSearch aria-hidden="true" className="size-3.5 text-muted-foreground" />
      Investigate these runs
    </button>
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

function SetupPage({ onBack, ...props }: ComponentProps<typeof TracingSetupCard> & { onBack?: () => void }) {
  return (
    <div className="p-6">
      {onBack && (
        <button
          type="button"
          onClick={onBack}
          className="mb-4 inline-flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground"
        >
          <ArrowLeft aria-hidden="true" className="size-3.5" /> Back to traces
        </button>
      )}
      <TracingSetupCard {...props} />
    </div>
  );
}
