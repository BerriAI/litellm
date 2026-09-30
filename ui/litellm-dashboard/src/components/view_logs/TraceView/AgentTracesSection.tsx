"use client";

import { Search } from "lucide-react";
import { useMemo, useState, type ReactNode } from "react";

import { Input } from "@/components/ui/input";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { cn } from "@/lib/cva.config";

import { AgentTracesTable } from "./AgentTracesTable";
import { TraceDrawer } from "./TraceDrawer";
import { TracingSetupCard } from "./TracingSetupCard";
import type { TraceSummary } from "./traceTypes";
import { previewText } from "./traceUtils";
import type { AgentTracesResult } from "./useAgentTraces";

export type LogsView = "runs" | "requests";

export const LOGS_VIEW_STORAGE_KEY = "logsView";

const VIEWS: { id: LogsView; label: string }[] = [
  { id: "runs", label: "Runs" },
  { id: "requests", label: "Requests" },
];

export const readStoredLogsView = (): LogsView | null => {
  try {
    const stored = sessionStorage.getItem(LOGS_VIEW_STORAGE_KEY);
    return stored === "runs" || stored === "requests" ? stored : null;
  } catch {
    return null;
  }
};

export const storeLogsView = (view: LogsView): void => {
  try {
    sessionStorage.setItem(LOGS_VIEW_STORAGE_KEY, view);
  } catch {
    /* storage unavailable: the choice just isn't remembered */
  }
};

/**
 * The view to show: the remembered choice, else Runs when there is at least one run,
 * else Requests. `null` while the first page of runs is still loading.
 */
export function resolveLogsView(
  stored: LogsView | null,
  traces: Pick<AgentTracesResult, "isLoading" | "traces" | "notEnabledDetail">,
): LogsView | null {
  if (traces.notEnabledDetail !== null) return "requests";
  if (stored) return stored;
  if (traces.isLoading) return null;
  return traces.traces.length > 0 ? "runs" : "requests";
}

/** Quiet segmented control. Runs is hidden when tracing isn't enabled on the proxy. */
export function LogsViewSwitch({
  value,
  onChange,
  showRuns,
}: {
  value: LogsView | null;
  onChange: (view: LogsView) => void;
  showRuns: boolean;
}) {
  const views = showRuns ? VIEWS : VIEWS.filter((v) => v.id !== "runs");
  if (views.length < 2) return null;
  return (
    <div role="group" aria-label="Log type" className="inline-flex rounded-md bg-muted p-0.5">
      {views.map((view) => (
        <button
          key={view.id}
          type="button"
          aria-pressed={value === view.id}
          onClick={() => onChange(view.id)}
          className={cn(
            "h-7 rounded-[5px] px-3 text-[13px] text-muted-foreground transition-colors hover:text-foreground",
            value === view.id && "bg-background font-medium text-foreground shadow-xs",
          )}
        >
          {view.label}
        </button>
      ))}
    </div>
  );
}

/** Header link shown instead of the Runs segment when tracing is off. */
export function TracingSetupLink({ detail }: { detail: string }) {
  return (
    <Popover>
      <PopoverTrigger
        render={
          <button type="button" className="text-[13px] text-muted-foreground underline-offset-4 hover:underline" />
        }
      >
        Set up agent tracing
      </PopoverTrigger>
      <PopoverContent align="end" className="w-[440px] p-5">
        <TracingSetupCard detail={detail} />
      </PopoverContent>
    </Popover>
  );
}

/** Client-side search over the loaded runs: input text, agent name, service or trace id. */
export function filterRuns(traces: readonly TraceSummary[], query: string): TraceSummary[] {
  const needle = query.trim().toLowerCase();
  if (!needle) return [...traces];
  return traces.filter((t) =>
    [previewText(t.input_preview), t.name, t.service, t.trace_id].some((field) => field.toLowerCase().includes(needle)),
  );
}

interface AgentTracesSectionProps {
  traces: AgentTracesResult;
  accessToken: string;
  /** The page's time range / live tail controls, shared with Requests. */
  toolbar: ReactNode;
  /** Open the request-log drawer for a LiteLLM request id (from a step's Request tab). */
  onOpenRequestLog: (requestId: string) => void;
}

/** The Runs view: search + page toolbar, the runs table and the trace drawer. */
export function AgentTracesSection({ traces, accessToken, toolbar, onOpenRequestLog }: AgentTracesSectionProps) {
  const [openTraceId, setOpenTraceId] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const visible = useMemo(() => filterRuns(traces.traces, search), [traces.traces, search]);

  const handleOpenRequestLog = (requestId: string) => {
    setOpenTraceId(null);
    onOpenRequestLog(requestId);
  };

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <div className="relative w-72">
          <Search className="pointer-events-none absolute top-1/2 left-2.5 size-3.5 -translate-y-1/2 text-muted-foreground" />
          <Input
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="Search runs by input or trace ID…"
            aria-label="Search runs"
            className="h-8 pl-8 text-[13px]"
          />
        </div>
        {toolbar}
      </div>
      <AgentTracesTable
        traces={visible}
        isLoading={traces.isLoading}
        error={traces.error}
        hasMore={traces.hasMore}
        onLoadMore={traces.loadMore}
        onOpenTrace={setOpenTraceId}
        filtered={search.trim() !== ""}
      />
      <TraceDrawer
        open={openTraceId !== null}
        traceId={openTraceId}
        accessToken={accessToken}
        onClose={() => setOpenTraceId(null)}
        onOpenRequestLog={handleOpenRequestLog}
      />
    </div>
  );
}
