"use client";

import { useState } from "react";

import { Button } from "@/components/ui/button";
import { ButtonGroup } from "@/components/ui/button-group";

import { AgentTracesTable } from "./AgentTracesTable";
import { TraceDrawer } from "./TraceDrawer";
import { TracingSetupCard } from "./TracingSetupCard";
import { useAgentTraces } from "./useAgentTraces";

export type LogsView = "all" | "traces" | "llm";

const VIEWS: { id: LogsView; label: string }[] = [
  { id: "all", label: "All" },
  { id: "traces", label: "Agent traces" },
  { id: "llm", label: "LLM requests" },
];

export function LogsViewSwitch({ value, onChange }: { value: LogsView; onChange: (view: LogsView) => void }) {
  return (
    <ButtonGroup aria-label="Log type">
      {VIEWS.map((view) => (
        <Button
          key={view.id}
          size="sm"
          variant={value === view.id ? "secondary" : "outline"}
          aria-pressed={value === view.id}
          onClick={() => onChange(view.id)}
        >
          {view.label}
        </Button>
      ))}
    </ButtonGroup>
  );
}

interface AgentTracesSectionProps {
  view: LogsView;
  accessToken: string;
  isActive: boolean;
  startTime: string;
  endTime: string;
  isCustomDate: boolean;
  isLiveTail: boolean;
  /** Open the request-log drawer for a LiteLLM request id (from a span's LiteLLM request card). */
  onOpenRequestLog: (requestId: string) => void;
}

/**
 * Agent trace rows for the Logs page. In "Agent traces" it is the whole list (or the
 * setup card on 501); in "All" the current trace page sits above the request logs.
 */
export function AgentTracesSection({
  view,
  accessToken,
  isActive,
  startTime,
  endTime,
  isCustomDate,
  isLiveTail,
  onOpenRequestLog,
}: AgentTracesSectionProps) {
  const [openTrace, setOpenTrace] = useState<{ traceId: string; spanId: string | null } | null>(null);
  const traces = useAgentTraces({
    accessToken,
    startTime,
    endTime,
    isCustomDate,
    isLiveTail,
    enabled: isActive && view !== "llm",
  });

  if (view === "llm") return null;

  const handleOpenRequestLog = (requestId: string) => {
    setOpenTrace(null);
    onOpenRequestLog(requestId);
  };

  return (
    <>
      {view === "traces" && traces.notEnabledDetail !== null && <TracingSetupCard detail={traces.notEnabledDetail} />}
      {traces.notEnabledDetail === null && (
        <AgentTracesTable
          accessToken={accessToken}
          traces={traces.traces}
          isLoading={traces.isLoading}
          error={traces.error}
          hasMore={traces.hasMore}
          onLoadMore={traces.loadMore}
          onOpenTrace={(traceId, spanId) => setOpenTrace({ traceId, spanId: spanId ?? null })}
          compact={view === "all"}
        />
      )}
      <TraceDrawer
        open={openTrace !== null}
        traceId={openTrace?.traceId ?? null}
        initialSpanId={openTrace?.spanId}
        accessToken={accessToken}
        onClose={() => setOpenTrace(null)}
        onOpenRequestLog={handleOpenRequestLog}
      />
    </>
  );
}
