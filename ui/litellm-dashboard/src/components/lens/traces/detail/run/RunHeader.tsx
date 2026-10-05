"use client";

import { ArrowLeft, Check, Copy } from "lucide-react";
import { useState } from "react";
import { useTimeout } from "usehooks-ts";

import { Button } from "@/components/ui/button";
import { TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/cva.config";
import { copyToClipboard } from "@/utils/dataUtils";

import type { TraceHandoff } from "../../api";
import { formatCost } from "../../list/AgentTracesTable";
import { IdChip } from "../../ui/IdChip";
import { SpanIcon } from "../../ui/SpanIcon";
import { FrameworkLogo, traceFramework } from "../../ui/TraceFramework";
import type { Trace } from "../../types";
import { fmtMs, fmtTok, traceAgentNames, traceDisplayName } from "../../utils";

function CopyForAgent({ handoff }: { handoff: TraceHandoff }) {
  const [copied, setCopied] = useState(false);
  useTimeout(() => setCopied(false), copied ? 1600 : null);
  return (
    <Button
      variant="outline"
      size="xs"
      className="h-7 shrink-0 gap-1.5 text-xs shadow-none"
      onClick={async () => setCopied(await copyToClipboard(handoff.text, handoff.copied))}
    >
      {copied ? <Check className="size-3" /> : <Copy className="size-3" />}
      {copied ? "Copied" : "Copy for agent"}
    </Button>
  );
}

function Stat({ label, value, error = false }: { label: string; value: string; error?: boolean }) {
  return (
    <span className={cn("inline-flex items-center gap-1 tabular-nums", error && "text-destructive")}>
      <span>{label}</span> <span className={cn("font-medium", !error && "text-foreground")}>{value}</span>
    </span>
  );
}

function StatusPill({ failed }: { failed: boolean }) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 font-medium",
        failed ? "bg-destructive/10 text-destructive" : "bg-trace-ok text-trace-ok-glyph",
      )}
    >
      <span className={cn("size-1.5 rounded-full", failed ? "bg-destructive" : "bg-trace-ok-glyph")} />
      {failed ? "Failed" : "Completed"}
    </span>
  );
}

function RunIcon({ summary, failed }: { summary: Trace["summary"]; failed: boolean }) {
  const framework = traceFramework(summary);
  if (!framework) return <SpanIcon type="agent" error={failed} size="lg" />;
  return (
    <span
      className="inline-flex h-6 shrink-0 items-center gap-1.5 rounded-md border border-border px-1.5 text-xs text-muted-foreground"
      data-testid="run-framework"
      title={framework.label}
    >
      <FrameworkLogo framework={framework} />
      {traceAgentNames(summary).join(", ") || framework.label}
    </span>
  );
}

interface RunHeaderProps {
  trace: Trace;
  handoff: TraceHandoff;
  onBack: () => void;
  embedded: boolean;
}

/** Run identity, view switch and totals in two tight rows. */
export function RunHeader({ trace, handoff, onBack, embedded }: RunHeaderProps) {
  const { summary } = trace;
  const failed = summary.status === "error";
  return (
    <header className="flex shrink-0 flex-col gap-2 border-b bg-background px-4 pt-3 pb-2.5">
      <div className="flex min-w-0 items-center gap-2">
        {!embedded && (
          <Button variant="ghost" size="icon-xs" onClick={onBack} aria-label="Back to runs">
            <ArrowLeft className="size-4" />
          </Button>
        )}
        <h1 className="min-w-0 truncate text-base font-semibold">{traceDisplayName(summary)}</h1>
        <IdChip value={summary.trace_id} label="Copy trace ID" />
        <RunIcon summary={summary} failed={failed} />
        <div className="ml-auto flex shrink-0 items-center gap-2">
          <TabsList aria-label="Trace view" className="group-data-horizontal/tabs:h-7">
            <TabsTrigger value="steps" className="px-2.5 text-xs">
              Steps
            </TabsTrigger>
            <TabsTrigger value="conversation" className="px-2.5 text-xs">
              Conversation
            </TabsTrigger>
          </TabsList>
          <CopyForAgent handoff={handoff} />
        </div>
      </div>
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 text-xs text-muted-foreground">
        <StatusPill failed={failed} />
        <Stat label="Duration" value={fmtMs(summary.duration_ms)} />
        <Stat label="Steps" value={summary.span_count.toLocaleString()} />
        <Stat label="Tokens" value={fmtTok(summary.input_tokens + summary.output_tokens)} />
        <Stat label="Cost" value={summary.spend == null ? "Not reported" : formatCost(summary.spend)} />
        {summary.error_count > 0 && <Stat label="Step errors" value={summary.error_count.toLocaleString()} error />}
      </div>
    </header>
  );
}
