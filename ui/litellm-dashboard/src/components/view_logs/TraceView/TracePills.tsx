"use client";

import { cn } from "@/lib/cva.config";

import type { SpanStatus, SpanType } from "./traceTypes";

/** Pill colours per span type: agent violet, llm blue, tool amber, chain/framework slate (matches TypeBadges). */
export const SPAN_PILL_CLASS: Record<SpanType, string> = {
  agent:
    "bg-violet-50 text-violet-700 border-violet-200 dark:bg-violet-950 dark:text-violet-300 dark:border-violet-800",
  llm: "bg-info/10 text-info border-info/20",
  tool: "bg-amber-50 text-amber-700 border-amber-200 dark:bg-amber-950 dark:text-amber-300 dark:border-amber-800",
  chain: "bg-slate-100 text-slate-600 border-slate-200 dark:bg-slate-800 dark:text-slate-300 dark:border-slate-700",
  framework: "bg-slate-100 text-slate-600 border-slate-200 dark:bg-slate-800 dark:text-slate-300 dark:border-slate-700",
};

/** Waterfall bar fill per span type. */
export const SPAN_BAR_CLASS: Record<SpanType, string> = {
  agent: "bg-violet-500",
  llm: "bg-info",
  tool: "bg-amber-500",
  chain: "bg-slate-400",
  framework: "bg-slate-300 dark:bg-slate-600",
};

const PILL_BASE =
  "inline-flex shrink-0 items-center gap-1 whitespace-nowrap rounded-full border px-2 py-px text-[11px] font-medium";

export function SpanTypePill({ type, className }: { type: SpanType; className?: string }) {
  return <span className={cn(PILL_BASE, SPAN_PILL_CLASS[type], className)}>{type === "llm" ? "LLM" : type}</span>;
}

/** The violet "◆ 2 agents · 7 LLM · 26 tool" badge used on agent trace rows. */
export function AgentTracePill({ label }: { label: string }) {
  return <span className={cn(PILL_BASE, SPAN_PILL_CLASS.agent)}>{label}</span>;
}

const STATUS_CLASS: Record<SpanStatus, string> = {
  ok: "border-success/20 bg-success/10 text-success",
  error: "border-destructive/20 bg-destructive/10 text-destructive",
  unset: "border-border bg-muted text-muted-foreground",
};

const STATUS_LABEL: Record<SpanStatus, string> = { ok: "Success", error: "Failure", unset: "Unset" };

export function SpanStatusBadge({ status, compact = false }: { status: SpanStatus; compact?: boolean }) {
  return (
    <span
      className={cn(
        "inline-flex items-center justify-center rounded-md border px-1.5 py-0.5 text-[11px] font-medium",
        compact ? "" : "min-w-[58px]",
        STATUS_CLASS[status],
      )}
    >
      {compact && status === "ok" ? "OK" : STATUS_LABEL[status]}
    </span>
  );
}
