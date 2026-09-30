"use client";

import { PanelRightClose } from "lucide-react";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/cva.config";

import { AttributesDetail } from "./AttributesDetail";
import { CopyButton } from "./CopyButton";
import { DetailContent, errorHeadline, useSpanDetail } from "./DetailContent";
import { RequestDetail } from "./RequestDetail";
import { agentHandoffText } from "./TraceDrawer";
import type { GroupRowData, TreeRow } from "./traceTree";
import type { Span, Trace } from "./traceTypes";
import { fmtMs, fmtTok } from "./traceUtils";

interface DetailPaneProps {
  trace: Trace;
  row: TreeRow | undefined;
  accessToken: string;
  onClose: () => void;
}

type Tab = "content" | "request" | "attributes";

const TABS: { id: Tab; label: string }[] = [
  { id: "content", label: "Content" },
  { id: "request", label: "Request" },
  { id: "attributes", label: "Attributes" },
];

function PaneHeader({ children, onClose }: { children: React.ReactNode; onClose: () => void }) {
  return (
    <div className="flex h-10 shrink-0 items-center gap-2 border-b border-border bg-card px-3">
      {children}
      <Button variant="ghost" size="icon-xs" onClick={onClose} aria-label="Close details" className="ml-auto">
        <PanelRightClose className="size-3.5" />
      </Button>
    </div>
  );
}

function PaneFooter({ children }: { children: React.ReactNode }) {
  return <div className="flex h-9 shrink-0 items-center gap-2 border-t border-border bg-card px-3">{children}</div>;
}

function Meta({ label, value }: { label: string; value: string }) {
  return (
    <span>
      <span className="text-muted-foreground/70">{label}=</span>
      {value}
    </span>
  );
}

function SpanPane({
  trace,
  span,
  accessToken,
  onClose,
}: {
  trace: Trace;
  span: Span;
  accessToken: string;
  onClose: () => void;
}) {
  const [tab, setTab] = useState<Tab>("content");
  const traceId = trace.summary.trace_id;
  const detailQuery = useSpanDetail(accessToken, traceId, tab === "attributes" ? span.span_id : null);
  const tokens = span.input_tokens + span.output_tokens;
  return (
    <aside className="flex h-full min-w-0 flex-col bg-background" aria-label="Span details">
      <PaneHeader onClose={onClose}>
        <span
          className={cn("size-1.5 rounded-full", span.status === "error" ? "bg-destructive" : "bg-muted-foreground/60")}
        />
        <div className="min-w-0 flex-1">
          <div className="truncate font-mono text-[12px] font-medium text-foreground">
            {span.type === "llm" ? span.model || span.name : span.name}
          </div>
          <div className="truncate font-mono text-[9px] text-muted-foreground">{span.span_id}</div>
        </div>
        <CopyButton value={span.span_id} label="Copy span ID" iconOnly />
      </PaneHeader>
      <div role="tablist" className="flex h-8 shrink-0 items-end gap-1 border-b border-border bg-card px-3">
        {TABS.map((t) => (
          <button
            key={t.id}
            type="button"
            role="tab"
            aria-selected={tab === t.id}
            onClick={() => setTab(t.id)}
            className={cn(
              "-mb-px h-8 border-b-2 px-2 font-mono text-[10px]",
              tab === t.id
                ? "border-foreground text-foreground"
                : "border-transparent text-muted-foreground hover:text-foreground",
            )}
          >
            {t.label}
          </button>
        ))}
      </div>
      <div className="min-h-0 flex-1 overflow-auto">
        {tab === "content" && <DetailContent accessToken={accessToken} traceId={traceId} span={span} />}
        {tab === "request" && (
          <RequestDetail span={span} accessToken={accessToken} traceStartMs={Date.parse(trace.summary.start_time)} />
        )}
        {tab === "attributes" && (
          <AttributesDetail
            traceId={traceId}
            span={span}
            attributes={detailQuery.data?.attributes}
            isLoading={detailQuery.isLoading}
          />
        )}
      </div>
      <PaneFooter>
        <CopyButton value={agentHandoffText(traceId, span.span_id)} label="Copy step" copiedLabel="Command copied" />
        <div className="ml-auto flex items-center gap-3 font-mono text-[10px] tabular-nums text-muted-foreground">
          <Meta label="time" value={fmtMs(span.duration_ms)} />
          {tokens > 0 && <Meta label="tokens" value={fmtTok(tokens)} />}
        </div>
      </PaneFooter>
    </aside>
  );
}

function GroupMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="border-r border-b border-border p-3">
      <div className="font-mono text-[9px] tracking-[0.1em] text-muted-foreground uppercase">{label}</div>
      <div className="mt-1 font-mono text-[12px] tabular-nums text-foreground">{value}</div>
    </div>
  );
}

/** ×N group: rollup of every invocation plus the first failure's message. */
function GroupPane({ trace, row, onClose }: { trace: Trace; row: GroupRowData; onClose: () => void }) {
  const tokens = row.members.reduce((sum, m) => sum + m.input_tokens + m.output_tokens, 0);
  const firstFailure = row.members.find((m) => m.status === "error" && m.error);
  return (
    <aside className="flex h-full min-w-0 flex-col bg-background" aria-label="Group details">
      <PaneHeader onClose={onClose}>
        <span className={cn("size-1.5 rounded-full", row.failedCount ? "bg-destructive" : "bg-muted-foreground/60")} />
        <span className="truncate font-mono text-[12px] font-medium text-foreground">
          {row.name} <span className="text-muted-foreground">×{row.members.length}</span>
        </span>
      </PaneHeader>
      <div className="min-h-0 flex-1 overflow-auto p-3">
        <div className="grid grid-cols-2 overflow-hidden rounded border border-border bg-card">
          <GroupMetric label="Invocations" value={row.members.length.toLocaleString()} />
          <GroupMetric label="Failed" value={row.failedCount.toLocaleString()} />
          <GroupMetric label="p50 latency" value={fmtMs(row.p50Duration)} />
          <GroupMetric label="Tokens" value={fmtTok(tokens)} />
          <GroupMetric label="Agent" value={row.agent || "—"} />
          <GroupMetric label="Type" value={row.type} />
        </div>
        {firstFailure?.error && (
          <section className="mt-3 rounded border border-destructive/30 p-3">
            <div className="font-mono text-[9px] tracking-[0.1em] text-destructive uppercase">Failure pattern</div>
            <p className="mt-2 font-mono text-[11px] leading-5 text-foreground">{errorHeadline(firstFailure.error)}</p>
          </section>
        )}
      </div>
      <PaneFooter>
        <CopyButton
          value={agentHandoffText(trace.summary.trace_id, (firstFailure ?? row.members[0]).span_id)}
          label="Copy group sample"
        />
      </PaneFooter>
    </aside>
  );
}

/** Right pane of the run view: switches on the selected tree row. */
export function DetailPane({ trace, row, accessToken, onClose }: DetailPaneProps) {
  if (!row || row.kind === "load-more") {
    return (
      <div className="grid h-full place-items-center bg-background font-mono text-[11px] text-muted-foreground">
        Select a span to inspect it.
      </div>
    );
  }
  if (row.kind === "group") return <GroupPane key={row.id} trace={trace} row={row} onClose={onClose} />;
  return <SpanPane key={row.id} trace={trace} span={row.span} accessToken={accessToken} onClose={onClose} />;
}
