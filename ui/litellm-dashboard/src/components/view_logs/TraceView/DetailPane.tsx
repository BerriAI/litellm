"use client";

import { PanelRightClose } from "lucide-react";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/cva.config";

import { AttributesDetail } from "./AttributesDetail";
import { CopyButton } from "./CopyButton";
import { DetailContent, errorHeadline, useSpanDetail } from "./DetailContent";
import { IdChip } from "./IdChip";
import { RequestDetail } from "./RequestDetail";
import { SpanIcon } from "./SpanIcon";
import { agentHandoffText } from "./TraceDrawer";
import type { GroupRowData, TreeRow } from "./traceTree";
import type { Span, SpanType, Trace } from "./traceTypes";
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

function PaneHeader({
  type,
  model,
  failed,
  title,
  idValue,
  onClose,
}: {
  type: SpanType;
  model: string | null;
  failed: boolean;
  title: React.ReactNode;
  idValue?: string;
  onClose: () => void;
}) {
  return (
    <div className="flex h-6 shrink-0 items-center justify-between gap-4 px-7">
      <div className="flex min-w-0 items-center">
        <span className="mr-2 shrink-0">
          <SpanIcon type={type} model={model} error={failed} size="md" />
        </span>
        <h2 className="min-w-0 truncate text-[14px] leading-5 font-medium tracking-[-0.56px] text-trace-text">
          {title}
        </h2>
        {idValue && (
          <span className="ml-2 flex shrink-0">
            <IdChip value={idValue} label="Copy span ID" />
          </span>
        )}
      </div>
      <Button
        variant="ghost"
        size="icon-xs"
        onClick={onClose}
        aria-label="Close details"
        className="size-6 shrink-0 rounded-[4px] text-trace-text-2"
      >
        <PanelRightClose className="size-4" />
      </Button>
    </div>
  );
}

function PaneFooter({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex h-10 shrink-0 items-center gap-2 border-t border-trace-border bg-trace-surface px-7">
      {children}
    </div>
  );
}

function Meta({ label, value }: { label: string; value: string }) {
  return (
    <span>
      <span className="text-trace-key">{label} </span>
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
  const detailQuery = useSpanDetail(
    accessToken,
    traceId,
    tab === "attributes" ? span.span_id : null,
    trace.summary.trace_ref,
  );
  const tokens = span.input_tokens + span.output_tokens;
  return (
    <aside
      className="flex h-full min-w-0 animate-view-fade-in flex-col bg-trace-surface pt-4 text-[13px] text-trace-text motion-reduce:animate-none"
      aria-label="Span details"
    >
      <PaneHeader
        type={span.type}
        model={span.model}
        failed={span.status === "error"}
        title={span.type === "llm" ? span.model || span.name : span.name}
        idValue={span.span_id}
        onClose={onClose}
      />
      <div role="tablist" aria-label="Span detail sections" className="flex shrink-0 items-center gap-2 py-2 pr-2 pl-6">
        {TABS.map((t) => (
          <button
            key={t.id}
            type="button"
            role="tab"
            aria-selected={tab === t.id}
            onClick={() => setTab(t.id)}
            className={cn(
              "rounded-[4px] px-2 py-1 text-[13px] leading-[18px] font-medium transition-colors duration-100 focus-visible:outline-2 focus-visible:outline-trace-brand motion-reduce:transition-none",
              tab === t.id
                ? "bg-trace-tab-active text-trace-text hover:bg-trace-tab-hover"
                : "text-trace-key hover:bg-trace-tab-active",
            )}
          >
            {t.label}
          </button>
        ))}
      </div>
      <div key={tab} role="tabpanel" className="min-h-0 flex-1 overflow-auto">
        {tab === "content" && (
          <DetailContent accessToken={accessToken} traceId={traceId} traceRef={trace.summary.trace_ref} span={span} />
        )}
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
        <CopyButton
          value={agentHandoffText(traceId, span.span_id, trace.summary.trace_ref)}
          label="Copy step"
          copiedLabel="Command copied"
        />
        <div className="ml-auto flex items-center gap-3 text-[13px] text-trace-duration tabular-nums">
          <Meta label="time" value={fmtMs(span.duration_ms)} />
          {tokens > 0 && <Meta label="tokens" value={fmtTok(tokens)} />}
        </div>
      </PaneFooter>
    </aside>
  );
}

function GroupMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="border-r border-b border-trace-card-border p-3">
      <div className="text-[13px] font-medium text-trace-key">{label}</div>
      <div className="mt-1 text-[13px] text-trace-text tabular-nums">{value}</div>
    </div>
  );
}

/** ×N group: rollup of every invocation plus the first failure's message. */
function GroupPane({ trace, row, onClose }: { trace: Trace; row: GroupRowData; onClose: () => void }) {
  const tokens = row.members.reduce((sum, m) => sum + m.input_tokens + m.output_tokens, 0);
  const firstFailure = row.members.find((m) => m.status === "error" && m.error);
  return (
    <aside
      className="flex h-full min-w-0 animate-view-fade-in flex-col bg-trace-surface pt-4 text-[13px] text-trace-text motion-reduce:animate-none"
      aria-label="Group details"
    >
      <PaneHeader
        type={row.type}
        model={row.members[0]?.model ?? null}
        failed={row.failedCount > 0}
        title={
          <>
            {row.name} <span className="text-trace-duration">×{row.members.length}</span>
          </>
        }
        onClose={onClose}
      />
      <div className="min-h-0 flex-1 overflow-auto px-7 py-3">
        <div className="grid grid-cols-2 overflow-hidden rounded-[4px] border border-trace-card-border bg-trace-surface">
          <GroupMetric label="Invocations" value={row.members.length.toLocaleString()} />
          <GroupMetric label="Failed" value={row.failedCount.toLocaleString()} />
          <GroupMetric label="p50 latency" value={fmtMs(row.p50Duration)} />
          <GroupMetric label="Tokens" value={fmtTok(tokens)} />
          <GroupMetric label="Agent" value={row.agent || "—"} />
          <GroupMetric label="Type" value={row.type} />
        </div>
        {firstFailure?.error && (
          <section className="mt-3 rounded-[4px] border border-destructive/30 px-3.5 py-3">
            <div className="text-[12px] font-medium text-destructive">Failure pattern</div>
            <p className="mt-2 font-mono text-[12px] leading-5 text-foreground">{errorHeadline(firstFailure.error)}</p>
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
      <div className="grid h-full place-items-center bg-trace-surface text-[13px] text-trace-duration">
        Select a span to inspect it.
      </div>
    );
  }
  if (row.kind === "group") return <GroupPane key={row.id} trace={trace} row={row} onClose={onClose} />;
  return <SpanPane key={row.id} trace={trace} span={row.span} accessToken={accessToken} onClose={onClose} />;
}
