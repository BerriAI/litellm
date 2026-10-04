"use client";

import { PanelRightClose } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";

import { AttributesDetail } from "./AttributesDetail";
import { CopyButton } from "./CopyButton";
import { DetailContent, errorHeadline, useSpanDetail } from "./DetailContent";
import { IdChip } from "./IdChip";
import { PaneBar } from "./PaneBar";
import { RequestDetail } from "./RequestDetail";
import { SpanIcon } from "./SpanIcon";
import { useTracesApi } from "./tracesApi";
import { SPAN_TABS, type SpanTab } from "./traceRouting";
import type { GroupRowData, TreeRow } from "./traceTree";
import type { Span, SpanType, Trace } from "./traceTypes";
import { fmtMs, fmtTok } from "./traceUtils";

interface SpanTabProps {
  spanTab: SpanTab;
  onSpanTabChange: (tab: SpanTab) => void;
}

interface DetailPaneProps extends SpanTabProps {
  trace: Trace;
  row: TreeRow | undefined;
  accessToken: string;
  onClose: () => void;
}

const TAB_LABELS: Record<SpanTab, string> = { content: "Content", request: "Request", attributes: "Attributes" };

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
    <PaneBar className="justify-between gap-3">
      <div className="flex min-w-0 items-center">
        <span className="mr-2 shrink-0">
          <SpanIcon type={type} model={model} error={failed} size="md" />
        </span>
        <h2 className="min-w-0 truncate text-sm leading-5 font-semibold text-foreground">{title}</h2>
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
        className="size-6 shrink-0 rounded-sm text-muted-foreground"
      >
        <PanelRightClose className="size-4" />
      </Button>
    </PaneBar>
  );
}

function PaneFooter({ children }: { children: React.ReactNode }) {
  return <div className="flex h-10 shrink-0 items-center gap-3 border-t bg-background px-3">{children}</div>;
}

function Meta({ label, value }: { label: string; value: string }) {
  return (
    <span>
      <span className="text-muted-foreground">{label} </span>
      {value}
    </span>
  );
}

function SpanPane({
  trace,
  span,
  accessToken,
  spanTab: tab,
  onSpanTabChange,
  onClose,
}: SpanTabProps & {
  trace: Trace;
  span: Span;
  accessToken: string;
  onClose: () => void;
}) {
  const handoff = useTracesApi(accessToken).handoff(trace.summary.trace_id, span.span_id, trace.summary.trace_ref);
  const traceId = trace.summary.trace_id;
  const detailQuery = useSpanDetail(
    accessToken,
    traceId,
    tab === "attributes" ? span.span_id : null,
    trace.summary.trace_ref,
  );
  const tokens = span.input_tokens + span.output_tokens;
  return (
    <aside className="flex h-full min-w-0 flex-col bg-background text-sm text-foreground" aria-label="Span details">
      <PaneHeader
        type={span.type}
        model={span.model}
        failed={span.status === "error"}
        title={span.type === "llm" ? span.model || span.name : span.name}
        idValue={span.span_id}
        onClose={onClose}
      />
      <Tabs value={tab} onValueChange={(value) => onSpanTabChange(value as SpanTab)} className="min-h-0 flex-1 gap-0">
        <PaneBar>
          <TabsList variant="line" aria-label="Span detail sections" className="h-full gap-4 px-0">
            {SPAN_TABS.map((id) => (
              <TabsTrigger key={id} value={id} className="flex-none px-0 text-xs">
                {TAB_LABELS[id]}
              </TabsTrigger>
            ))}
          </TabsList>
        </PaneBar>
        <TabsContent value="content" className="min-h-0 overflow-auto pt-3">
          <DetailContent accessToken={accessToken} traceId={traceId} traceRef={trace.summary.trace_ref} span={span} />
        </TabsContent>
        <TabsContent value="request" className="min-h-0 overflow-auto pt-3">
          <RequestDetail span={span} accessToken={accessToken} traceStartMs={Date.parse(trace.summary.start_time)} />
        </TabsContent>
        <TabsContent value="attributes" className="min-h-0 overflow-auto pt-3">
          <AttributesDetail
            traceId={traceId}
            span={span}
            attributes={detailQuery.data?.attributes}
            isLoading={detailQuery.isLoading}
          />
        </TabsContent>
      </Tabs>
      <PaneFooter>
        <CopyButton value={handoff.text} label="Copy step" copiedLabel={handoff.copied} />
        <div className="ml-auto flex items-center gap-3 text-xs text-muted-foreground tabular-nums">
          <Meta label="time" value={fmtMs(span.duration_ms)} />
          {tokens > 0 && <Meta label="tokens" value={fmtTok(tokens)} />}
        </div>
      </PaneFooter>
    </aside>
  );
}

function GroupMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="border-b border-border/60 py-3">
      <div className="text-sm font-medium text-muted-foreground">{label}</div>
      <div className="mt-1 text-sm text-foreground tabular-nums">{value}</div>
    </div>
  );
}

/** ×N group: rollup of every invocation plus the first failure's message. */
function GroupPane({
  trace,
  row,
  accessToken,
  onClose,
}: {
  trace: Trace;
  row: GroupRowData;
  accessToken: string;
  onClose: () => void;
}) {
  const tokens = row.members.reduce((sum, m) => sum + m.input_tokens + m.output_tokens, 0);
  const firstFailure = row.members.find((m) => m.status === "error" && m.error);
  const sample = (firstFailure ?? row.members[0]).span_id;
  const handoff = useTracesApi(accessToken).handoff(trace.summary.trace_id, sample, trace.summary.trace_ref);
  return (
    <aside className="flex h-full min-w-0 flex-col bg-background text-sm text-foreground" aria-label="Group details">
      <PaneHeader
        type={row.type}
        model={row.members[0]?.model ?? null}
        failed={row.failedCount > 0}
        title={
          <>
            {row.name} <span className="text-muted-foreground">×{row.members.length}</span>
          </>
        }
        onClose={onClose}
      />
      <div className="min-h-0 flex-1 overflow-auto px-3 py-3">
        <div className="grid grid-cols-2 gap-x-5">
          <GroupMetric label="Invocations" value={row.members.length.toLocaleString()} />
          <GroupMetric label="Failed" value={row.failedCount.toLocaleString()} />
          <GroupMetric label="p50 latency" value={fmtMs(row.p50Duration)} />
          <GroupMetric label="Tokens" value={fmtTok(tokens)} />
          <GroupMetric label="Agent" value={row.agent || "—"} />
          <GroupMetric label="Type" value={row.type} />
        </div>
        {firstFailure?.error && (
          <section className="mt-3 rounded-sm border border-destructive/30 px-3.5 py-3">
            <div className="text-xs font-medium text-destructive">Failure pattern</div>
            <p className="mt-2 font-mono text-xs leading-5 text-foreground">{errorHeadline(firstFailure.error)}</p>
          </section>
        )}
      </div>
      <PaneFooter>
        <CopyButton value={handoff.text} label="Copy group sample" copiedLabel={handoff.copied} />
      </PaneFooter>
    </aside>
  );
}

/** Right pane of the run view: switches on the selected tree row. */
export function DetailPane({ trace, row, accessToken, spanTab, onSpanTabChange, onClose }: DetailPaneProps) {
  if (!row || row.kind === "load-more") {
    return (
      <div className="grid h-full place-items-center bg-background text-sm text-muted-foreground">
        Select a span to inspect it.
      </div>
    );
  }
  if (row.kind === "group")
    return <GroupPane key={row.id} trace={trace} row={row} accessToken={accessToken} onClose={onClose} />;
  return (
    <SpanPane
      key={row.id}
      trace={trace}
      span={row.span}
      accessToken={accessToken}
      spanTab={spanTab}
      onSpanTabChange={onSpanTabChange}
      onClose={onClose}
    />
  );
}
