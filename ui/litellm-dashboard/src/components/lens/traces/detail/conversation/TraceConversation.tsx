"use client";

import { useQueries } from "@tanstack/react-query";
import { useState } from "react";
import { ChevronRight, Wrench } from "lucide-react";
import { cn } from "@/lib/cva.config";
import CopyButton from "@/components/shared/CopyButton";
import { ToolArguments, ToolOutput } from "../content/ToolContent";
import { toolSummary } from "../content/payload";
import { useTracesApi } from "../../api";
import { Button } from "@/components/ui/button";
import {
  buildConversation,
  conversationSteps,
  conversationWarnings,
  groupConversation,
  CONVERSATION_PAGE_SIZE,
  type ConversationItem,
  type ConversationGroup,
} from "./conversation";
import { ErrorBlock } from "../content/SpanError";
import { Markdown } from "../content/Markdown";
import { ToolCallBlock, ToolResultCard } from "../content/Messages";
import type { SpanDetail, Trace, TraceMessage } from "../../types";
import { fmtMs } from "../../utils";

export function TraceConversation({
  trace,
  accessToken,
  onOpenStep,
}: {
  trace: Trace;
  accessToken: string;
  onOpenStep: (id: string) => void;
}) {
  const traces = useTracesApi(accessToken);
  const [limit, setLimit] = useState(CONVERSATION_PAGE_SIZE);
  const steps = conversationSteps(trace.spans);
  const visible = steps.slice(0, limit);
  const { trace_id: traceId, trace_ref: traceRef } = trace.summary;
  const queries = useQueries({
    queries: visible.map((span) => ({
      queryKey: ["agentTraceSpan", traceId, traceRef, span.span_id, accessToken],
      queryFn: (): Promise<SpanDetail> => traces.span(traceId, span.span_id, traceRef),
      staleTime: Infinity,
      retry: false,
    })),
  });
  const loading = queries.some((query) => query.isPending);
  const failed = queries.some((query) => query.isError);
  const unresolvedIndex = queries.findIndex((query) => !query.isSuccess);
  const loadedCount = unresolvedIndex < 0 ? queries.length : unresolvedIndex;
  const details = new Map(
    queries.slice(0, loadedCount).map((query, index) => [visible[index].span_id, query.data!] as const),
  );
  const complete = loadedCount === steps.length;
  const traceComplete = complete && !trace.next_cursor;
  const items = buildConversation(trace.spans, details, complete);
  const warnings = conversationWarnings(details, traceComplete);
  const multipleAgents = new Set(items.map((item) => item.agentId).filter(Boolean)).size > 1;
  const inlineErrorIds = new Set(items.filter((item) => item.showError).map((item) => item.span.span_id));
  const rootErrors = trace.spans.filter((span) => {
    const failedRoot = span.parent_span_id === null && span.status === "error" && span.type !== "tool";
    return failedRoot && !inlineErrorIds.has(span.span_id);
  });
  return (
    <section aria-label="Trace conversation" className="min-h-0 flex-1 overflow-y-auto">
      <div className="mx-auto min-w-0 max-w-3xl space-y-5 px-3 py-4 sm:px-6 sm:py-5">
        {rootErrors.map((span) => (
          <ErrorBlock key={span.span_id} span={span} />
        ))}
        {warnings.map((warning) => (
          <p key={warning} role="status" className="rounded-md border p-3 text-sm text-muted-foreground">
            {warning}
          </p>
        ))}
        <ConversationGroups
          groups={groupConversation(items, trace.spans)}
          multipleAgents={multipleAgents}
          onOpenStep={onOpenStep}
        />
        {queries.map(
          (query, index) =>
            query.isError && (
              <div key={visible[index].span_id} role="alert" className="rounded-md border p-3 text-sm">
                Could not load {visible[index].name}. Retry this step to continue the conversation.
                <Button variant="outline" size="sm" className="mt-2" onClick={() => query.refetch()}>
                  Retry step
                </Button>
              </div>
            ),
        )}
        {loading && (
          <p role="status" className="py-4 text-sm text-muted-foreground">
            Loading conversation…
          </p>
        )}
        {traceComplete && items.length === 0 && (
          <p className="text-sm text-muted-foreground">No conversation content recorded.</p>
        )}
        <div className="flex items-center justify-between gap-3 border-t pt-4 text-xs text-muted-foreground">
          <span>{traceComplete ? "End of conversation" : `${loadedCount} of ${steps.length} steps loaded`}</span>
          {visible.length < steps.length && (
            <Button
              variant="outline"
              size="sm"
              disabled={loading || failed}
              onClick={() => setLimit((current) => current + CONVERSATION_PAGE_SIZE)}
            >
              Load next {Math.min(CONVERSATION_PAGE_SIZE, steps.length - visible.length)} steps
            </Button>
          )}
        </div>
      </div>
    </section>
  );
}

function ConversationGroups({
  groups,
  multipleAgents,
  onOpenStep,
}: {
  groups: ConversationGroup[];
  multipleAgents: boolean;
  onOpenStep: (id: string) => void;
}) {
  return (
    <>
      {groups.map((group) =>
        group.kind === "item" ? (
          <ConversationStep
            key={group.item.id}
            item={group.item}
            multipleAgents={multipleAgents}
            onOpenStep={onOpenStep}
          />
        ) : (
          <details key={group.id} className="min-w-0 rounded-md border p-3">
            <summary className="cursor-pointer text-sm font-medium">Subagent: {group.name}</summary>
            <div className="mt-4 min-w-0 space-y-5 border-l pl-3">
              <ConversationGroups groups={group.children} multipleAgents={multipleAgents} onOpenStep={onOpenStep} />
            </div>
          </details>
        ),
      )}
    </>
  );
}

function ConversationStep({
  item,
  multipleAgents,
  onOpenStep,
}: {
  item: ConversationItem;
  multipleAgents: boolean;
  onOpenStep: (id: string) => void;
}) {
  return (
    <section className="min-w-0 space-y-2" aria-label={`Conversation step ${item.span.name}`}>
      {(multipleAgents || item.toolResult === undefined) && (
        <div className="flex min-w-0 items-center justify-between gap-2 text-xs text-muted-foreground">
          <div className="flex min-w-0 items-center gap-2">
            {multipleAgents && (
              <span className="truncate" title={item.agentName}>
                {item.agentName}
              </span>
            )}
            {item.model && (
              <span className="truncate" title={item.model}>
                {item.model}
              </span>
            )}
            <span className="shrink-0 tabular-nums">{fmtMs(item.time ?? item.span.start_offset_ms)}</span>
          </div>
          {item.toolResult === undefined && (
            <Button
              variant="ghost"
              size="xs"
              aria-label={`Inspect step ${item.span.name}`}
              onClick={() => onOpenStep(item.span.span_id)}
              className="shrink-0 text-muted-foreground"
            >
              Inspect step
            </Button>
          )}
        </div>
      )}
      {item.showError && <ErrorBlock span={item.span} />}
      {item.messages.map((message, index) => (
        <ConversationMessage key={index} message={message} />
      ))}
      {item.toolResult !== undefined && <ConversationTool item={item} onOpenStep={onOpenStep} />}
    </section>
  );
}

function ConversationTool({ item, onOpenStep }: { item: ConversationItem; onOpenStep: (id: string) => void }) {
  const [open, setOpen] = useState(item.span.status === "error");
  const failed = item.span.status === "error";
  const summary = toolSummary(item.toolCall?.args);
  return (
    <div className={cn("min-w-0 overflow-hidden rounded-md border", failed && "border-destructive/40")}>
      <div className="flex min-w-0 items-start gap-1 pr-2">
        <button
          type="button"
          aria-expanded={open}
          aria-label={`${open ? "Collapse" : "Expand"} ${item.span.name} tool call`}
          onClick={() => setOpen((value) => !value)}
          className="flex min-w-0 flex-1 items-center gap-2 px-3 py-2.5 text-left text-sm hover:bg-muted/40"
        >
          <ChevronRight
            className={cn("size-3.5 shrink-0 text-muted-foreground transition-transform", open && "rotate-90")}
          />
          <Wrench className="size-3.5 shrink-0 text-muted-foreground" />
          <span className="min-w-0 flex-1">
            <span className="block truncate font-medium">{item.span.name}</span>
            {summary && !open && (
              <span className="mt-1 block truncate font-mono text-xs text-muted-foreground" title={summary}>
                {summary}
              </span>
            )}
          </span>
          <span className={cn("ml-auto shrink-0 text-xs", failed ? "text-destructive" : "text-muted-foreground")}>
            {failed ? "Failed" : "Completed"}
          </span>
        </button>
        <Button
          variant="ghost"
          size="xs"
          aria-label={`Inspect step ${item.span.name}`}
          onClick={() => onOpenStep(item.span.span_id)}
          className="my-2.5 shrink-0 text-muted-foreground"
        >
          Inspect
        </Button>
      </div>
      {open && (
        <div className="min-w-0 space-y-3 border-t px-3 py-3">
          {item.toolCall && <ToolArguments args={item.toolCall.args} />}
          {item.span.error && item.span.error !== item.toolResult && (
            <p className="text-sm text-destructive">{item.span.error}</p>
          )}
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span>Result</span>
            <CopyButton
              variant="action"
              value={item.toolResult ?? ""}
              label={`Copy ${item.span.name} result`}
              iconOnly
            />
          </div>
          {item.toolResult ? (
            <ToolOutput result={item.toolResult} failed={failed} />
          ) : (
            <p className="text-sm text-muted-foreground">No result content recorded.</p>
          )}
        </div>
      )}
    </div>
  );
}

function ConversationMessage({ message }: { message: TraceMessage }) {
  if (message.role === "system" && (message.content.startsWith("Agent ") || message.content === "Context compacted"))
    return <p className="text-xs text-muted-foreground">{message.content}</p>;
  if (message.role === "system")
    return (
      <details className="text-sm text-muted-foreground">
        <summary className="cursor-pointer">Session context</summary>
        <div className="pt-3">
          <Markdown text={message.content} />
        </div>
      </details>
    );
  if (message.role === "tool") return <ToolResultCard name={message.name || "Tool result"} result={message.content} />;
  return (
    <div className={message.role === "user" ? "flex justify-end" : "space-y-3"}>
      {message.content && (
        <div className={message.role === "user" ? "max-w-[85%] rounded-xl bg-muted px-4 py-3" : "px-1 py-1"}>
          <Markdown text={message.content} />
        </div>
      )}
      {message.tool_calls?.map((call, index) => (
        <details key={index} className="rounded-md border p-3 text-sm">
          <summary className="cursor-pointer">{call.name}</summary>
          <div className="pt-3">
            <ToolCallBlock call={call} />
          </div>
        </details>
      ))}
    </div>
  );
}
