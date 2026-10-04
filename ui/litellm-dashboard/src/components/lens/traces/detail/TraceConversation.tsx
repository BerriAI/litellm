"use client";

import { useQueries } from "@tanstack/react-query";
import { useState } from "react";
import { ChevronRight, Wrench } from "lucide-react";
import { cn } from "@/lib/cva.config";
import { CopyButton } from "../ui/CopyButton";
import { useTracesApi } from "../api";
import { Button } from "@/components/ui/button";
import { buildConversation, conversationSteps, CONVERSATION_PAGE_SIZE, type ConversationItem } from "./conversation";
import { ErrorBlock } from "./DetailContent";
import { Markdown, ToolCallBlock } from "./MessageCard";
import type { SpanDetail, Trace, TraceMessage } from "../types";
import { fmtMs } from "../utils";

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
  const items = buildConversation(trace.spans, details, complete);
  const inlineErrorIds = new Set(items.filter((item) => item.showError).map((item) => item.span.span_id));
  const rootErrors = trace.spans.filter((span) => {
    const failedRoot = span.parent_span_id === null && span.status === "error" && span.type !== "tool";
    return failedRoot && !inlineErrorIds.has(span.span_id);
  });
  return (
    <section aria-label="Trace conversation" className="min-h-0 flex-1 overflow-y-auto">
      <div className="mx-auto max-w-3xl space-y-6 px-6 py-5">
        {rootErrors.map((span) => (
          <ErrorBlock key={span.span_id} span={span} />
        ))}
        {items.map((item) => (
          <section
            key={item.id}
            className="group/conversation relative space-y-3"
            aria-label={`Conversation step ${item.span.name}`}
          >
            {item.showError && <ErrorBlock span={item.span} />}
            {item.messages.map((message, index) => (
              <ConversationMessage key={index} message={message} />
            ))}
            {item.toolResult !== undefined && <ConversationTool item={item} />}
            <Button
              variant="ghost"
              size="xs"
              title={`${item.span.agent || item.span.name} · ${fmtMs(item.span.start_offset_ms)}`}
              aria-label={`Inspect step ${item.span.name}`}
              className="absolute right-0 -bottom-5 z-raised bg-background text-muted-foreground opacity-0 group-hover/conversation:opacity-100 focus-visible:opacity-100"
              onClick={() => onOpenStep(item.span.span_id)}
            >
              Inspect step
            </Button>
          </section>
        ))}
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
        {complete && items.length === 0 && (
          <p className="text-sm text-muted-foreground">No conversation content recorded.</p>
        )}
        <div className="flex items-center justify-between gap-3 border-t pt-4 text-xs text-muted-foreground">
          <span>{complete ? "End of conversation" : `${loadedCount} of ${steps.length} steps loaded`}</span>
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

function ConversationTool({ item }: { item: ConversationItem }) {
  const [open, setOpen] = useState(false);
  const failed = item.span.status === "error";
  return (
    <div className="rounded-md border">
      <button
        type="button"
        aria-expanded={open}
        aria-label={`${open ? "Collapse" : "Expand"} ${item.span.name} tool call`}
        onClick={() => setOpen((value) => !value)}
        className="flex w-full items-center gap-2 px-3 py-2.5 text-left text-sm hover:bg-muted/40"
      >
        <ChevronRight
          className={cn("size-3.5 shrink-0 text-muted-foreground transition-transform", open && "rotate-90")}
        />
        <Wrench className="size-3.5 shrink-0 text-muted-foreground" />
        <span className="min-w-0 truncate font-medium">{item.span.name}</span>
        <span className={cn("ml-auto shrink-0 text-xs", failed ? "text-destructive" : "text-muted-foreground")}>
          {failed ? "Failed" : "Completed"}
        </span>
      </button>
      {open && (
        <div className="space-y-3 border-t px-3 py-3">
          {item.toolCall && <ToolCallBlock call={item.toolCall} />}
          {item.span.error && item.span.error !== item.toolResult && (
            <p className="text-sm text-destructive">{item.span.error}</p>
          )}
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span>Result</span>
            <CopyButton value={item.toolResult ?? ""} label={`Copy ${item.span.name} result`} iconOnly />
          </div>
          <pre className="max-h-80 overflow-auto whitespace-pre-wrap break-words text-xs leading-5">
            {item.toolResult || "No output recorded"}
          </pre>
        </div>
      )}
    </div>
  );
}

function ConversationMessage({ message }: { message: TraceMessage }) {
  if (message.role === "system")
    return (
      <details className="text-sm text-muted-foreground">
        <summary className="cursor-pointer">System instructions</summary>
        <div className="pt-3">
          <Markdown text={message.content} />
        </div>
      </details>
    );
  if (message.role === "tool")
    return (
      <details className="rounded-md border p-3 text-sm">
        <summary className="cursor-pointer">{message.name || "Tool result"}</summary>
        <pre className="max-h-80 overflow-auto whitespace-pre-wrap pt-3 text-xs">{message.content}</pre>
      </details>
    );
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
