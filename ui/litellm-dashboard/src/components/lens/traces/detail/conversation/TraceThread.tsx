"use client";

import { ChevronRight, MessageSquareText, TriangleAlert, Wrench } from "lucide-react";
import { useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/cva.config";

import type { Trace } from "../../types";
import { fmtMs } from "../../utils";
import { ErrorBlock } from "../content/SpanError";
import { Markdown } from "../content/Markdown";
import {
  buildConversation,
  conversationWarnings,
  groupConversation,
  pendingConversationBranches,
} from "./conversation";
import { ConversationMessage, ConversationStep, ConversationSteps } from "./ConversationParts";
import {
  buildThread,
  replyErrorSpanIds,
  threadDurationMs,
  withoutErrorsOf,
  type ThreadTurn,
  type ThreadWork,
} from "./thread";
import { useConversationDetails } from "./useConversationDetails";

export interface ConversationTracePaging {
  loading: boolean;
  failed: boolean;
  loadMore: () => void;
}

const AUTO_LOAD_STEPS = 200;

interface TraceThreadProps {
  trace: Trace;
  accessToken: string;
  onOpenStep: (id: string) => void;
  paging?: ConversationTracePaging;
}

export function TraceThread({ trace, accessToken, onOpenStep, paging }: TraceThreadProps) {
  const { details, entries, complete, loading, failed, hasMore, loadMore } = useConversationDetails(trace, accessToken);
  const pending = pendingConversationBranches(trace.spans, details, Boolean(trace.next_cursor));
  const groups = groupConversation(buildConversation(trace.spans, details, complete, pending), trace.spans);
  const builtTurns = buildThread(groups);
  const traceComplete = complete && !trace.next_cursor;
  const shownErrors = replyErrorSpanIds(builtTurns);
  const toolSpanIds = new Set(trace.spans.flatMap((span) => (span.type === "tool" ? [span.span_id] : [])));
  const rootErrors = trace.spans.filter(
    (span) => span.parent_span_id === null && span.status === "error" && !shownErrors.has(span.span_id),
  );
  const rootErrorIds = new Set(rootErrors.map((span) => span.span_id));
  const turns = builtTurns.map((turn) => withoutErrorsOf(turn, rootErrorIds));
  const busy = loading || Boolean(paging?.loading);
  const blocked = failed || Boolean(paging?.failed);
  const sourceRemaining = hasMore || Boolean(trace.next_cursor && paging);
  const canLoad = !busy && !blocked && sourceRemaining;
  const autoLoad = canLoad && details.size < AUTO_LOAD_STEPS;
  const loadTracePage = paging?.loadMore;
  useEffect(() => {
    if (!autoLoad) return;
    if (!hasMore) {
      loadTracePage?.();
      return;
    }
    const controller = new AbortController();
    void loadMore(controller.signal);
    return () => controller.abort();
  }, [autoLoad, hasMore, loadMore, loadTracePage]);
  const loadNext = () => {
    if (hasMore) void loadMore(new AbortController().signal);
    else loadTracePage?.();
  };
  return (
    <section aria-label="Trace thread" className="min-h-0 flex-1 overflow-y-auto">
      <div className="mx-auto min-w-0 max-w-3xl space-y-8 px-3 py-4 sm:px-6 sm:py-6">
        {rootErrors.map((span) => (
          <ErrorBlock key={span.span_id} span={span} />
        ))}
        {conversationWarnings(details, traceComplete, toolSpanIds).map((warning) => (
          <p key={warning} role="status" className="rounded-md border p-3 text-sm text-muted-foreground">
            {warning}
          </p>
        ))}
        {turns.map((turn) => (
          <ThreadTurnView key={turn.id} turn={turn} onOpenStep={onOpenStep} />
        ))}
        <ThreadFooter
          turnCount={turns.length}
          busy={busy}
          stepsFailed={failed}
          traceFailed={Boolean(paging?.failed)}
          complete={traceComplete}
          onRetry={() => entries.forEach(({ query }) => query.isError && void query.refetch())}
          onLoadMore={canLoad && !autoLoad ? loadNext : undefined}
        />
      </div>
    </section>
  );
}

interface ThreadFooterProps {
  turnCount: number;
  busy: boolean;
  stepsFailed: boolean;
  traceFailed: boolean;
  complete: boolean;
  onRetry: () => void;
  onLoadMore?: () => void;
}

function ThreadFooter({ turnCount, busy, stepsFailed, traceFailed, complete, onRetry, onLoadMore }: ThreadFooterProps) {
  return (
    <>
      {busy && (
        <p role="status" className="text-sm text-muted-foreground">
          Loading thread…
        </p>
      )}
      {stepsFailed && (
        <div role="alert" className="flex items-center justify-between gap-3 text-sm text-destructive">
          Could not load some steps of this thread.
          <Button variant="outline" size="sm" onClick={onRetry}>
            Retry
          </Button>
        </div>
      )}
      {traceFailed && (
        <p role="alert" className="text-sm text-destructive">
          Could not load more of this trace. Use Retry or Refresh trace above.
        </p>
      )}
      {complete && !turnCount && <p className="text-sm text-muted-foreground">No conversation content recorded.</p>}
      <div className="flex items-center justify-between gap-3 border-t pt-4 text-xs text-muted-foreground">
        <span>{complete ? "End of thread" : `${turnCount} ${turnCount === 1 ? "turn" : "turns"} loaded`}</span>
        {onLoadMore && (
          <Button variant="outline" size="sm" onClick={onLoadMore}>
            Load more
          </Button>
        )}
      </div>
    </>
  );
}

function ThreadTurnView({ turn, onOpenStep }: { turn: ThreadTurn; onOpenStep: (id: string) => void }) {
  return (
    <article aria-label="Thread turn" className="min-w-0 space-y-3">
      {turn.context.length > 0 && <PromptContext count={turn.context.length} turn={turn} />}
      {turn.prompt.map((message, index) => (
        <ConversationMessage key={index} message={message} />
      ))}
      {turn.work.length > 0 && <WorkedBar turn={turn} onOpenStep={onOpenStep} />}
      {turn.replyItem?.showError && <ErrorBlock span={turn.replyItem.span} />}
      {turn.reply && (
        <div className="space-y-1.5">
          <div className="px-1 py-1">
            <Markdown text={turn.reply.content} />
          </div>
          <ReplyMeta turn={turn} onOpenStep={onOpenStep} />
        </div>
      )}
    </article>
  );
}

function PromptContext({ count, turn }: { count: number; turn: ThreadTurn }) {
  return (
    <details className="group flex justify-end text-xs text-muted-foreground">
      <summary className="ml-auto flex cursor-pointer list-none items-center gap-1.5 hover:text-foreground">
        Prompt context
        <MessageSquareText className="size-3.5" />
        <span className="tabular-nums">{count}</span>
        <ChevronRight className="size-3.5 transition-transform group-open:rotate-90" />
      </summary>
      <div className="mt-2 space-y-3 rounded-lg border p-3 text-sm">
        {turn.context.map((message, index) => (
          <Markdown key={index} text={message.content} />
        ))}
      </div>
    </details>
  );
}

function WorkedBar({ turn, onOpenStep }: { turn: ThreadTurn; onOpenStep: (id: string) => void }) {
  const [open, setOpen] = useState(false);
  const duration = threadDurationMs(turn);
  return (
    <div className="min-w-0">
      <button
        type="button"
        aria-expanded={open}
        onClick={() => setOpen((value) => !value)}
        className="flex items-center gap-2.5 rounded-md py-1 text-sm text-muted-foreground hover:text-foreground"
      >
        <span>{duration === null ? "Worked" : `Worked ${fmtMs(duration)}`}</span>
        <Count icon={MessageSquareText} value={turn.llmCalls} label="model calls" />
        <Count icon={Wrench} value={turn.toolCalls} label="tool calls" />
        {turn.failed && <TriangleAlert aria-label="A step failed" className="size-3.5 text-destructive" />}
        <ChevronRight className={cn("size-3.5 transition-transform", open && "rotate-90")} />
      </button>
      {open && (
        <div className="mt-3 min-w-0 space-y-4 border-l pl-4">
          {turn.work.map((work) => (
            <WorkItem key={work.kind === "step" ? work.item.id : work.id} work={work} onOpenStep={onOpenStep} />
          ))}
        </div>
      )}
    </div>
  );
}

function Count({ icon: Icon, value, label }: { icon: typeof Wrench; value: number; label: string }) {
  return (
    <span className="inline-flex items-center gap-1 tabular-nums" aria-label={`${value} ${label}`}>
      <Icon className="size-3.5" />
      {value}
    </span>
  );
}

function WorkItem({ work, onOpenStep }: { work: ThreadWork; onOpenStep: (id: string) => void }) {
  if (work.kind === "step") return <ConversationStep item={work.item} onOpenStep={onOpenStep} />;
  return (
    <details className="min-w-0 rounded-md border p-3">
      <summary className="cursor-pointer text-sm font-medium">Subagent: {work.name}</summary>
      <div className="mt-4 min-w-0 space-y-5 border-l pl-3">
        <ConversationSteps groups={work.groups} onOpenStep={onOpenStep} />
      </div>
    </details>
  );
}

function ReplyMeta({ turn, onOpenStep }: { turn: ThreadTurn; onOpenStep: (id: string) => void }) {
  const item = turn.replyItem;
  if (!item) return null;
  const tokens = item.span.input_tokens + item.span.output_tokens;
  return (
    <div className="flex items-center gap-3 px-1 text-xs text-muted-foreground tabular-nums">
      {item.model && <span className="truncate">{item.model}</span>}
      <span>{fmtMs(item.span.duration_ms)}</span>
      {tokens > 0 && <span>{tokens.toLocaleString()} tok</span>}
      <Button
        variant="ghost"
        size="xs"
        className="ml-auto text-muted-foreground"
        aria-label={`Inspect step ${item.span.name}`}
        onClick={() => onOpenStep(item.span.span_id)}
      >
        Inspect
      </Button>
    </div>
  );
}
