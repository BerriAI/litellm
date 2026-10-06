"use client";

import { useEffect, useState } from "react";
import { ChevronRight, Wrench } from "lucide-react";
import { cn } from "@/lib/cva.config";
import CopyButton from "@/components/shared/CopyButton";
import { ToolArguments, ToolOutput } from "../content/ToolContent";
import { toolSummary } from "../content/payload";
import { Button } from "@/components/ui/button";
import {
  buildConversation,
  conversationWarnings,
  conversationBranchGroups,
  pendingConversationBranches,
  groupConversation,
  CONVERSATION_PAGE_SIZE,
  type ConversationItem,
  type ConversationGroup,
} from "./conversation";
import { ErrorBlock } from "../content/SpanError";
import { Markdown } from "../content/Markdown";
import { ToolCallBlock, ToolResultCard } from "../content/Messages";
import type { Trace, TraceMessage } from "../../types";
import { fmtMs } from "../../utils";
import { useConversationDetails } from "./useConversationDetails";

export interface ConversationTracePaging {
  loading: boolean;
  failed: boolean;
  loadMore: () => void;
}

function hasRemainingSource(branchId: string | null, sourceRemaining: boolean, pendingBranches: ReadonlySet<string>) {
  return sourceRemaining && (branchId === null || pendingBranches.has(branchId));
}

export function TraceConversation({
  trace,
  accessToken,
  onOpenStep,
  paging,
}: {
  trace: Trace;
  accessToken: string;
  onOpenStep: (id: string) => void;
  paging?: ConversationTracePaging;
}) {
  const [pages, setPages] = useState<ReadonlyMap<string | null, number>>(new Map());
  const [requestedBranch, setRequestedBranch] = useState<string | null>();
  const { details, entries, complete, loading, failed, hasMore, loadMore } = useConversationDetails(trace, accessToken);
  const traceComplete = complete && !trace.next_cursor;
  const pendingBranches = pendingConversationBranches(trace.spans, details, Boolean(trace.next_cursor));
  const items = buildConversation(trace.spans, details, complete, pendingBranches);
  const groups = groupConversation(items, trace.spans);
  const requestedId = requestedBranch ?? null;
  const requestedGroups = conversationBranchGroups(groups, requestedId);
  const requestedLimit = pages.get(requestedId) ?? CONVERSATION_PAGE_SIZE;
  const sourceRemaining = hasMore || Boolean(trace.next_cursor && paging);
  const requestedSourceRemaining = hasRemainingSource(requestedId, sourceRemaining, pendingBranches);
  const needsMore =
    requestedBranch !== undefined && requestedGroups.length < requestedLimit && requestedSourceRemaining;
  const busy = loading || Boolean(paging?.loading);
  const blocked = failed || Boolean(paging?.failed);
  const loadTracePage = paging?.loadMore;
  useEffect(() => {
    if (!needsMore || busy || blocked) return;
    if (hasMore) {
      const controller = new AbortController();
      void loadMore(controller.signal);
      return () => controller.abort();
    }
    if (trace.next_cursor) loadTracePage?.();
  }, [needsMore, busy, blocked, hasMore, loadMore, trace.next_cursor, loadTracePage]);
  const fillingPage = needsMore && sourceRemaining && !blocked;
  const loadEntries = (branchId: string | null, shown: number) => {
    setPages((current) => new Map([...current, [branchId, shown + CONVERSATION_PAGE_SIZE]]));
    setRequestedBranch(branchId);
  };
  const warnings = conversationWarnings(details, traceComplete);
  const multipleAgents = new Set(items.map((item) => item.agentId).filter(Boolean)).size > 1;
  const rootLimit = pages.get(null) ?? CONVERSATION_PAGE_SIZE;
  const inlineErrorIds = new Set(
    groups
      .slice(0, rootLimit)
      .flatMap((group) => (group.kind === "item" && group.item.showError ? [group.item.span.span_id] : [])),
  );
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
          groups={groups}
          multipleAgents={multipleAgents}
          onOpenStep={onOpenStep}
          pages={pages}
          sourceRemaining={sourceRemaining}
          pendingBranches={pendingBranches}
          disabled={busy || blocked || fillingPage}
          onLoadEntries={loadEntries}
          complete={traceComplete}
        />
        <ConversationFeedback
          entries={entries}
          loading={busy || fillingPage}
          complete={traceComplete}
          empty={items.length === 0}
          paging={paging}
        />
      </div>
    </section>
  );
}

function ConversationFeedback({
  entries,
  loading,
  complete,
  empty,
  paging,
}: {
  entries: ReturnType<typeof useConversationDetails>["entries"];
  loading: boolean;
  complete: boolean;
  empty: boolean;
  paging?: ConversationTracePaging;
}) {
  return (
    <>
      {entries.map(
        ({ query, span }) =>
          query.isError && (
            <div key={span.span_id} role="alert" className="rounded-md border p-3 text-sm">
              Could not load {span.name}. Retry this step to continue the conversation.
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
      {complete && empty && <p className="text-sm text-muted-foreground">No conversation content recorded.</p>}
      {paging?.failed && (
        <p role="alert" className="text-sm text-destructive">
          Could not load more conversation entries. Use Retry or Refresh trace above to continue.
        </p>
      )}
    </>
  );
}

function ConversationGroups({
  groups,
  multipleAgents,
  onOpenStep,
  branchId = null,
  name,
  pages,
  sourceRemaining,
  pendingBranches,
  disabled,
  onLoadEntries,
  complete,
}: {
  groups: ConversationGroup[];
  multipleAgents: boolean;
  onOpenStep: (id: string) => void;
  branchId?: string | null;
  name?: string;
  pages: ReadonlyMap<string | null, number>;
  sourceRemaining: boolean;
  pendingBranches: ReadonlySet<string>;
  disabled: boolean;
  onLoadEntries: (branchId: string | null, shown: number) => void;
  complete: boolean;
}) {
  const limit = pages.get(branchId) ?? CONVERSATION_PAGE_SIZE;
  const visible = groups.slice(0, limit);
  const branchRemaining = hasRemainingSource(branchId, sourceRemaining, pendingBranches);
  const hasMore = visible.length < groups.length || branchRemaining;
  const nextCount = branchRemaining
    ? CONVERSATION_PAGE_SIZE
    : Math.min(CONVERSATION_PAGE_SIZE, groups.length - visible.length);
  return (
    <>
      {visible.map((group) =>
        group.kind === "item" ? (
          <ConversationStep
            key={group.item.id}
            item={group.item}
            multipleAgents={multipleAgents}
            onOpenStep={onOpenStep}
          />
        ) : (
          <details key={group.id} className="min-w-0 rounded-md border p-3">
            <summary className="cursor-pointer text-sm font-medium">
              Subagent: {group.name}
              <span className="ml-2 text-xs font-normal text-muted-foreground">
                {group.children.length} entries loaded
              </span>
            </summary>
            <div className="mt-4 min-w-0 space-y-5 border-l pl-3">
              <ConversationGroups
                groups={group.children}
                multipleAgents={multipleAgents}
                onOpenStep={onOpenStep}
                branchId={group.id}
                name={group.name}
                pages={pages}
                sourceRemaining={sourceRemaining}
                pendingBranches={pendingBranches}
                disabled={disabled}
                onLoadEntries={onLoadEntries}
                complete={complete}
              />
            </div>
          </details>
        ),
      )}
      <div className="flex items-center justify-between gap-3 border-t pt-4 text-xs text-muted-foreground">
        <span>
          {complete && !hasMore && branchId === null ? "End of conversation" : `${visible.length} entries shown`}
        </span>
        {hasMore && (
          <Button
            variant="outline"
            size="sm"
            disabled={disabled}
            aria-label={name ? `Load next ${nextCount} entries in ${name}` : undefined}
            onClick={() => onLoadEntries(branchId, visible.length)}
          >
            Load next {nextCount} entries
          </Button>
        )}
      </div>
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
