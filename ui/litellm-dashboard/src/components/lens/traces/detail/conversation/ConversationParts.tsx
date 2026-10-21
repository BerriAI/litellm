"use client";

import { ChevronRight, Wrench } from "lucide-react";
import { useState } from "react";

import CopyButton from "@/components/shared/CopyButton";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/cva.config";

import type { TraceMessage } from "../../types";
import { fmtMs } from "../../utils";
import { Markdown } from "../content/Markdown";
import { ToolCallBlock, ToolResultCard } from "../content/Messages";
import { toolSummary } from "../content/payload";
import { ErrorBlock } from "../content/SpanError";
import { ToolArguments, ToolOutput } from "../content/ToolContent";
import type { ConversationGroup, ConversationItem } from "./conversation";

export function ConversationSteps({
  groups,
  onOpenStep,
}: {
  groups: readonly ConversationGroup[];
  onOpenStep: (id: string) => void;
}) {
  return groups.map((group) =>
    group.kind === "item" ? (
      <ConversationStep key={group.item.id} item={group.item} onOpenStep={onOpenStep} />
    ) : (
      <details key={group.id} className="min-w-0 rounded-md border p-3">
        <summary className="cursor-pointer text-sm font-medium">Subagent: {group.name}</summary>
        <div className="mt-4 min-w-0 space-y-5 border-l pl-3">
          <ConversationSteps groups={group.children} onOpenStep={onOpenStep} />
        </div>
      </details>
    ),
  );
}

export function ConversationStep({ item, onOpenStep }: { item: ConversationItem; onOpenStep: (id: string) => void }) {
  return (
    <section className="min-w-0 space-y-2" aria-label={`Conversation step ${item.span.name}`}>
      {item.toolResult === undefined && (
        <div className="flex min-w-0 items-center justify-between gap-2 text-xs text-muted-foreground">
          <div className="flex min-w-0 items-center gap-2">
            {item.model && (
              <span className="truncate" title={item.model}>
                {item.model}
              </span>
            )}
            <span className="shrink-0 tabular-nums">{fmtMs(item.time ?? item.span.start_offset_ms)}</span>
          </div>
          <Button
            variant="ghost"
            size="xs"
            aria-label={`Inspect step ${item.span.name}`}
            onClick={() => onOpenStep(item.span.span_id)}
            className="shrink-0 text-muted-foreground"
          >
            Inspect step
          </Button>
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

export function ConversationMessage({ message }: { message: TraceMessage }) {
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
