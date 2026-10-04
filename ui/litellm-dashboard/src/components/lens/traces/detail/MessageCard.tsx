"use client";

import { Wrench } from "lucide-react";
import { useState } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";

import { cn } from "@/lib/cva.config";

import { FoldChevron } from "../ui/Collapse";
import CopyButton from "@/components/shared/CopyButton";
import { displayValue, type KeyValue, KeyValueRows, objectEntries } from "./KeyValueRows";
import type { TraceMessage, TraceToolCall } from "../types";

const ROLE_LABEL: Record<string, string> = { user: "User", system: "System", assistant: "Assistant", tool: "Tool" };

const BLOCK_GAP = "mb-[1lh] last:mb-0";

const markdownComponents: Components = {
  p: ({ node, ...props }) => <p className={BLOCK_GAP} {...props} />,
  ul: ({ node, ...props }) => <ul className={cn(BLOCK_GAP, "list-disc pl-6")} {...props} />,
  ol: ({ node, ...props }) => <ol className={cn(BLOCK_GAP, "list-decimal pl-6")} {...props} />,
  li: ({ node, ...props }) => <li className="pl-0.5 [&>p]:mb-0 [&>ol]:mb-0 [&>ul]:mb-0" {...props} />,
  h1: ({ node, ...props }) => <h1 className={cn(BLOCK_GAP, "font-semibold")} {...props} />,
  h2: ({ node, ...props }) => <h2 className={cn(BLOCK_GAP, "font-semibold")} {...props} />,
  h3: ({ node, ...props }) => <h3 className={cn(BLOCK_GAP, "font-semibold")} {...props} />,
  blockquote: ({ node, ...props }) => (
    <blockquote className={cn(BLOCK_GAP, "border-l-2 border-border pl-3 text-muted-foreground")} {...props} />
  ),
  pre: ({ node, ...props }) => (
    <pre
      className={cn(
        BLOCK_GAP,
        "max-w-full overflow-x-auto rounded-sm bg-muted/40 p-2.5 font-mono text-sm leading-normal tracking-normal",
      )}
      {...props}
    />
  ),
  code: ({ node, className, ...props }) => <code className={cn("font-mono", className)} {...props} />,
  a: ({ node, ...props }) => <a className="text-primary underline" target="_blank" rel="noreferrer" {...props} />,
  img: ({ alt }) => <span>{alt || "Image omitted"}</span>,
};

export function Markdown({ text }: { text: string }) {
  return (
    <div className="text-sm leading-6 break-words text-foreground">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
        {text}
      </ReactMarkdown>
    </div>
  );
}

export function Card({ children, className }: { children: React.ReactNode; className?: string }) {
  return <section className={cn("rounded-md border border-border bg-background", className)}>{children}</section>;
}

const TILE = "grid size-4 shrink-0 place-items-center rounded-sm p-0.5";

function RoleTile({ failed = false }: { role: string; failed?: boolean }) {
  return <Wrench className={cn("size-3.5 shrink-0", failed ? "text-destructive" : "text-muted-foreground")} />;
}

function FoldTile({ label, open, onToggle }: { label: string; open: boolean; onToggle: () => void }) {
  return (
    <button
      type="button"
      aria-expanded={open}
      aria-label={`${open ? "Collapse" : "Expand"} ${label}`}
      onClick={onToggle}
      className={cn(
        TILE,
        "group/tile cursor-pointer transition-colors duration-150 group-hover/header:bg-muted focus-visible:bg-muted motion-reduce:transition-none",
        "text-muted-foreground",
      )}
    >
      <FoldChevron open={open} className="size-3 text-muted-foreground" />
    </button>
  );
}

const HEADER =
  "group/header flex min-w-0 items-center gap-2 border border-border bg-background py-2 pr-2 pl-3 transition-colors hover:bg-muted/30";
const LABEL = "min-w-0 truncate text-xs leading-5 font-medium text-foreground";
const CARD_COPY =
  "ml-auto size-5 shrink-0 rounded-sm text-muted-foreground opacity-60 transition-opacity duration-150 group-hover/header:opacity-100 focus-visible:opacity-100 motion-reduce:transition-none [&_svg]:size-4";
const INLINE_COPY = "size-4 shrink-0 rounded-sm text-muted-foreground";

export function ToolCallBlock({ call }: { call: TraceToolCall }) {
  const argEntries = objectEntries(call.args);
  const entries: KeyValue[] = argEntries?.length ? argEntries : [["arguments", displayValue(call.args)]];
  return (
    <div className="space-y-1">
      <div className="flex items-center gap-2 py-1">
        <RoleTile role="tool" />
        <span className="min-w-0 truncate text-sm font-medium">{call.name}</span>
        <CopyButton
          variant="action"
          value={call.name}
          label={`Copy ${call.name} name`}
          iconOnly
          className={INLINE_COPY}
        />
      </div>
      <KeyValueRows entries={entries} />
    </div>
  );
}

export function MessageCard({ message }: { message: TraceMessage; model: string | null }) {
  const [open, setOpen] = useState(true);
  const label = message.role === "tool" ? message.name ?? "Tool" : ROLE_LABEL[message.role] ?? message.role;
  if (message.role === "tool") return <ToolResultCard name={label} result={message.content} />;
  const calls = message.tool_calls ?? [];
  const copyValue = message.content || calls.map((c) => `${c.name}(${displayValue(c.args)})`).join("\n");
  const hasBody = Boolean(message.content) || calls.length > 0;
  const expanded = open && hasBody;
  return (
    <article className="rounded-md">
      <div className={cn(HEADER, "sticky top-10 z-sticky", expanded ? "rounded-t-sm border-b-0" : "rounded-sm")}>
        <FoldTile label={label} open={open} onToggle={() => setOpen((v) => !v)} />
        <span className={LABEL}>{label}</span>
        <CopyButton variant="action" value={copyValue} label={`Copy ${label}`} iconOnly className={CARD_COPY} />
      </div>
      {expanded && (
        <div
          className={cn(
            "flex flex-col gap-3 rounded-b-md border border-t-0 border-border bg-background px-3 pt-2 pb-3",
            message.role === "system" && "[&_*]:text-muted-foreground",
          )}
        >
          {message.content && <Markdown text={message.content} />}
          {calls.map((call, i) => (
            <ToolCallBlock key={`${call.name}-${i}`} call={call} />
          ))}
        </div>
      )}
    </article>
  );
}

const LONG_RESULT_CHARS = 120;

/** Tool result card: one line by default; long or multiline results expand in place. Red when the tool failed. */
export function ToolResultCard({ name, result, failed = false }: { name: string; result: string; failed?: boolean }) {
  const [open, setOpen] = useState(false);
  const expandable = result.length > LONG_RESULT_CHARS || result.includes("\n");
  const tone = failed ? "text-destructive" : "text-muted-foreground";
  return (
    <article className="flex flex-col">
      <div className={cn(HEADER, open ? "rounded-t-sm" : "rounded-sm")}>
        <RoleTile role="tool" failed={failed} />
        <span className="flex min-w-0 max-w-[50%] shrink-0 items-center gap-1.5">
          <span className={cn(LABEL, failed && "text-destructive")}>{name}</span>
          <CopyButton variant="action" value={name} label={`Copy ${name} name`} iconOnly className={INLINE_COPY} />
        </span>
        {expandable ? (
          <button
            type="button"
            aria-expanded={open}
            aria-label={`${open ? "Collapse" : "Expand"} ${name} result`}
            onClick={() => setOpen((prev) => !prev)}
            className={cn("flex min-w-0 cursor-pointer items-center gap-1 text-left", tone)}
          >
            <FoldChevron open={open} className="size-3 shrink-0" />
            {!open && <span className="min-w-0 truncate text-sm leading-tight">{result}</span>}
          </button>
        ) : (
          <span className={cn("min-w-0 break-words text-sm leading-5", tone)}>{result || "No output"}</span>
        )}
        <CopyButton variant="action" value={result} label={`Copy ${name} result`} iconOnly className={CARD_COPY} />
      </div>
      {open && (
        <pre
          className={cn(
            "max-h-80 overflow-auto rounded-b-sm border border-t-0 border-border bg-muted/40 px-3 py-2 font-mono text-xs leading-normal break-words whitespace-pre-wrap",
            tone,
          )}
        >
          {result}
        </pre>
      )}
    </article>
  );
}

/** Collapsible "Input" / "Output" section: sticky header, chevron turns, body snaps open and shut. */
export function Section({
  title,
  children,
  defaultOpen = true,
  count,
}: {
  title: string;
  children: React.ReactNode;
  defaultOpen?: boolean;
  count?: number;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <section aria-label={title}>
      <button
        type="button"
        aria-expanded={open}
        onClick={() => setOpen((prev) => !prev)}
        className="sticky top-0 z-sticky-pinned mb-2 flex h-10 w-full cursor-pointer items-center gap-2.5 bg-background px-3 text-left transition-colors duration-150 hover:bg-muted/40 motion-reduce:transition-none"
      >
        <FoldChevron open={open} className="size-3 shrink-0 text-foreground" />
        <span className="text-sm leading-tight font-medium text-foreground">{title}</span>{" "}
        {count !== undefined && (
          <span className="text-xs font-normal text-muted-foreground">
            {count} {count === 1 ? "message" : "messages"}
          </span>
        )}
      </button>
      <div hidden={!open} inert={!open} className="flex flex-col gap-3 px-3 pb-4">
        {children}
      </div>
    </section>
  );
}
