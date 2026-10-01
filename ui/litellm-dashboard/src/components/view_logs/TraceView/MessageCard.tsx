"use client";

import { useState } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";

import { cn } from "@/lib/cva.config";

import { Collapse, FoldChevron } from "./Collapse";
import { CopyButton } from "./CopyButton";
import { displayValue, KeyValueRows, objectEntries } from "./KeyValueRows";
import { RoleTile, SpanIcon } from "./SpanIcon";
import type { TraceMessage, TraceToolCall } from "./traceTypes";

const ROLE_LABEL: Record<string, string> = { user: "User", system: "System", assistant: "AI", tool: "Tool" };

const markdownComponents: Components = {
  p: ({ node, ...props }) => <p className="my-2 first:mt-0 last:mb-0" {...props} />,
  ul: ({ node, ...props }) => <ul className="my-2 list-disc space-y-1 pl-5" {...props} />,
  ol: ({ node, ...props }) => <ol className="my-2 list-decimal space-y-1 pl-5" {...props} />,
  pre: ({ node, ...props }) => (
    <pre className="my-2 max-w-full overflow-x-auto rounded-md bg-muted p-2.5 font-mono text-[12px]" {...props} />
  ),
  code: ({ node, className, ...props }) => (
    <code className={cn("rounded bg-muted px-1 py-px font-mono text-[12.5px]", className)} {...props} />
  ),
  a: ({ node, ...props }) => <a className="text-info underline" target="_blank" rel="noreferrer" {...props} />,
  img: ({ alt }) => <span>{alt || "Image omitted"}</span>,
};

export function Markdown({ text }: { text: string }) {
  return (
    <div className="max-w-[92ch] text-[16px] leading-6 break-words text-trace-text">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
        {text}
      </ReactMarkdown>
    </div>
  );
}

export function Card({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <section
      className={cn(
        "animate-slot-slide-in overflow-hidden rounded-sm border border-trace-border bg-card motion-reduce:animate-none",
        className,
      )}
    >
      {children}
    </section>
  );
}

function CardHeader({ tile, label, copyValue }: { tile: React.ReactNode; label: string; copyValue: string }) {
  return (
    <div className="flex w-full items-center gap-2.5 py-1.5 pr-2 pl-3 transition-colors duration-150 hover:bg-muted/60">
      {tile}
      <span className="text-[13px] font-medium text-trace-text">{label}</span>
      <CopyButton value={copyValue} label={`Copy ${label}`} iconOnly className="ml-auto" />
    </div>
  );
}

function ToolCallBlock({ call }: { call: TraceToolCall }) {
  const argEntries = objectEntries(call.args);
  return (
    <div className="px-3 pb-3">
      <KeyValueRows
        entries={[
          ["name", call.name],
          ["arguments", displayValue(call.args)],
        ]}
      />
      <div className="mt-1 ml-[22px]">
        <div className="flex items-center gap-2">
          <SpanIcon type="tool" size="sm" />
          <span className="font-medium">{call.name}</span>
        </div>
        {argEntries && argEntries.length > 0 && <KeyValueRows entries={argEntries} className="mt-1 ml-3" />}
      </div>
    </div>
  );
}

/** One chat message as a card: role tile, markdown body, and any tool calls as key / value rows. */
export function MessageCard({ message, model }: { message: TraceMessage; model: string | null }) {
  const label = message.role === "tool" ? message.name ?? "Tool" : ROLE_LABEL[message.role] ?? message.role;
  if (message.role === "tool") return <ToolResultCard name={label} result={message.content} />;
  const calls = message.tool_calls ?? [];
  const copyValue = message.content || calls.map((c) => `${c.name}(${displayValue(c.args)})`).join("\n");
  return (
    <Card>
      <CardHeader tile={<RoleTile role={message.role} model={model} />} label={label} copyValue={copyValue} />
      {message.content && (
        <div className={cn("px-3 pt-1 pb-3", message.role === "system" && "[&_*]:text-trace-key")}>
          <Markdown text={message.content} />
        </div>
      )}
      {calls.map((call, i) => (
        <ToolCallBlock key={`${call.name}-${i}`} call={call} />
      ))}
    </Card>
  );
}

/** One-line card for a tool's result: name, then the result text; red when the tool failed. */
export function ToolResultCard({ name, result, failed = false }: { name: string; result: string; failed?: boolean }) {
  return (
    <Card>
      <div className="flex min-w-0 items-center gap-2.5 py-1.5 pr-2 pl-3 transition-colors duration-150 hover:bg-muted/60">
        <SpanIcon type="tool" error={failed} size="sm" />
        <span className={cn("shrink-0 text-[13px] font-medium text-trace-text", failed && "text-destructive")}>
          {name}
        </span>
        <span
          className={cn("min-w-0 truncate text-[13px]", failed ? "text-destructive" : "text-trace-key")}
          title={result}
        >
          {result || "No output"}
        </span>
        <CopyButton value={result} label={`Copy ${name} result`} iconOnly className="ml-auto" />
      </div>
    </Card>
  );
}

/** Collapsible "Input" / "Output" section with a chevron header. */
export function Section({ title, children }: { title: string; children: React.ReactNode }) {
  const [open, setOpen] = useState(true);
  return (
    <section aria-label={title}>
      <button
        type="button"
        aria-expanded={open}
        onClick={() => setOpen((prev) => !prev)}
        className="flex items-center gap-2 py-2 text-[13px] font-medium text-trace-text transition duration-200"
      >
        <FoldChevron open={open} className="size-3 text-muted-foreground" />
        {title}
      </button>
      <Collapse open={open}>
        <div className="flex flex-col gap-2.5 pt-1 pb-2.5 pl-5">{children}</div>
      </Collapse>
    </section>
  );
}
