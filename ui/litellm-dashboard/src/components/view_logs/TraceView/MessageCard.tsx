"use client";

import { Bot, ScrollText, UserRound, Wrench } from "lucide-react";
import { useState } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";

import { Logo } from "@/components/molecules/logo/Logo";
import { cn } from "@/lib/cva.config";

import { FoldChevron } from "./Collapse";
import { CopyButton } from "./CopyButton";
import { displayValue, type KeyValue, KeyValueRows, objectEntries } from "./KeyValueRows";
import { useSpanProvider } from "./spanProvider";
import type { TraceMessage, TraceToolCall } from "./traceTypes";

const ROLE_LABEL: Record<string, string> = { user: "User", system: "System", assistant: "AI", tool: "Tool" };

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
    <blockquote className={cn(BLOCK_GAP, "border-l-2 border-trace-border pl-3 text-trace-duration")} {...props} />
  ),
  pre: ({ node, ...props }) => (
    <pre
      className={cn(
        BLOCK_GAP,
        "max-w-full overflow-x-auto rounded-[4px] bg-trace-chip p-2.5 font-mono text-[13px] leading-[1.5] tracking-normal",
      )}
      {...props}
    />
  ),
  code: ({ node, className, ...props }) => <code className={cn("font-mono", className)} {...props} />,
  a: ({ node, ...props }) => <a className="text-trace-brand underline" target="_blank" rel="noreferrer" {...props} />,
  img: ({ alt }) => <span>{alt || "Image omitted"}</span>,
};

export function Markdown({ text }: { text: string }) {
  return (
    <div className="text-[14px] leading-[1.65] tracking-[-0.42px] break-words text-trace-text">
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
        "rounded-[4px] border-[0.67px] border-trace-card-border bg-trace-surface shadow-[0_1px_1px_0_rgba(16,24,40,0.02)]",
        className,
      )}
    >
      {children}
    </section>
  );
}

const TILE = "grid size-4 shrink-0 place-items-center rounded-[2px] p-0.5";

function RoleGlyph({ role, model, failed }: { role: string; model: string | null; failed: boolean }) {
  const provider = useSpanProvider(role === "assistant" ? model : null);
  if (role === "user") return <UserRound className="size-3 fill-current text-trace-human-glyph" strokeWidth={2.5} />;
  if (role === "tool") return <Wrench className={cn("size-3", failed ? "text-white" : "text-trace-glyph")} />;
  if (role === "system") return <ScrollText className="size-3 text-trace-glyph" />;
  if (provider) return <Logo provider={provider} className="size-3" />;
  return <Bot className="size-3 text-trace-glyph" />;
}

const tileTone = (role: string, failed: boolean, hasLogo = false): string => {
  if (failed) return "bg-destructive";
  if (role === "user") return "bg-trace-human";
  if (role === "assistant") return hasLogo ? "bg-white ring-1 ring-trace-line ring-inset" : "bg-trace-llm";
  if (role === "tool") return "bg-trace-tool";
  return "bg-trace-duration";
};

function RoleTile({ role, model = null, failed = false }: { role: string; model?: string | null; failed?: boolean }) {
  const provider = useSpanProvider(role === "assistant" ? model : null);
  return (
    <span className={cn(TILE, tileTone(role, failed, provider !== null))}>
      <RoleGlyph role={role} model={model} failed={failed} />
    </span>
  );
}

function FoldTile({
  role,
  model,
  label,
  open,
  onToggle,
}: {
  role: string;
  model: string | null;
  label: string;
  open: boolean;
  onToggle: () => void;
}) {
  const provider = useSpanProvider(role === "assistant" ? model : null);
  return (
    <button
      type="button"
      aria-expanded={open}
      aria-label={`${open ? "Collapse" : "Expand"} ${label}`}
      onClick={onToggle}
      className={cn(
        TILE,
        "group/tile cursor-pointer transition-colors duration-150 group-hover/header:bg-trace-tab-active focus-visible:bg-trace-tab-active motion-reduce:transition-none",
        tileTone(role, false, provider !== null),
      )}
    >
      <span className="contents group-hover/header:hidden group-focus-visible/tile:hidden">
        <RoleGlyph role={role} model={model} failed={false} />
      </span>
      <FoldChevron
        open={open}
        className="hidden size-3 text-trace-duration group-hover/header:block group-focus-visible/tile:block"
      />
    </button>
  );
}

const HEADER =
  "group/header flex min-w-0 items-center gap-2.5 border-[0.67px] border-trace-card-border bg-trace-surface py-1.5 pr-2 pl-3 transition-colors duration-150 hover:bg-trace-chip motion-reduce:transition-none dark:bg-trace-chip";
const LABEL = "shrink-0 text-[13px] leading-[1.2] font-semibold tracking-[-0.26px] text-trace-text";
const CARD_COPY =
  "ml-auto size-5 shrink-0 rounded-[3px] text-trace-text-2 opacity-60 transition-opacity duration-150 group-hover/header:opacity-100 focus-visible:opacity-100 motion-reduce:transition-none [&_svg]:size-4";
const INLINE_COPY = "size-4 shrink-0 rounded-[3px] text-trace-duration";

function ToolCallBlock({ call }: { call: TraceToolCall }) {
  const raw: KeyValue[] = [
    ["arguments", displayValue(call.args)],
    ["name", call.name],
  ];
  const argEntries = objectEntries(call.args);
  return (
    <>
      <KeyValueRows entries={raw} />
      <div className="-mx-3 flex h-7 items-center gap-2.5 py-1.5 pr-3 pl-[30px] transition-colors duration-150 hover:bg-trace-row-hover motion-reduce:transition-none">
        <RoleTile role="tool" />
        <span className="flex min-w-0 items-center gap-1.5">
          <span className="truncate text-[13px] leading-[1.2] font-medium tracking-[-0.26px] text-trace-text">
            {call.name}
          </span>
          <CopyButton value={call.name} label={`Copy ${call.name} name`} iconOnly className={INLINE_COPY} />
        </span>
      </div>
      {argEntries && argEntries.length > 0 && <KeyValueRows entries={argEntries} className="pl-10" />}
    </>
  );
}

/** One chat message as a card with a sticky header; hovering the role tile turns it into a fold chevron. */
export function MessageCard({ message, model }: { message: TraceMessage; model: string | null }) {
  const [open, setOpen] = useState(true);
  const label = message.role === "tool" ? message.name ?? "Tool" : ROLE_LABEL[message.role] ?? message.role;
  if (message.role === "tool") return <ToolResultCard name={label} result={message.content} />;
  const calls = message.tool_calls ?? [];
  const copyValue = message.content || calls.map((c) => `${c.name}(${displayValue(c.args)})`).join("\n");
  const hasBody = Boolean(message.content) || calls.length > 0;
  const expanded = open && hasBody;
  return (
    <article className="rounded-[4px] shadow-[0_1px_1px_0_rgba(16,24,40,0.02)]">
      <div className={cn(HEADER, "sticky top-10 z-sticky", expanded ? "rounded-t-[4px] border-b-0" : "rounded-[4px]")}>
        <FoldTile role={message.role} model={model} label={label} open={open} onToggle={() => setOpen((v) => !v)} />
        <span className={LABEL}>{label}</span>
        <CopyButton value={copyValue} label={`Copy ${label}`} iconOnly className={CARD_COPY} />
      </div>
      {expanded && (
        <div
          className={cn(
            "flex flex-col gap-2 rounded-b-[4px] border-[0.67px] border-t-0 border-trace-card-border bg-trace-surface px-3 pt-1.5 pb-3",
            message.role === "system" && "[&_*]:text-trace-duration",
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
  const tone = failed ? "text-destructive" : "text-trace-duration";
  return (
    <article className="flex flex-col">
      <div
        className={cn(HEADER, open ? "rounded-t-[4px]" : "rounded-[4px]", "shadow-[0_1px_1px_0_rgba(16,24,40,0.02)]")}
      >
        <RoleTile role="tool" failed={failed} />
        <span className="flex shrink-0 items-center gap-1.5">
          <span className={cn(LABEL, failed && "text-destructive")}>{name}</span>
          <CopyButton value={name} label={`Copy ${name} name`} iconOnly className={INLINE_COPY} />
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
            {!open && <span className="min-w-0 truncate text-[13px] leading-[1.2]">{result}</span>}
          </button>
        ) : (
          <span className={cn("min-w-0 truncate text-[13px] leading-[1.2] tracking-[-0.26px]", tone)}>
            {result || "No output"}
          </span>
        )}
        <CopyButton value={result} label={`Copy ${name} result`} iconOnly className={CARD_COPY} />
      </div>
      {open && (
        <pre
          className={cn(
            "max-h-80 overflow-auto rounded-b-[4px] border-[0.67px] border-t-0 border-trace-card-border bg-trace-chip px-3 py-2 font-mono text-[12px] leading-[1.5] break-words whitespace-pre-wrap",
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
export function Section({ title, children }: { title: string; children: React.ReactNode }) {
  const [open, setOpen] = useState(true);
  return (
    <section aria-label={title}>
      <button
        type="button"
        aria-expanded={open}
        onClick={() => setOpen((prev) => !prev)}
        className="sticky top-0 z-sticky-pinned mb-2 flex h-10 w-full cursor-pointer items-center gap-2.5 bg-trace-surface py-1 pr-3 pl-2 text-left transition-colors duration-150 hover:bg-trace-chip motion-reduce:transition-none"
      >
        <FoldChevron open={open} className="size-3 shrink-0 text-trace-text" />
        <span className="text-[13px] leading-[1.2] font-medium tracking-[-0.26px] text-trace-text">{title}</span>
      </button>
      <div hidden={!open} inert={!open} className="flex flex-col gap-2.5 pr-4 pb-4 pl-7">
        {children}
      </div>
    </section>
  );
}
