"use client";

import { useState } from "react";

import { cn } from "@/lib/cva.config";

import { FoldChevron } from "../../ui/Collapse";
import { Markdown } from "./Markdown";
import { MessageList } from "./Messages";
import type { FieldEntry, FieldNode } from "./payload";

const LONG_TEXT_CHARS = 90;
const OPEN_CHILD_LIMIT = 8;
const ID_KEY = /(^|_)id$/i;

const ROW = "grid min-w-0 grid-cols-[minmax(5rem,32%)_minmax(0,1fr)] items-baseline gap-x-3 py-1.5";
const KEY = "flex min-w-0 items-baseline gap-1 text-xs leading-5 text-muted-foreground";
const VALUE = "min-w-0 text-sm leading-5 break-words text-foreground";

const isLongText = (text: string): boolean => text.length > LONG_TEXT_CHARS || text.includes("\n");

const nodeSize = (node: FieldNode): number => {
  if (node.kind === "object") return node.entries.length;
  if (node.kind === "array") return node.items.length;
  if (node.kind === "messages") return node.messages.length;
  return 0;
};

const summary = (node: FieldNode): string => {
  if (node.kind === "object") return `{ ${node.entries.length} ${node.entries.length === 1 ? "key" : "keys"} }`;
  if (node.kind === "array") return `[ ${node.items.length} ${node.items.length === 1 ? "item" : "items"} ]`;
  if (node.kind === "messages") return `${node.messages.length} ${node.messages.length === 1 ? "message" : "messages"}`;
  return node.text.replace(/\s+/g, " ");
};

const isInline = (node: Extract<FieldNode, { kind: "scalar" | "text" }>): boolean =>
  node.kind === "scalar" || (node.format !== "markdown" && !isLongText(node.text));

const startsOpen = (node: FieldNode, depth: number): boolean => {
  if (node.kind === "text") return node.format === "markdown";
  if (node.kind === "scalar") return false;
  return depth === 0 && nodeSize(node) <= OPEN_CHILD_LIMIT;
};

function TextValue({ text, format, mono }: { text: string; format: "markdown" | "code" | "plain"; mono: boolean }) {
  if (format === "markdown") return <Markdown text={text} />;
  return (
    <pre
      className={cn(
        VALUE,
        "max-h-96 overflow-auto whitespace-pre-wrap",
        (mono || format === "code") && "font-mono text-xs",
      )}
    >
      {text}
    </pre>
  );
}

function Children({ node, depth, mono }: { node: FieldNode; depth: number; mono: boolean }) {
  switch (node.kind) {
    case "object":
      return <FieldList entries={node.entries} depth={depth + 1} mono={mono} />;
    case "array":
      return (
        <FieldList entries={node.items.map((item, i): FieldEntry => [String(i), item])} depth={depth + 1} mono={mono} />
      );
    case "messages":
      return (
        <div className="py-1.5">
          <MessageList messages={node.messages} />
        </div>
      );
    case "text":
      return <TextValue text={node.text} format={node.format} mono={mono} />;
    case "scalar":
      return null;
  }
}

function FieldRow({ name, node, depth, mono }: { name: string; node: FieldNode; depth: number; mono: boolean }) {
  const [open, setOpen] = useState(() => startsOpen(node, depth));
  const monoValue = mono || ID_KEY.test(name);
  if ((node.kind === "scalar" || node.kind === "text") && isInline(node)) {
    const monoText = monoValue || node.kind === "scalar" || node.format === "code";
    return (
      <li className={ROW}>
        <span className={cn(KEY, "pl-4.5")}>
          <span className="break-all">{name}</span>
        </span>
        <span className={cn(VALUE, "whitespace-pre-wrap", monoText && "font-mono text-xs leading-5")}>{node.text}</span>
      </li>
    );
  }

  const block = node.kind === "text" || node.kind === "messages";
  return (
    <li className="min-w-0">
      <button
        type="button"
        aria-expanded={open}
        aria-label={`${open ? "Collapse" : "Expand"} ${name}`}
        onClick={() => setOpen((value) => !value)}
        className={cn(
          ROW,
          "w-full cursor-pointer rounded-sm text-left outline-none hover:bg-muted/50 focus-visible:ring-2 focus-visible:ring-ring",
        )}
      >
        <span className={KEY}>
          <FoldChevron open={open} className="size-3.5 shrink-0 self-center" />
          <span className="break-all">{name}</span>
        </span>
        {!open && (
          <span className={cn(VALUE, "truncate font-mono text-xs text-muted-foreground")}>{summary(node)}</span>
        )}
      </button>
      {open && (
        <div className={cn("min-w-0", block ? "pb-2 pl-4.5" : "ml-1.5 border-l border-border pl-3")}>
          <Children node={node} depth={depth} mono={monoValue} />
        </div>
      )}
    </li>
  );
}

function FieldList({ entries, depth, mono }: { entries: readonly FieldEntry[]; depth: number; mono: boolean }) {
  return (
    <ul className="flex min-w-0 flex-col">
      {entries.map(([name, node], i) => (
        <FieldRow key={`${name}-${i}`} name={name} node={node} depth={depth} mono={mono} />
      ))}
    </ul>
  );
}

/** Nested key / value tree: short values inline, long text and nested objects fold open in place. */
export function FieldTree({ entries, mono = false }: { entries: readonly FieldEntry[]; mono?: boolean }) {
  return <FieldList entries={entries} depth={0} mono={mono} />;
}
