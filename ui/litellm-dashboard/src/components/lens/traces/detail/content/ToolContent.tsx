"use client";

import { useState } from "react";
import { cn } from "@/lib/cva.config";
import { FieldTree } from "./FieldTree";
import { MessageList } from "./Messages";
import { fieldNode, toolAction, toolResult, type FieldNode } from "./payload";

export function ToolText({ text, label }: { text: string; label: string }) {
  const [expanded, setExpanded] = useState(false);
  const preview = text.split("\n").slice(0, 8).join("\n").slice(0, 1200);
  const truncated = preview.length < text.length;
  return (
    <div className="min-w-0">
      <pre className="whitespace-pre-wrap break-words font-mono text-xs leading-5 wrap-anywhere">
        {expanded ? text : preview || "No output recorded"}
        {!expanded && truncated && "\n…"}
      </pre>
      {truncated && (
        <button
          type="button"
          aria-expanded={expanded}
          aria-label={`${expanded ? "Collapse" : "Expand"} ${label}`}
          onClick={() => setExpanded((value) => !value)}
          className="mt-2 rounded-sm text-xs font-medium text-muted-foreground underline-offset-4 hover:text-foreground hover:underline focus-visible:outline-2 focus-visible:outline-ring"
        >
          {expanded ? "Show less" : `Show full ${label} (${text.length.toLocaleString()} characters)`}
        </button>
      )}
    </div>
  );
}

function ToolValue({ node, label }: { node: FieldNode; label: string }) {
  if (node.kind === "messages") return <MessageList messages={node.messages} />;
  if (node.kind === "text" || node.kind === "scalar") return <ToolText text={node.text} label={label} />;
  const entries = node.kind === "object" ? node.entries : node.items.map((item, i) => [String(i), item] as const);
  return entries.length ? (
    <FieldTree entries={entries} />
  ) : (
    <ToolText text={node.kind === "object" ? "{}" : "[]"} label={label} />
  );
}

export function ToolArguments({ args }: { args: unknown }) {
  const node = fieldNode(args);
  const action = toolAction(args);
  const rest = node.kind === "object" ? node.entries.filter(([key]) => key !== action?.key) : [];
  if (!action) return <ToolValue node={node} label="arguments" />;
  return (
    <div className="min-w-0 space-y-2">
      <div className="rounded-md bg-muted/40 px-3 py-2.5">
        <div className="mb-1.5 text-xs text-muted-foreground">{action.key}</div>
        <ToolText text={action.text} label={action.key} />
      </div>
      {rest.length > 0 && <FieldTree entries={rest} />}
    </div>
  );
}

export function ToolOutput({ result, failed = false }: { result: string; failed?: boolean }) {
  const { body, metadata } = toolResult(result);
  return (
    <div
      role="group"
      aria-label={failed ? "Failed tool result" : "Tool result"}
      className={cn("min-w-0 space-y-3", failed && "text-destructive")}
    >
      <ToolValue node={body} label="result" />
      {metadata.length > 0 && (
        <div className="border-t border-border/70 pt-1">
          <FieldTree entries={metadata} />
        </div>
      )}
    </div>
  );
}
