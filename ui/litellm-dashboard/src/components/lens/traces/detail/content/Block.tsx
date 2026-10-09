"use client";

import { useState } from "react";

import CopyButton from "@/components/shared/CopyButton";
import { cn } from "@/lib/cva.config";

import { FoldChevron } from "../../ui/Collapse";

const COPY =
  "size-6 shrink-0 rounded-sm text-muted-foreground opacity-0 transition-opacity duration-150 group-hover/block:opacity-100 focus-visible:opacity-100 motion-reduce:transition-none";

interface BlockProps {
  icon: React.ReactNode;
  label: React.ReactNode;
  /** Names the fold toggle and copy button for assistive tech. */
  name: string;
  copyValue?: string;
  meta?: React.ReactNode;
  tone?: "default" | "muted" | "failed";
  defaultOpen?: boolean;
  children?: React.ReactNode;
}

/** A titled card in the content tab (a message, a tool call, a field set) whose body folds from its header. */
export function Block({
  icon,
  label,
  name,
  copyValue,
  meta,
  tone = "default",
  defaultOpen = true,
  children,
}: BlockProps) {
  const [open, setOpen] = useState(defaultOpen);
  const foldable = children !== undefined && children !== null && children !== false;
  const expanded = open && foldable;
  return (
    <article
      className={cn(
        "group/block overflow-hidden rounded-lg border bg-card",
        tone === "failed" ? "border-destructive/40" : "border-border",
      )}
    >
      <header className="flex min-h-9 items-center gap-2 py-1 pr-1.5 pl-2">
        {foldable ? (
          <button
            type="button"
            aria-expanded={open}
            aria-label={`${open ? "Collapse" : "Expand"} ${name}`}
            onClick={() => setOpen((value) => !value)}
            className="flex min-w-0 flex-1 cursor-pointer items-center gap-2 rounded-sm py-0.5 text-left outline-none focus-visible:ring-2 focus-visible:ring-ring"
          >
            <FoldChevron open={open} className="size-3.5 shrink-0 text-muted-foreground" />
            {icon}
            <span className="min-w-0 truncate text-sm font-medium text-foreground">{label}</span>
          </button>
        ) : (
          <span className="flex min-w-0 flex-1 items-center gap-2 py-0.5 pl-5.5">
            {icon}
            <span className="min-w-0 truncate text-sm font-medium text-foreground">{label}</span>
          </span>
        )}
        {meta}
        {copyValue !== undefined && (
          <CopyButton variant="action" value={copyValue} label={`Copy ${name}`} iconOnly className={COPY} />
        )}
      </header>
      {expanded && (
        <div className={cn("border-t border-border/70 px-3 py-3", tone === "muted" && "[&_*]:text-muted-foreground")}>
          {children}
        </div>
      )}
    </article>
  );
}

const BADGE_TONE = {
  user: "bg-trace-chain-soft text-trace-chain",
  assistant: "bg-trace-llm-soft text-trace-llm",
  tool: "bg-trace-tool-soft text-trace-tool",
  neutral: "bg-muted text-muted-foreground",
  failed: "bg-destructive/10 text-destructive",
} as const;

export type BadgeTone = keyof typeof BADGE_TONE;

export function BlockBadge({ tone, children }: { tone: BadgeTone; children: React.ReactNode }) {
  return (
    <span className={cn("grid size-5 shrink-0 place-items-center rounded-md [&_svg]:size-3", BADGE_TONE[tone])}>
      {children}
    </span>
  );
}
