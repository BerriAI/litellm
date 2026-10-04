"use client";

import { useState } from "react";

import { cn } from "@/lib/cva.config";

import { FoldChevron } from "../../ui/Collapse";

const MODES = ["Formatted", "Raw"] as const;
export type SectionMode = (typeof MODES)[number];

function ModeSwitch({
  mode,
  onChange,
  title,
}: {
  mode: SectionMode;
  onChange: (mode: SectionMode) => void;
  title: string;
}) {
  return (
    <div role="radiogroup" aria-label={`${title} format`} className="flex items-center rounded-md bg-muted p-0.5">
      {MODES.map((option) => (
        <button
          key={option}
          type="button"
          role="radio"
          aria-checked={mode === option}
          onClick={() => onChange(option)}
          className={cn(
            "h-6 rounded-sm px-2 text-xs transition-colors duration-150 motion-reduce:transition-none",
            mode === option
              ? "bg-background font-medium text-foreground shadow-xs"
              : "text-muted-foreground hover:text-foreground",
          )}
        >
          {option}
        </button>
      ))}
    </div>
  );
}

interface SectionProps {
  title: string;
  count?: number;
  defaultOpen?: boolean;
  /** When given, the header offers a Formatted / Raw switch and the body is a render function of the mode. */
  children: React.ReactNode | ((mode: SectionMode) => React.ReactNode);
}

/** Collapsible "Input" / "Output" section with a sticky header. */
export function Section({ title, count, defaultOpen = true, children }: SectionProps) {
  const [open, setOpen] = useState(defaultOpen);
  const [mode, setMode] = useState<SectionMode>("Formatted");
  const switchable = typeof children === "function";
  return (
    <section aria-label={title} className="flex flex-col">
      <div className="sticky top-0 z-sticky-pinned flex h-11 items-center gap-2 bg-background px-4">
        <button
          type="button"
          aria-expanded={open}
          onClick={() => setOpen((value) => !value)}
          className="-ml-1 flex min-w-0 flex-1 cursor-pointer items-center gap-2 rounded-sm px-1 py-1 text-left outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          <FoldChevron open={open} className="size-3.5 shrink-0 text-muted-foreground" />
          <span className="text-sm font-semibold text-foreground">{title}</span>{" "}
          {count !== undefined && (
            <span className="text-xs text-muted-foreground">
              {count} {count === 1 ? "message" : "messages"}
            </span>
          )}
        </button>
        {switchable && open && <ModeSwitch mode={mode} onChange={setMode} title={title} />}
      </div>
      <div hidden={!open} inert={!open} className="flex flex-col gap-2 px-4 pb-5">
        {switchable ? children(mode) : children}
      </div>
    </section>
  );
}
