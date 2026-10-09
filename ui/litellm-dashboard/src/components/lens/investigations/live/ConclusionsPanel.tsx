"use client";

import { memo, useEffect, useRef, useState } from "react";

import { cn } from "@/lib/cva.config";

import { grownGroups, share, type Conclusion } from "../../model/live";

const FLASH_MS = 150;

function useFlashing(groups: readonly Conclusion[]): ReadonlySet<string> {
  const previous = useRef(groups);
  const [flashing, setFlashing] = useState<ReadonlySet<string>>(new Set());
  useEffect(() => {
    const grown = grownGroups(previous.current, groups);
    previous.current = groups;
    if (!grown.size) return;
    setFlashing(grown);
    const timer = window.setTimeout(() => setFlashing(new Set()), FLASH_MS);
    return () => window.clearTimeout(timer);
  }, [groups]);
  return flashing;
}

export const ConclusionsPanel = memo(function ConclusionsPanel({
  groups,
  total,
  scope,
  selected,
  onSelect,
}: {
  groups: readonly Conclusion[];
  total: number;
  scope: string;
  selected: string | null;
  onSelect: (key: string | null) => void;
}) {
  const flashing = useFlashing(groups);
  if (!groups.length) return <p className="text-xs text-muted-foreground">No observations flagged yet.</p>;
  return (
    <div className="flex flex-col gap-1.5">
      <ol aria-label="Observations by check" className="flex flex-col gap-1.5">
        {groups.map((group) => {
          const active = selected === group.key;
          return (
            <li key={group.key}>
              <button
                type="button"
                aria-pressed={active}
                title={group.latest}
                onClick={() => onSelect(active ? null : group.key)}
                className={cn(
                  "flex w-full flex-col gap-1.5 rounded-lg px-3 py-2 text-left transition-colors duration-150 active:scale-[0.99] motion-reduce:transition-none",
                  active ? "bg-background ring-[1.5px] ring-inset ring-foreground" : "bg-muted/50 hover:bg-muted",
                  flashing.has(group.key) && "bg-muted",
                )}
              >
                <span className="flex items-start justify-between gap-3">
                  <span className="flex min-w-0 items-start gap-2">
                    <span
                      aria-hidden="true"
                      className={cn(
                        "mt-[0.4rem] size-1.5 shrink-0 rounded-full",
                        group.issue ? "bg-destructive" : "bg-muted-foreground/40",
                      )}
                    />
                    <span className="flex min-w-0 flex-col gap-0.5">
                      <span className="line-clamp-2 text-sm leading-snug font-medium text-foreground">
                        {group.label}
                      </span>
                      <span className="line-clamp-1 text-xs text-muted-foreground">{group.latest}</span>
                    </span>
                  </span>
                  <span className="flex shrink-0 flex-col items-end gap-0.5">
                    <span
                      className={cn(
                        "text-xl leading-none font-semibold tabular-nums",
                        group.count ? "text-foreground" : "text-muted-foreground",
                      )}
                    >
                      {group.count}
                    </span>
                    {group.noted > 0 && (
                      <span className="text-xs tabular-nums text-muted-foreground">{group.noted} noted</span>
                    )}
                  </span>
                </span>
                <span aria-hidden="true" className="h-1 w-full overflow-hidden rounded-full bg-muted">
                  <span
                    className={cn(
                      "block h-full rounded-full transition-[width] duration-300 motion-reduce:transition-none",
                      group.issue ? "bg-destructive/70" : "bg-muted-foreground/40",
                    )}
                    style={{ width: `${share(group.count, total) * 100}%` }}
                  />
                </span>
              </button>
            </li>
          );
        })}
      </ol>
      {scope && <p className="text-xs text-muted-foreground">{scope}</p>}
    </div>
  );
});
