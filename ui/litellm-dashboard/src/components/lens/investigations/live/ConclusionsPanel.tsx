"use client";

import { useEffect, useRef, useState } from "react";

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

export function ConclusionsPanel({
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
  if (!groups.length) return <p className="text-[12px] text-muted-foreground">Nothing concluded yet.</p>;
  return (
    <div className="flex flex-col gap-1.5">
      {SECTIONS.map(({ issue, title }) => (
        <GroupList
          key={title}
          title={title}
          groups={groups.filter((group) => group.issue === issue)}
          total={total}
          selected={selected}
          flashing={flashing}
          onSelect={onSelect}
        />
      ))}
      {scope && <p className="text-[11px] text-muted-foreground">{scope}</p>}
    </div>
  );
}

const SECTIONS = [
  { issue: true, title: "Issues" },
  { issue: false, title: "Patterns" },
] as const;

function GroupList({
  title,
  groups,
  total,
  selected,
  flashing,
  onSelect,
}: {
  title: string;
  groups: readonly Conclusion[];
  total: number;
  selected: string | null;
  flashing: ReadonlySet<string>;
  onSelect: (key: string | null) => void;
}) {
  if (!groups.length) return null;
  return (
    <>
      <h3 className="pt-1 text-[11px] text-muted-foreground">{title}</h3>
      <ol aria-label={title} className="flex flex-col gap-1.5">
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
                        group.issue ? "bg-[#e5484d]" : "bg-muted-foreground/40",
                      )}
                    />
                    <span className="line-clamp-2 text-[13px] leading-snug text-foreground">{group.label}</span>
                  </span>
                  <span className="text-[20px] leading-none font-semibold tabular-nums text-foreground">
                    {group.count}
                  </span>
                </span>
                <span aria-hidden="true" className="h-1 w-full overflow-hidden rounded-full bg-muted">
                  <span
                    className={cn(
                      "block h-full rounded-full transition-[width] duration-300 motion-reduce:transition-none",
                      group.issue ? "bg-[#e5484d]/70" : "bg-muted-foreground/40",
                    )}
                    style={{ width: `${share(group.count, total) * 100}%` }}
                  />
                </span>
              </button>
            </li>
          );
        })}
      </ol>
    </>
  );
}
