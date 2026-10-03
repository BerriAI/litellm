"use client";

import { useRef, useState, type KeyboardEvent } from "react";
import { watches } from "./watches";
import { cn } from "@/lib/cva.config";
import { DotFlow } from "../DotFlow";

const dotColors = ["#8b5cf6", "#22b3e8", "#e3a32b", "#eb6b93", "#22b3e8", "#8b5cf6", "#e3a32b", "#eb6b93"];

export function WatchPicker({
  selected,
  onChange,
  onAddCustom,
}: {
  selected: ReadonlySet<string>;
  onChange: (next: ReadonlySet<string>) => void;
  onAddCustom: () => void;
}) {
  const [cursor, setCursor] = useState(0);
  const items = useRef<(HTMLButtonElement | null)[]>([]);
  const toggle = (id: string) =>
    onChange(new Set(selected.has(id) ? [...selected].filter((item) => item !== id) : [...selected, id]));
  const move = (index: number) => {
    const next = (index + watches.length) % watches.length;
    setCursor(next);
    items.current[next]?.focus();
  };
  const onKey = (event: KeyboardEvent<HTMLDivElement>) => {
    const digit = Number(event.key);
    if (event.key === "ArrowRight" || event.key === "l") move(cursor + 1);
    else if (event.key === "ArrowLeft" || event.key === "h") move(cursor - 1);
    else if (event.key === "ArrowDown" || event.key === "j") move(cursor + 4);
    else if (event.key === "ArrowUp" || event.key === "k") move(cursor - 4);
    else if (digit >= 1 && digit <= watches.length) {
      move(digit - 1);
      toggle(watches[digit - 1].id);
    } else return;
    event.preventDefault();
  };
  const activeColors = watches.flatMap((watch, index) => (selected.has(watch.id) ? [dotColors[index]] : []));

  return (
    <fieldset className="space-y-2.5">
      <div className="flex items-end justify-between gap-3">
        <legend className="text-sm font-medium">Watch for</legend>
        <span className="text-xs tabular-nums text-muted-foreground">
          {selected.size} of {watches.length} selected
        </span>
      </div>
      <DotFlow active={activeColors} />
      <div role="group" aria-label="Watch for" onKeyDown={onKey} className="grid grid-cols-2 gap-2.5 sm:grid-cols-4">
        {watches.map((watch, index) => {
          const on = selected.has(watch.id);
          return (
            <button
              key={watch.id}
              ref={(node) => {
                items.current[index] = node;
              }}
              type="button"
              aria-pressed={on}
              title={watch.summary}
              tabIndex={index === cursor ? 0 : -1}
              data-state={on ? "active" : "inactive"}
              onFocus={() => setCursor(index)}
              onClick={() => toggle(watch.id)}
              className={cn(
                "flex h-[5.25rem] flex-col justify-start gap-1 rounded-xl px-3.5 py-3 text-left outline-none transition-all duration-200 ease-out focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring active:scale-[0.97]",
                "data-[state=active]:bg-background data-[state=active]:text-foreground data-[state=active]:ring-[1.5px] data-[state=active]:ring-inset data-[state=active]:ring-foreground",
                "data-[state=inactive]:bg-muted/60 data-[state=inactive]:text-muted-foreground data-[state=inactive]:hover:bg-muted data-[state=inactive]:hover:text-foreground",
              )}
            >
              <span className="flex items-center justify-between gap-1">
                <span className="text-sm font-medium">{watch.name}</span>
                <svg
                  viewBox="0 0 16 16"
                  aria-hidden="true"
                  data-state={on ? "active" : "inactive"}
                  className="size-3 transition-all duration-200 data-[state=active]:scale-100 data-[state=active]:opacity-100 data-[state=inactive]:scale-50 data-[state=inactive]:opacity-0"
                >
                  <path
                    d="M3 8.5l3.2 3.2L13 5"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2.2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              </span>
              <span className="line-clamp-2 text-xs leading-snug text-muted-foreground">{watch.summary}</span>
            </button>
          );
        })}
      </div>
      <button
        type="button"
        onClick={onAddCustom}
        className="flex h-11 w-full items-center gap-2.5 rounded-xl bg-muted/60 px-3.5 text-left text-sm outline-none transition-colors hover:bg-muted focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
      >
        <span
          aria-hidden="true"
          className="flex size-5 items-center justify-center rounded-full bg-background text-sm leading-none"
        >
          +
        </span>
        <span className="font-medium">Add your own</span>
        <span className="text-muted-foreground">describe anything else in plain English</span>
      </button>
    </fieldset>
  );
}
