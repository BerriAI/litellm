"use client";

import { useRef, useState, type KeyboardEvent } from "react";
import { Check, Plus } from "lucide-react";
import { watches } from "../model/watches";
import { cn } from "@/lib/cva.config";

const COLUMNS = 2;

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
    else if (event.key === "ArrowDown" || event.key === "j") move(cursor + COLUMNS);
    else if (event.key === "ArrowUp" || event.key === "k") move(cursor - COLUMNS);
    else if (digit >= 1 && digit <= watches.length) {
      move(digit - 1);
      toggle(watches[digit - 1].id);
    } else return;
    event.preventDefault();
  };

  return (
    <fieldset className="grid gap-2">
      <div className="flex items-baseline justify-between gap-3">
        <legend className="text-sm font-medium">Watch for</legend>
        <span className="text-xs tabular-nums text-muted-foreground">
          {selected.size} of {watches.length}
        </span>
      </div>
      <div role="group" aria-label="Watch for" onKeyDown={onKey} className="grid grid-cols-2 gap-1.5">
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
              title={watch.instruction}
              tabIndex={index === cursor ? 0 : -1}
              data-state={on ? "active" : "inactive"}
              onFocus={() => setCursor(index)}
              onClick={() => toggle(watch.id)}
              className={cn(
                "group/watch flex items-start gap-2.5 rounded-md border px-2.5 py-2 text-left outline-none transition-colors focus-visible:ring-2 focus-visible:ring-ring/50",
                "data-[state=active]:border-foreground/25 data-[state=active]:bg-muted/50",
                "data-[state=inactive]:hover:bg-muted/40",
              )}
            >
              <span
                aria-hidden="true"
                className="mt-0.5 flex size-4 shrink-0 items-center justify-center rounded-sm border border-input transition-colors group-data-[state=active]/watch:border-foreground group-data-[state=active]/watch:bg-foreground group-data-[state=active]/watch:text-background"
              >
                {on && <Check className="size-3" strokeWidth={3} />}
              </span>
              <span className="grid min-w-0 gap-0.5">
                <span className="text-sm leading-5 font-medium capitalize">{watch.name}</span>
                <span className="text-xs leading-snug text-muted-foreground">{watch.summary}</span>
              </span>
            </button>
          );
        })}
      </div>
      <button
        type="button"
        onClick={onAddCustom}
        className="flex items-center gap-2 justify-self-start rounded-md px-1 py-1 text-sm text-muted-foreground outline-none hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring/50"
      >
        <Plus aria-hidden="true" className="size-4" />
        Add your own check
      </button>
    </fieldset>
  );
}
