"use client";

import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import { watches } from "./lensData";

const dotColors = ["#8b5cf6", "#22b3e8", "#e3a32b", "#eb6b93", "#22b3e8", "#8b5cf6", "#e3a32b", "#eb6b93"];
const lensBlue = { light: "#0011b3", dark: "#8b9bff" };
const columns = 96;
const rows = 9;
const cell = 7;

function dotColor(lit: boolean, pastLens: boolean, incoming: string, blue: string): string {
  if (!lit) return "#94a3b8";
  return pastLens ? blue : incoming;
}

function DotFlow({ active }: { active: readonly string[] }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const node = canvas.current;
    const context = node?.getContext("2d");
    if (!node || !context) return;
    const colors = active.length ? active : ["#94a3b8"];
    const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const blue = document.documentElement.classList.contains("dark") ? lensBlue.dark : lensBlue.light;
    const lensColumn = Math.floor(columns * 0.62);
    const draw = (time: number) => {
      context.clearRect(0, 0, node.width, node.height);
      for (let row = 0; row < rows; row++) {
        for (let column = 0; column < columns; column++) {
          const x = column * cell + cell / 2;
          const y = row * cell + cell / 2;
          const center = (rows - 1) / 2;
          const funnel =
            column < lensColumn ? Math.abs(row - center) <= center * (1 - column / lensColumn) + 0.6 : row === center;
          const wave = Math.sin(column * 0.55 - time / 260 + row * 1.7);
          const lit = funnel && wave > 0.35;
          context.globalAlpha = lit ? 0.9 : 0.12;
          context.fillStyle = dotColor(lit, column >= lensColumn, colors[(row + column) % colors.length], blue);
          context.beginPath();
          context.arc(x, y, lit ? 1.6 : 1, 0, Math.PI * 2);
          context.fill();
        }
      }
      context.globalAlpha = 1;
      context.strokeStyle = blue;
      context.lineWidth = 1.5;
      const lx = lensColumn * cell - 1;
      context.beginPath();
      context.moveTo(lx + 3, 1);
      context.lineTo(lx, 1);
      context.lineTo(lx, rows * cell - 1);
      context.lineTo(lx + 3, rows * cell - 1);
      context.stroke();
    };
    if (still) {
      draw(0);
      return;
    }
    let frame = requestAnimationFrame(function loop(time) {
      draw(time);
      frame = requestAnimationFrame(loop);
    });
    return () => cancelAnimationFrame(frame);
  }, [active]);
  return <canvas ref={canvas} aria-hidden="true" width={columns * cell} height={rows * cell} className="h-16 w-full" />;
}

export function WatchPicker({
  selected,
  onChange,
}: {
  selected: ReadonlySet<string>;
  onChange: (next: ReadonlySet<string>) => void;
}) {
  const [cursor, setCursor] = useState(0);
  const items = useRef<(HTMLLIElement | null)[]>([]);
  const toggle = (id: string) =>
    onChange(new Set(selected.has(id) ? [...selected].filter((item) => item !== id) : [...selected, id]));
  const move = (index: number) => {
    const next = (index + watches.length) % watches.length;
    setCursor(next);
    items.current[next]?.focus();
  };
  const onKey = (event: KeyboardEvent<HTMLUListElement>) => {
    const digit = Number(event.key);
    if (event.key === "ArrowDown" || event.key === "j") move(cursor + 1);
    else if (event.key === "ArrowUp" || event.key === "k") move(cursor - 1);
    else if (event.key === " " || event.key === "Enter") toggle(watches[cursor].id);
    else if (digit >= 1 && digit <= watches.length) {
      move(digit - 1);
      toggle(watches[digit - 1].id);
    } else return;
    event.preventDefault();
  };
  const activeColors = watches.flatMap((watch, index) => (selected.has(watch.id) ? [dotColors[index]] : []));

  return (
    <fieldset className="space-y-2">
      <div className="flex items-end justify-between gap-3">
        <legend className="text-sm font-medium">Watch for</legend>
        <span className="text-xs tabular-nums text-muted-foreground">
          {selected.size}/{watches.length} on
        </span>
      </div>
      <div className="overflow-hidden rounded-lg border bg-background">
        <div className="border-b bg-muted/20 px-3 pt-3 pb-2">
          <DotFlow active={activeColors} />
          <div className="mt-1 flex justify-between text-[11px] text-muted-foreground">
            <span>your agents</span>
            <span className="hidden sm:block">↑↓ move · space toggle · 1–{watches.length}</span>
            <span>findings</span>
          </div>
        </div>
        <ul role="listbox" aria-label="Watch for" aria-multiselectable="true" onKeyDown={onKey} className="py-1">
          {watches.map((watch, index) => {
            const on = selected.has(watch.id);
            const focused = index === cursor;
            return (
              <li
                key={watch.id}
                ref={(node) => {
                  items.current[index] = node;
                }}
                role="option"
                aria-selected={on}
                tabIndex={focused ? 0 : -1}
                onFocus={() => setCursor(index)}
                onClick={() => {
                  setCursor(index);
                  toggle(watch.id);
                }}
                className={`group relative grid cursor-pointer grid-cols-[1rem_1.25rem_minmax(0,1fr)_auto] items-start gap-2 px-3 py-2 outline-none transition-colors ${
                  focused ? "bg-muted/60" : "hover:bg-muted/30"
                }`}
              >
                <span
                  aria-hidden="true"
                  className={`pt-px text-sm font-medium text-[#0011b3] transition-opacity dark:text-[#8b9bff] ${
                    focused ? "opacity-100" : "opacity-0"
                  }`}
                >
                  ❯
                </span>
                <span
                  aria-hidden="true"
                  className={`mt-1 flex size-3.5 items-center justify-center rounded-[3px] border transition-colors ${
                    on ? "border-foreground bg-foreground" : "border-muted-foreground/40"
                  }`}
                >
                  <svg
                    viewBox="0 0 16 16"
                    className={`size-2.5 text-background transition-transform duration-150 ${on ? "scale-100" : "scale-0"}`}
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
                <span className="grid gap-0.5">
                  <span className={`text-sm font-medium ${on ? "text-foreground" : "text-muted-foreground"}`}>
                    {watch.name}
                  </span>
                  <span className="text-xs text-muted-foreground">{watch.summary}</span>
                </span>
                <span aria-hidden="true" className="pt-0.5 text-[11px] tabular-nums text-muted-foreground/60">
                  {index + 1}
                </span>
              </li>
            );
          })}
        </ul>
      </div>
    </fieldset>
  );
}
