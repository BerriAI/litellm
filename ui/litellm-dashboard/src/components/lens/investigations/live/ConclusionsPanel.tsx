import { useEffect, useRef, type RefObject } from "react";

import { cn } from "@/lib/cva.config";

import { type Conclusion } from "../../model/live";

export interface Arrival {
  key: string;
  checkId: string;
  text: string;
}

const FLY_MS = 550;

function fly(from: DOMRect, to: DOMRect, text: string) {
  const node = document.createElement("div");
  node.textContent = text;
  node.setAttribute("aria-hidden", "true");
  node.className =
    "pointer-events-none fixed z-floating truncate rounded-lg border border-[#e5484d]/30 bg-background px-3 py-2 text-xs font-semibold text-[#e5484d] shadow-lg";
  Object.assign(node.style, { left: `${from.left}px`, top: `${from.top}px`, width: `${from.width}px` });
  document.body.appendChild(node);
  const animation = node.animate(
    [
      { transform: "translate(0, 0)", width: `${from.width}px`, opacity: 1 },
      {
        transform: `translate(${to.left - from.left}px, ${to.top - from.top}px)`,
        width: `${Math.max(200, to.width)}px`,
        opacity: 0.2,
      },
    ],
    { duration: FLY_MS, easing: "cubic-bezier(.3,.7,.2,1)" },
  );
  animation.onfinish = () => node.remove();
  return () => node.remove();
}

function useFlyIn(arrival: Arrival | null, from: RefObject<HTMLElement | null>, list: RefObject<HTMLElement | null>) {
  useEffect(() => {
    const source = from.current;
    const box = list.current;
    if (!arrival || !source || !box || typeof HTMLElement.prototype.animate !== "function") return;
    if (window.matchMedia?.("(prefers-reduced-motion: reduce)").matches) return;
    const target = box.querySelector<HTMLElement>(`[data-check="${CSS.escape(arrival.checkId)}"]`) ?? box;
    return fly(source.getBoundingClientRect(), target.getBoundingClientRect(), arrival.text);
  }, [arrival, from, list]);
}

export function ConclusionsPanel({
  groups,
  reviewed,
  arrival,
  verdictRef,
}: {
  groups: readonly Conclusion[];
  reviewed: number;
  arrival: Arrival | null;
  verdictRef: RefObject<HTMLElement | null>;
}) {
  const list = useRef<HTMLOListElement>(null);
  useFlyIn(arrival, verdictRef, list);
  return (
    <ol ref={list} aria-label="Conclusions" className="flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto px-3 py-2.5">
      {groups.map((group) => (
        <li
          key={group.checkId}
          data-check={group.checkId}
          data-hit={arrival?.checkId === group.checkId || undefined}
          className={cn(
            "rounded-lg border bg-background px-3 py-2 transition-shadow duration-500 motion-safe:animate-in motion-safe:fade-in",
            "data-hit:border-[#e5484d]/30 data-hit:ring-3 data-hit:ring-[#e5484d]/10",
          )}
        >
          <p className="flex items-start gap-1.5 text-[12.5px] font-semibold">
            <span
              aria-hidden="true"
              className={cn("mt-1.5 size-1.5 shrink-0 rounded-full", group.issue ? "bg-[#e5484d]" : "bg-muted-foreground/50")}
            />
            <span className="line-clamp-2">{group.label}</span>
          </p>
          <p className="mt-1 flex justify-between gap-3 font-mono text-[11px] text-muted-foreground">
            <span>
              {group.count} {group.count === 1 ? "trace" : "traces"}
            </span>
            <span>{Math.round((group.count / Math.max(1, reviewed)) * 100)}% of reviewed</span>
          </p>
          {group.latest !== group.label && (
            <p className="mt-1 line-clamp-1 text-[11px] text-muted-foreground" title={group.latest}>
              latest: {group.latest}
            </p>
          )}
        </li>
      ))}
      {!groups.length && <li className="py-4 text-xs text-muted-foreground">Nothing flagged yet.</li>}
    </ol>
  );
}
