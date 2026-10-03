import { useEffect, useRef, type RefObject } from "react";

import { cn } from "@/lib/cva.config";

import { type Conclusion } from "../../model/live";

export interface Arrival {
  key: string;
  checkId: string;
  text: string;
}

const FLY_MS = 400;

function fly(from: DOMRect, to: DOMRect, text: string) {
  const node = document.createElement("div");
  node.textContent = text;
  node.setAttribute("aria-hidden", "true");
  node.className =
    "pointer-events-none fixed z-floating truncate rounded-md border bg-background px-3 py-1.5 text-[12px] font-medium text-[#e5484d] shadow-sm";
  Object.assign(node.style, { left: `${from.left}px`, top: `${from.top}px`, width: `${from.width}px` });
  document.body.appendChild(node);
  const animation = node.animate(
    [
      { transform: "translate(0, 0)", opacity: 1 },
      { transform: `translate(${to.left - from.left}px, ${to.top - from.top}px)`, opacity: 0 },
    ],
    { duration: FLY_MS, easing: "ease-out" },
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
  const list = useRef<HTMLTableSectionElement>(null);
  useFlyIn(arrival, verdictRef, list);
  return (
    <table aria-label="Conclusions" className="w-full table-fixed border-collapse text-left">
      <tbody ref={list}>
        {groups.map((group) => (
          <tr
            key={group.checkId}
            data-check={group.checkId}
            className="border-b border-border/60 text-[12px]"
            title={group.latest}
          >
            <td className="w-6 pl-3 align-top">
              <span
                aria-hidden="true"
                className={cn(
                  "mt-3.5 block size-1.5 rounded-full",
                  group.issue ? "bg-[#e5484d]" : "bg-muted-foreground/40",
                )}
              />
            </td>
            <td className="py-2 pr-2">
              <span className="line-clamp-2 text-foreground">{group.label}</span>
              <span className="text-[11px] text-muted-foreground">
                {Math.round((group.count / Math.max(1, reviewed)) * 100)}% of reviewed
              </span>
            </td>
            <td className="w-12 px-3 py-2 text-right align-top font-mono tabular-nums text-foreground">
              {group.count}
            </td>
          </tr>
        ))}
        {!groups.length && (
          <tr className="text-[12px] text-muted-foreground">
            <td className="px-3 py-6">Nothing flagged yet.</td>
          </tr>
        )}
      </tbody>
    </table>
  );
}
