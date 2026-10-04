import { cn } from "@/lib/cva.config";

import { type Conclusion } from "../../model/live";

export function ConclusionsPanel({ groups, scope }: { groups: readonly Conclusion[]; scope: string }) {
  return (
    <ol aria-label="Conclusions" className="divide-y divide-border/60">
      {groups.map((group) => (
        <li
          key={group.checkId}
          data-check={group.checkId}
          title={group.latest}
          className="grid grid-cols-[0.75rem_minmax(0,1fr)_auto] items-start gap-2 px-4 py-2 text-[12px] motion-safe:animate-in motion-safe:fade-in"
        >
          <span
            aria-hidden="true"
            className={cn("mt-1.5 size-1.5 rounded-full", group.issue ? "bg-[#e5484d]" : "bg-muted-foreground/40")}
          />
          <span className="line-clamp-2 text-foreground">{group.label}</span>
          <span className="font-mono tabular-nums text-foreground">{group.count}</span>
        </li>
      ))}
      {!groups.length && <li className="px-4 py-3 text-[12px] text-muted-foreground">Nothing flagged yet.</li>}
      {groups.length > 0 && scope && <li className="px-4 py-1.5 text-[11px] text-muted-foreground">{scope}</li>}
    </ol>
  );
}
