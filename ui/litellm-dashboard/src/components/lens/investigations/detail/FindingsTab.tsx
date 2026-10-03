"use client";
import { ListRow } from "@/components/shared/ListRow";

import { ChevronRight, CheckCircle2 } from "lucide-react";
import { Button } from "@/components/ui/button";

import { TabsContent } from "@/components/ui/tabs";
import { type Lens, type Finding, type Job } from "../../model/types";
import { cn } from "@/lib/cva.config";

const priorityColors = { high: "bg-red-500", medium: "bg-amber-500", low: "bg-slate-400" };
function emptyFindingTitle(active: boolean, scanned: boolean, status?: string) {
  if (status === "failed" || status === "cancelled") return "No findings from this run";
  if (active) return "Your findings will appear here";
  return scanned ? "No matching findings" : "Ready for the first analysis";
}

export function FindingsTab({
  kind,
  setKind,
  batchFindings,
  filter,
  setFilter,
  visibleFindings,
  setFindingId,
  active,
  lens,
  job,
}: {
  kind: "issue" | "pattern";
  setKind: (kind: "issue" | "pattern") => void;
  batchFindings: Finding[];
  filter: string;
  setFilter: (filter: string) => void;
  visibleFindings: Finding[];
  setFindingId: (id: string) => void;

  active: Job | undefined;
  lens: Lens;
  job: Job | undefined;
}) {
  return (
    <TabsContent value="findings" className="pt-4 space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex gap-1" aria-label="Finding category">
          <Button size="sm" variant={kind === "issue" ? "secondary" : "ghost"} onClick={() => setKind("issue")}>
            Needs attention ({batchFindings.filter((f) => f.kind === "issue" && f.status === "open").length ?? 0})
          </Button>
          <Button size="sm" variant={kind === "pattern" ? "secondary" : "ghost"} onClick={() => setKind("pattern")}>
            Patterns ({batchFindings.filter((f) => f.kind === "pattern" && f.status === "open").length ?? 0})
          </Button>
        </div>
        <select
          aria-label="Finding status"
          className="rounded-md border bg-background px-2 py-1 text-xs"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
        >
          <option value="open">Open</option>
          <option value="resolved">Resolved</option>
          <option value="dismissed">Dismissed</option>
          <option value="all">All statuses</option>
        </select>
      </div>
      <p className="text-xs text-muted-foreground">
        {kind === "issue"
          ? "Problems worth investigating, highest priority first."
          : "Useful behavior and trends. These do not necessarily need a fix."}
      </p>
      <div className="divide-y border-y">
        {visibleFindings.map((f) => (
          <ListRow
            key={f.id}
            onClick={() => {
              setFindingId(f.id);
            }}
            className="flex w-full gap-3 py-4 text-left hover:bg-muted/30 focus-visible:outline-2 focus-visible:outline-ring"
          >
            <span
              data-state={f.priority ?? "medium"}
              className={cn("mt-1 size-2 shrink-0 rounded-full", priorityColors[f.priority ?? "medium"])}
              aria-label={`${f.priority} priority`}
            />
            <div className="min-w-0 flex-1">
              <p className="text-sm font-medium">{f.title}</p>
              <p className="mt-1 line-clamp-2 text-sm text-muted-foreground">{f.description}</p>
              <p className="mt-2 text-xs text-muted-foreground">
                {f.occurrences?.length ?? 0} linked {f.occurrences?.length === 1 ? "run" : "runs"} ·{" "}
                {f.kind === "issue" ? `${f.priority} priority` : "Pattern"}
              </p>
            </div>
            <ChevronRight className="size-4 shrink-0 text-muted-foreground" />
          </ListRow>
        ))}
        {visibleFindings.length === 0 && (
          <div className="px-6 py-14 text-center">
            <CheckCircle2 className="mx-auto mb-3 size-5 text-muted-foreground" />
            <p className="text-sm font-medium">{emptyFindingTitle(!!active, !!lens.last_scan_at, job?.status)}</p>
            <p className="mt-2 text-xs text-muted-foreground">
              {active
                ? "Lens is reviewing the selected activity."
                : "Findings reflect the runs analyzed, not a guarantee about all activity."}
            </p>
          </div>
        )}
      </div>
    </TabsContent>
  );
}
