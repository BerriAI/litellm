"use client";

import { Pencil, Play, Search } from "lucide-react";
import { useState } from "react";

import { useNow } from "@/hooks/useNow";
import { Input } from "@/components/ui/input";
import { formatActivityTimestamp } from "@/utils/activityTimestamp";
import { cn } from "@/lib/cva.config";
import { agoLabel } from "@/components/view_logs/TraceView/lensField";

import { scheduleLabel } from "../model/inbox";
import { lensStatus } from "../model/status";
import { scopeLabel } from "../model/format";
import { type Lens } from "../model/types";

const TH = "px-3 font-medium";
const TH_NUM = "px-3 text-right font-medium";

export function InvestigationList({
  lenses,
  connected,
  readOnly = false,
  onEdit,
  onRunNow,
}: {
  lenses: Lens[];
  connected: boolean;
  readOnly?: boolean;
  onEdit: (id: string) => void;
  onRunNow: (id: string) => void;
}) {
  const [search, setSearch] = useState("");
  const now = useNow(15000);
  const shown = lenses.filter((lens) =>
    `${lens.settings.name} ${scopeLabel(lens.settings)}`.toLowerCase().includes(search.toLowerCase()),
  );
  return (
    <div className="flex min-h-[420px] flex-1 flex-col overflow-hidden border-y border-border bg-card">
      <div className="flex min-h-10 shrink-0 flex-wrap items-center gap-2 border-b border-border bg-card px-3 py-1.5">
        <div className="relative w-full max-w-[380px]">
          <Search className="pointer-events-none absolute top-1/2 left-2.5 size-3.5 -translate-y-1/2 text-muted-foreground" />
          <Input
            aria-label="Search investigations"
            placeholder="Search investigations"
            className="h-7 pl-8 text-[12px]"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>
      </div>
      <div className="min-h-0 flex-1 overflow-auto">
        <table aria-label="Investigations" className="w-full min-w-[900px] table-fixed border-collapse text-left">
          <thead className="sticky top-0 z-sticky bg-muted/40 backdrop-blur">
            <tr className="h-8 border-b border-border text-[10px] tracking-[0.08em] text-muted-foreground uppercase">
              <th className={TH}>Investigation</th>
              <th className={`w-[180px] ${TH}`}>Agent</th>
              <th className={`w-[220px] ${TH}`}>Schedule</th>
              <th className={`w-[150px] ${TH}`}>Last run</th>
              <th className={`w-[64px] ${TH_NUM}`}>Open</th>
              <th className="w-[92px]" />
            </tr>
          </thead>
          <tbody>
            {shown.map((lens) => {
              const latest = lens.jobs[0];
              const failed = latest?.status === "failed";
              const open = lens.findings.filter((f) => f.status === "open").length;
              return (
                <tr
                  key={lens.id}
                  onClick={readOnly ? undefined : () => onEdit(lens.id)}
                  data-testid="investigation-row"
                  aria-label={lens.settings.name}
                  className={cn(
                    "h-9 border-b border-border/60 text-[12px] transition-colors duration-150 hover:bg-trace-row-hover motion-reduce:transition-none",
                    !readOnly && "cursor-pointer",
                  )}
                >
                  <td className="truncate px-3 text-foreground">{lens.settings.name}</td>
                  <td className="truncate px-3 text-muted-foreground">{scopeLabel(lens.settings)}</td>
                  <td className="px-3 font-mono text-[11px]" data-testid="investigation-schedule">
                    <span
                      className={cn(
                        "inline-flex items-center gap-1.5",
                        lens.settings.enabled ? "text-foreground" : "text-muted-foreground",
                      )}
                    >
                      <span
                        className={cn(
                          "size-1.5 rounded-full",
                          lens.settings.enabled ? "bg-[#3b5bfd]" : "bg-muted-foreground/40",
                        )}
                      />
                      {scheduleLabel(lens, now)}
                    </span>
                  </td>
                  <td
                    className="px-3 font-mono text-[11px]"
                    title={latest ? formatActivityTimestamp(latest.created_at) : undefined}
                  >
                    <span className={failed ? "text-destructive" : "text-muted-foreground"}>
                      {lensStatus(lens, connected).toLowerCase()}
                    </span>
                    {latest && (
                      <span className="text-muted-foreground"> · {agoLabel(Date.parse(latest.created_at), now)}</span>
                    )}
                  </td>
                  <td className="px-3 text-right font-mono tabular-nums text-foreground">{open}</td>
                  <td className="pr-2">
                    {!readOnly && (
                      <span className="flex items-center justify-end gap-0.5">
                        <button
                          type="button"
                          aria-label={`Run ${lens.settings.name} now`}
                          title="Run now"
                          onClick={(event) => {
                            event.stopPropagation();
                            onRunNow(lens.id);
                          }}
                          className="inline-flex size-6 items-center justify-center rounded text-muted-foreground hover:bg-muted hover:text-foreground"
                        >
                          <Play className="size-3" />
                        </button>
                        <span
                          aria-hidden="true"
                          className="inline-flex size-6 items-center justify-center text-muted-foreground/60"
                        >
                          <Pencil className="size-3" />
                        </span>
                      </span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
        {!shown.length && (
          <div className="py-16 text-center text-[12px] text-muted-foreground">
            No investigations match your search.
          </div>
        )}
      </div>
      <footer className="flex h-8 shrink-0 items-center border-t border-border bg-muted/40 px-3 font-mono text-[11px] text-muted-foreground">
        {shown.length} {shown.length === 1 ? "investigation" : "investigations"} ·{" "}
        {lenses.filter((l) => l.settings.enabled).length} watching
      </footer>
    </div>
  );
}
