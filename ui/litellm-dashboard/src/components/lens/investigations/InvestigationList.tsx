"use client";

import { Fragment, useState, type ReactNode } from "react";
import {
  ChevronRight,
  Circle,
  CircleCheck,
  CircleDashed,
  CircleDot,
  CircleSlash,
  CircleX,
  Pencil,
  Play,
  Search,
} from "lucide-react";

import { useNow } from "@/hooks/useNow";
import { Input } from "@/components/ui/input";
import { formatActivityTimestamp } from "@/utils/activityTimestamp";
import { cn } from "@/lib/cva.config";
import { agoLabel } from "@/components/view_logs/TraceView/lensField";

import { findingAgents, openFindings, scheduleLabel } from "../model/inbox";
import { lensStatus } from "../model/status";
import { scopeLabel } from "../model/format";
import { type Finding, type Lens } from "../model/types";
import { useListSearchRoute } from "../route";

const PRIORITY_COLOR = { high: "text-[#e5484d]", medium: "text-amber-500", low: "text-muted-foreground" } as const;
const ROW =
  "border-b border-border/60 transition-colors duration-150 hover:bg-trace-row-hover motion-reduce:transition-none";
const META = "truncate text-[11px] text-muted-foreground";

function JobIcon({ lens }: { lens: Lens }) {
  const status = lens.jobs[0]?.status;
  const className = "size-4 shrink-0";
  if (status === "queued" || status === "running")
    return <CircleDashed aria-hidden="true" className={cn(className, "text-[#3b5bfd]")} />;
  if (status === "failed") return <CircleX aria-hidden="true" className={cn(className, "text-destructive")} />;
  if (status === "cancelled")
    return <CircleSlash aria-hidden="true" className={cn(className, "text-muted-foreground")} />;
  if (status === "completed") return <CircleCheck aria-hidden="true" className={cn(className, "text-emerald-600")} />;
  return <Circle aria-hidden="true" className={cn(className, "text-muted-foreground")} />;
}

function TwoLine({ meta, title, className }: { meta: ReactNode; title: ReactNode; className?: string }) {
  return (
    <span className="flex min-w-0 flex-col gap-0.5">
      <span className={META}>{meta}</span>
      <span className={cn("truncate text-[13px] text-foreground", className)}>{title}</span>
    </span>
  );
}

export function InvestigationList({
  lenses,
  connected,
  readOnly = false,
  onEdit,
  onRunNow,
  onOpenFinding,
}: {
  lenses: Lens[];
  connected: boolean;
  readOnly?: boolean;
  onEdit: (id: string) => void;
  onRunNow: (id: string) => void;
  onOpenFinding: (lens: Lens, finding: Finding) => void;
}) {
  const [search, setSearch] = useListSearchRoute();
  const now = useNow(15000);
  const [collapsed, setCollapsed] = useState<ReadonlySet<string>>(new Set());
  const toggle = (id: string) =>
    setCollapsed((current) => {
      const next = new Set(current);
      if (!next.delete(id)) next.add(id);
      return next;
    });
  const shown = lenses.filter((lens) =>
    `${lens.settings.name} ${scopeLabel(lens.settings)}`.toLowerCase().includes(search.toLowerCase()),
  );
  return (
    <div className="flex min-h-[420px] flex-1 flex-col overflow-hidden rounded-lg border border-border bg-card">
      <div className="flex min-h-11 shrink-0 flex-wrap items-center gap-2 border-b border-border bg-muted/40 px-3 py-2">
        <div className="relative w-full max-w-[380px]">
          <Search className="pointer-events-none absolute top-1/2 left-2.5 size-3.5 -translate-y-1/2 text-muted-foreground" />
          <Input
            aria-label="Search investigations"
            placeholder="Search investigations"
            className="h-7 bg-background pl-8 text-[12px]"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>
      </div>
      <div className="min-h-0 flex-1 overflow-auto">
        <table aria-label="Investigations" className="w-full min-w-[720px] table-fixed border-collapse text-left">
          <thead className="sr-only">
            <tr>
              <th>Investigation</th>
              <th>Status</th>
              <th>Open findings</th>
              <th>Last activity</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {shown.map((lens) => {
              const latest = lens.jobs[0];
              const findings = openFindings(lens);
              const expanded = findings.length > 0 && !collapsed.has(lens.id);
              return (
                <Fragment key={lens.id}>
                  <tr
                    onClick={readOnly ? undefined : () => onEdit(lens.id)}
                    aria-label={lens.settings.name}
                    className={cn(ROW, "group h-14", !readOnly && "cursor-pointer")}
                  >
                    <td className="pl-2">
                      <span className="flex min-w-0 items-center gap-2">
                        {findings.length > 0 ? (
                          <button
                            type="button"
                            aria-expanded={expanded}
                            aria-label={`${expanded ? "Hide" : "Show"} findings for ${lens.settings.name}`}
                            onClick={(event) => {
                              event.stopPropagation();
                              toggle(lens.id);
                            }}
                            className="inline-flex size-6 shrink-0 items-center justify-center rounded text-muted-foreground hover:bg-muted hover:text-foreground"
                          >
                            <ChevronRight
                              className={cn(
                                "size-3.5 transition-transform duration-150 motion-reduce:transition-none",
                                expanded && "rotate-90",
                              )}
                            />
                          </button>
                        ) : (
                          <span aria-hidden="true" className="size-6 shrink-0" />
                        )}
                        <JobIcon lens={lens} />
                        <TwoLine
                          meta={`${scopeLabel(lens.settings)} · ${scheduleLabel(lens, now)}`}
                          title={lens.settings.name}
                          className="font-semibold"
                        />
                      </span>
                    </td>
                    <td
                      className={cn(
                        "w-[180px] truncate px-3 text-right text-[12px]",
                        latest?.status === "failed" ? "text-destructive" : "text-muted-foreground",
                      )}
                    >
                      {lensStatus(lens, connected)}
                    </td>
                    <td className="w-[72px] px-3 text-right">
                      {findings.length > 0 && (
                        <span
                          title={`${findings.length} open ${findings.length === 1 ? "finding" : "findings"}`}
                          className="inline-flex min-w-5 justify-center rounded-full bg-muted px-1.5 font-mono text-[11px] tabular-nums text-foreground"
                        >
                          {findings.length}
                        </span>
                      )}
                    </td>
                    <td
                      className="w-[120px] px-3 text-right text-[12px] text-muted-foreground"
                      title={latest ? formatActivityTimestamp(latest.created_at) : undefined}
                    >
                      {latest ? agoLabel(Date.parse(latest.created_at), now) : "never run"}
                    </td>
                    <td className="w-[76px] pr-3">
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
                            className="inline-flex size-7 items-center justify-center rounded text-muted-foreground hover:bg-muted hover:text-foreground"
                          >
                            <Play className="size-3.5" />
                          </button>
                          <span
                            aria-hidden="true"
                            className="inline-flex size-7 items-center justify-center text-muted-foreground/60 group-hover:text-muted-foreground"
                          >
                            <Pencil className="size-3.5" />
                          </span>
                        </span>
                      )}
                    </td>
                  </tr>
                  {expanded &&
                    findings.map((finding, index) => {
                      const priority = finding.priority ?? "medium";
                      const last = index === findings.length - 1;
                      const runs = finding.occurrences.length;
                      return (
                        <tr
                          key={finding.id}
                          onClick={() => onOpenFinding(lens, finding)}
                          aria-label={finding.title}
                          className={cn(ROW, "h-12 cursor-pointer")}
                        >
                          <td className="pl-2" title={finding.suggestion ? `Fix: ${finding.suggestion}` : undefined}>
                            <span className="flex min-w-0 items-center gap-2">
                              <span aria-hidden="true" className="relative h-12 w-6 shrink-0">
                                <span
                                  className={cn("absolute left-3 top-0 w-px bg-border", last ? "h-1/2" : "h-full")}
                                />
                                <span className="absolute top-1/2 left-3 h-px w-3 bg-border" />
                              </span>
                              <span aria-hidden="true" className="w-4 shrink-0" />
                              <CircleDot
                                aria-hidden="true"
                                className={cn("size-4 shrink-0", PRIORITY_COLOR[priority])}
                              />
                              <TwoLine
                                meta={`${priority} priority · ${findingAgents(lens, finding).join(", ")}`}
                                title={finding.title}
                              />
                            </span>
                          </td>
                          <td className="px-3 text-right text-[12px] text-muted-foreground">
                            {runs} {runs === 1 ? "run" : "runs"}
                          </td>
                          <td />
                          <td
                            className="px-3 text-right text-[12px] text-muted-foreground"
                            title={formatActivityTimestamp(finding.last_seen)}
                          >
                            {agoLabel(Date.parse(finding.last_seen), now)}
                          </td>
                          <td className="pr-5">
                            <ChevronRight aria-hidden="true" className="ml-auto size-3.5 text-muted-foreground/60" />
                          </td>
                        </tr>
                      );
                    })}
                </Fragment>
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
