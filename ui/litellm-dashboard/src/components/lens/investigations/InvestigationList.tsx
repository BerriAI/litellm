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
import { Inspector } from "@/components/shared/Inspector";
import { Input } from "@/components/ui/input";
import { formatActivityTimestamp } from "@/utils/activityTimestamp";
import { cn } from "@/lib/cva.config";
import { agoLabel } from "@/components/view_logs/TraceView/lensField";

import { findingAgents, findingKey, openFindings, scheduleLabel } from "../model/inbox";
import { lensStatus } from "../model/status";
import { scopeLabel } from "../model/format";
import { type Finding, type Lens } from "../model/types";
import { useListSearchRoute } from "../route";
import { FINDING_PANEL_WIDTH_KEY } from "./FindingDetails";

const PRIORITY_COLOR = { high: "text-destructive", medium: "text-amber-500", low: "text-muted-foreground" } as const;
const ROW =
  "cursor-pointer border-b border-border/60 transition-colors duration-150 hover:bg-trace-row-hover focus-visible:outline-2 focus-visible:outline-ring data-[state=selected]:bg-trace-row-selected data-[state=selected]:shadow-[inset_2px_0_0_var(--trace-brand)] data-[state=selected]:hover:bg-trace-row-selected motion-reduce:transition-none";
const META = "truncate text-xs text-muted-foreground";

/** One row of the list: an investigation, or an open finding shown under the investigation that owns it. */
export type InvestigationRow =
  | { readonly kind: "investigation"; readonly lens: Lens }
  | { readonly kind: "finding"; readonly lens: Lens; readonly finding: Finding };

export const investigationRowKey = (row: InvestigationRow): string =>
  row.kind === "investigation" ? `investigation:${row.lens.id}` : `finding:${findingKey(row.lens, row.finding)}`;

const ROW_LABEL = { investigation: "Investigation details", finding: "Finding details" } as const;

function JobIcon({ lens }: { lens: Lens }) {
  const status = lens.jobs[0]?.status;
  const className = "size-4 shrink-0";
  if (status === "queued" || status === "running")
    return <CircleDashed aria-hidden="true" className={cn(className, "text-info")} />;
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
      <span className={cn("truncate text-sm text-foreground", className)}>{title}</span>
    </span>
  );
}

export interface InvestigationListProps {
  readonly lenses: readonly Lens[];
  readonly connected: boolean;
  readonly readOnly?: boolean;
  readonly selected: InvestigationRow | null;
  readonly onSelect: (row: InvestigationRow | null) => void;
  readonly onEdit: (id: string) => void;
  readonly onRunNow: (id: string) => void;
  readonly actions?: ReactNode;
  /** Body of the side panel for the selected row. */
  readonly children: (row: InvestigationRow) => ReactNode;
}

/** Investigations with their open findings; any row opens in the side panel and J/K walk the visible rows. */
export function InvestigationList({
  lenses,
  connected,
  readOnly = false,
  selected,
  onSelect,
  onEdit,
  onRunNow,
  actions,
  children,
}: InvestigationListProps) {
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
  const sections = shown.map((lens) => {
    const findings = openFindings(lens);
    return { lens, findings, expanded: findings.length > 0 && !collapsed.has(lens.id) };
  });
  const rows: readonly InvestigationRow[] = sections.flatMap(({ lens, findings, expanded }) => [
    { kind: "investigation", lens },
    ...(expanded ? findings.map((finding) => ({ kind: "finding", lens, finding }) as const) : []),
  ]);
  const noun = selected?.kind ?? "investigation";
  return (
    <Inspector.Root
      items={rows}
      itemKey={investigationRowKey}
      selected={selected}
      onSelectedChange={onSelect}
      noun={noun}
      storageKey={FINDING_PANEL_WIDTH_KEY}
    >
      <div className="flex min-h-[420px] flex-1 flex-col overflow-hidden rounded-lg border border-border bg-card">
        <div className="flex min-h-11 shrink-0 flex-wrap items-center gap-2 border-b border-border bg-muted/40 px-3 py-2">
          <div className="relative min-w-40 flex-1 basis-60 max-w-[380px]">
            <Search className="pointer-events-none absolute top-1/2 left-2.5 size-3.5 -translate-y-1/2 text-muted-foreground" />
            <Input
              aria-label="Search investigations"
              placeholder="Search investigations"
              className="h-7 bg-background pl-8 text-xs"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
          </div>
          {actions && <div className="ml-auto flex items-center gap-3">{actions}</div>}
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
              {sections.map(({ lens, findings, expanded }) => {
                const latest = lens.jobs[0];
                return (
                  <Fragment key={lens.id}>
                    <Inspector.Row
                      item={{ kind: "investigation", lens }}
                      render={<tr tabIndex={0} aria-label={lens.settings.name} className={cn(ROW, "group h-14")} />}
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
                          "w-[180px] truncate px-3 text-right text-xs",
                          latest?.status === "failed" ? "text-destructive" : "text-muted-foreground",
                        )}
                      >
                        {lensStatus(lens, connected)}
                      </td>
                      <td className="w-[72px] px-3 text-right">
                        {findings.length > 0 && (
                          <span
                            title={`${findings.length} open ${findings.length === 1 ? "finding" : "findings"}`}
                            className="inline-flex min-w-5 justify-center rounded-full bg-muted px-1.5 font-mono text-xs tabular-nums text-foreground"
                          >
                            {findings.length}
                          </span>
                        )}
                      </td>
                      <td
                        className="w-[120px] px-3 text-right text-xs text-muted-foreground"
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
                            <button
                              type="button"
                              aria-label={`Edit ${lens.settings.name}`}
                              title="Edit"
                              onClick={(event) => {
                                event.stopPropagation();
                                onEdit(lens.id);
                              }}
                              className="inline-flex size-7 items-center justify-center rounded text-muted-foreground/60 hover:bg-muted hover:text-foreground group-hover:text-muted-foreground"
                            >
                              <Pencil className="size-3.5" />
                            </button>
                          </span>
                        )}
                      </td>
                    </Inspector.Row>
                    {expanded &&
                      findings.map((finding, index) => {
                        const priority = finding.priority ?? "medium";
                        const last = index === findings.length - 1;
                        const runs = finding.occurrences.length;
                        return (
                          <Inspector.Row
                            key={finding.id}
                            item={{ kind: "finding", lens, finding }}
                            render={<tr tabIndex={0} aria-label={finding.title} className={cn(ROW, "h-12")} />}
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
                            <td className="px-3 text-right text-xs text-muted-foreground">
                              {runs} {runs === 1 ? "run" : "runs"}
                            </td>
                            <td />
                            <td
                              className="px-3 text-right text-xs text-muted-foreground"
                              title={formatActivityTimestamp(finding.last_seen)}
                            >
                              {agoLabel(Date.parse(finding.last_seen), now)}
                            </td>
                            <td className="pr-5">
                              <ChevronRight aria-hidden="true" className="ml-auto size-3.5 text-muted-foreground/60" />
                            </td>
                          </Inspector.Row>
                        );
                      })}
                  </Fragment>
                );
              })}
            </tbody>
          </table>
          {!shown.length && (
            <div className="py-16 text-center text-xs text-muted-foreground">No investigations match your search.</div>
          )}
        </div>
        <footer className="flex h-8 shrink-0 items-center border-t border-border bg-muted/40 px-3 font-mono text-xs text-muted-foreground">
          {shown.length} {shown.length === 1 ? "investigation" : "investigations"} ·{" "}
          {lenses.filter((l) => l.settings.enabled).length} watching
        </footer>
      </div>
      <Inspector.Panel label={ROW_LABEL[noun]} testId="investigation-panel">
        {children}
      </Inspector.Panel>
    </Inspector.Root>
  );
}
