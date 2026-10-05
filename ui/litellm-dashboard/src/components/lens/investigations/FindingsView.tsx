"use client";

import { useQuery } from "@tanstack/react-query";
import { ChevronRight } from "lucide-react";
import { Inspector } from "@/components/shared/Inspector";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useNow } from "@/hooks/useNow";
import { cn } from "@/lib/cva.config";
import { formatActivityTimestamp } from "@/utils/activityTimestamp";
import { useLensApi } from "../data/LensServices";
import { useLensUpdate } from "../data/mutations";
import { lensQueries } from "../data/queries";
import { agoLabel } from "../model/format";
import {
  ALL_AGENTS,
  filterInbox,
  findingKey,
  inboxAgents,
  inboxFinding,
  inboxRows,
  sampledExecutions,
  type InboxRow,
  type Priority,
} from "../model/inbox";
import type { Finding } from "../model/types";
import { useEvidenceRoute, useInboxFilters, useIssueRoute } from "../route";
import { FINDING_PANEL_WIDTH_KEY } from "../storage";
import { EvidenceView } from "./Evidence";
import { FindingDetails } from "./FindingDetails";
import { InvestigationError, InvestigationsLoading } from "./InvestigationStates";

const PRIORITY_DOT = { high: "bg-destructive", medium: "bg-warning", low: "bg-muted-foreground/50" } as const;
const PRIORITIES: { value: Priority | "all"; label: string }[] = [
  { value: "all", label: "All priorities" },
  { value: "high", label: "High" },
  { value: "medium", label: "Medium" },
  { value: "low", label: "Low" },
];

function FilterSelect<T extends string>({
  label,
  value,
  items,
  onChange,
}: {
  label: string;
  value: T;
  items: { value: T; label: string }[];
  onChange: (value: T) => void;
}) {
  return (
    <Select items={items} value={value} onValueChange={(next: T | null) => next !== null && onChange(next)}>
      <SelectTrigger size="sm" className="h-7 min-w-32 text-xs" aria-label={label}>
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        {items.map((item) => (
          <SelectItem key={item.value} value={item.value}>
            {item.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}

function InboxDetail({
  row,
  readOnly,
  busy,
  onReview,
}: {
  row: InboxRow;
  readOnly: boolean;
  busy: boolean;
  onReview: (row: InboxRow, status: Finding["status"], reason: string) => void;
}) {
  const { evidence, setEvidence } = useEvidenceRoute();
  const owner =
    row.sources.find(({ finding }) => finding.evidence.some((quote) => quote.execution_id === evidence?.id)) ??
    row.sources[0];
  return (
    <>
      <div hidden={evidence !== null} className={evidence ? undefined : "flex min-h-0 flex-1 flex-col"}>
        <FindingDetails
          key={row.key}
          finding={inboxFinding(row)}
          agents={row.agents}
          sampledRuns={row.sources.flatMap(({ lens }) => sampledExecutions(lens))}
          readOnly={readOnly}
          busy={busy}
          onOpenEvidence={setEvidence}
          onReview={(status, reason) => onReview(row, status, reason)}
        />
      </div>
      {evidence && (
        <EvidenceView
          lensId={owner.lens.id}
          evidence={evidence}
          backLabel="Back to finding"
          onBack={() => setEvidence(null)}
        />
      )}
    </>
  );
}

export function FindingsView({ readOnly = false }: { readOnly?: boolean }) {
  const api = useLensApi();
  const list = useQuery(lensQueries.list(api));
  const update = useLensUpdate();
  const { issueKey, setIssueKey } = useIssueRoute();
  const filters = useInboxFilters();
  const now = useNow(30000);
  const all = inboxRows(list.data?.lenses ?? []);
  const rows = filterInbox(all, filters);
  const selected =
    all.find((row) => row.sources.some(({ lens, finding }) => findingKey(lens, finding) === issueKey)) ?? null;
  const agents = [
    { value: ALL_AGENTS, label: "All agents" },
    ...inboxAgents(all).map((agent) => ({ value: agent, label: agent })),
  ];
  const review = (row: InboxRow, status: Finding["status"], reason: string) => {
    update.mutate(
      async (api) => {
        const results = await Promise.allSettled(
          row.sources.map(({ lens, finding }) => api.reviewFinding(lens.id, finding.id, status, reason)),
        );
        const failed = results.find((result) => result.status === "rejected");
        if (failed) throw failed.reason;
      },
      { onSuccess: () => setIssueKey(null) },
    );
  };
  if (list.isPending) return <InvestigationsLoading />;
  return (
    <Inspector.Root
      items={rows}
      itemKey={(row) => row.key}
      selected={selected}
      onSelectedChange={(row) => setIssueKey(row?.key ?? null)}
      noun="finding"
      storageKey={FINDING_PANEL_WIDTH_KEY}
    >
      <section aria-label="Findings" className="flex min-h-0 flex-1 flex-col">
        {(list.error || update.error) && (
          <InvestigationError
            message={(list.error ?? update.error)!.message}
            refresh={() => {
              update.reset();
              void list.refetch();
            }}
          />
        )}
        <div className="flex min-h-10 shrink-0 flex-wrap items-center gap-2 border-b px-3 py-1.5">
          <FilterSelect label="Filter by agent" value={filters.agent} items={agents} onChange={filters.setAgent} />
          <FilterSelect
            label="Filter by priority"
            value={filters.priority}
            items={PRIORITIES}
            onChange={filters.setPriority}
          />
        </div>
        <div className="min-h-0 flex-1 overflow-auto">
          <table aria-label="Findings" className="w-full table-fixed border-collapse text-left text-xs">
            <thead className="sticky top-0 z-sticky bg-muted/40 backdrop-blur">
              <tr className="h-8 border-b text-xs tracking-wider text-muted-foreground uppercase">
                <th className="w-20 px-3 font-medium">Priority</th>
                <th className="px-3 font-medium">Finding</th>
                <th className="hidden w-40 px-3 font-medium md:table-cell">Agent</th>
                <th className="hidden w-16 px-3 text-right font-medium sm:table-cell">Runs</th>
                <th className="hidden w-24 px-3 font-medium lg:table-cell">Last seen</th>
                <th className="w-7">
                  <span className="sr-only">Details</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <Inspector.Row
                  key={row.key}
                  item={row}
                  render={
                    <tr
                      tabIndex={0}
                      aria-label={row.title}
                      className="h-9 cursor-pointer border-b border-border/60 hover:bg-trace-row-hover focus-visible:outline-2 focus-visible:outline-ring data-[state=selected]:bg-trace-row-selected"
                    />
                  }
                >
                  <td className="px-3">
                    <span className="inline-flex items-center gap-1.5 text-muted-foreground">
                      <span aria-hidden="true" className={cn("size-1.5 rounded-full", PRIORITY_DOT[row.priority])} />
                      {row.priority}
                    </span>
                  </td>
                  <td className="px-3 py-2 sm:py-0" title={row.suggestion || undefined}>
                    <span className="line-clamp-2 text-foreground sm:block sm:truncate">{row.title}</span>
                    <span className="mt-1 block text-xs text-muted-foreground md:hidden">
                      {row.agents.join(", ")} · {row.runs} {row.runs === 1 ? "run" : "runs"}
                    </span>
                  </td>
                  <td
                    className="hidden truncate px-3 text-muted-foreground md:table-cell"
                    title={row.agents.join(", ")}
                  >
                    {row.agents.join(", ")}
                  </td>
                  <td className="hidden px-3 text-right font-mono tabular-nums sm:table-cell">{row.runs}</td>
                  <td
                    className="hidden px-3 tabular-nums text-muted-foreground lg:table-cell"
                    title={formatActivityTimestamp(row.lastSeen)}
                  >
                    {agoLabel(Date.parse(row.lastSeen), now)}
                  </td>
                  <td>
                    <ChevronRight aria-hidden="true" className="size-3 text-muted-foreground/60" />
                  </td>
                </Inspector.Row>
              ))}
            </tbody>
          </table>
          {!rows.length && !list.error && (
            <p className="px-4 py-16 text-center text-xs text-muted-foreground">
              {all.length
                ? "No findings match these filters."
                : "No open findings yet. New problems show up here as soon as an investigation spots them."}
            </p>
          )}
        </div>
        <footer className="flex h-8 shrink-0 items-center border-t bg-muted/30 px-3 text-xs text-muted-foreground">
          {rows.length} {rows.length === 1 ? "finding" : "findings"}
          {rows.length !== all.length && ` of ${all.length}`}
        </footer>
      </section>
      <Inspector.Panel label="Finding details" testId="finding-panel">
        {(row: InboxRow) => <InboxDetail row={row} readOnly={readOnly} busy={update.isPending} onReview={review} />}
      </Inspector.Panel>
    </Inspector.Root>
  );
}
