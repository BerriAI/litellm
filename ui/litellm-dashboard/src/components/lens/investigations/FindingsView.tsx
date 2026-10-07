"use client";

import { useQuery } from "@tanstack/react-query";
import { Inspector, useInspector } from "@/components/shared/Inspector";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useNow } from "@/hooks/useNow";
import { formatActivityTimestamp } from "@/utils/activityTimestamp";
import { useLensApi } from "../data/LensServices";
import { useLensUpdate } from "../data/mutations";
import { lensQueries } from "../data/queries";
import { agoLabel } from "../model/format";
import { findingFrequency, percentLabel } from "../model/frequency";
import {
  ALL_AGENTS,
  filterInbox,
  findingKey,
  inboxAgents,
  inboxFinding,
  inboxRows,
  inboxSampledRuns,
  type InboxRow,
  type Priority,
} from "../model/inbox";
import type { Finding } from "../model/types";
import { useEvidenceRoute, useInboxFilters, useIssueRoute } from "../route";
import { FINDING_PANEL_WIDTH_KEY } from "../storage";
import { EvidenceView } from "./Evidence";
import { FindingDetails } from "./FindingDetails";
import { InvestigationError, InvestigationsLoading } from "./InvestigationStates";

const PRIORITIES: { value: Priority | "all"; label: string }[] = [
  { value: "all", label: "All priorities" },
  { value: "high", label: "High" },
  { value: "medium", label: "Medium" },
  { value: "low", label: "Low" },
];

const DAY_MS = 86_400_000;

function dayGroup(iso: string, now: number): string {
  const startOfToday = new Date(now).setHours(0, 0, 0, 0);
  const seen = Date.parse(iso);
  if (seen >= startOfToday) return "Today";
  if (seen >= startOfToday - DAY_MS) return "Yesterday";
  if (seen >= startOfToday - 7 * DAY_MS) return "This week";
  return "Earlier";
}

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
      <SelectTrigger size="sm" className="h-7 min-w-0 flex-1 text-xs" aria-label={label}>
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

function FindingRow({ row, now }: { row: InboxRow; now: number }) {
  const frequency = findingFrequency(inboxFinding(row).occurrences, inboxSampledRuns(row));
  const percent = percentLabel(frequency.affected, frequency.total);
  return (
    <Inspector.Row
      item={row}
      render={
        <div
          role="row"
          tabIndex={0}
          aria-label={row.title}
          className="mx-2 block cursor-pointer space-y-1 rounded-md px-2 py-2 transition-colors outline-none hover:bg-muted/60 focus-visible:bg-muted/60 data-[state=selected]:bg-muted"
        />
      }
    >
      <div role="gridcell" className="line-clamp-2 text-xs text-foreground">
        {row.title}
      </div>
      <div className="flex flex-wrap items-center justify-between gap-x-2 gap-y-1 text-xs text-muted-foreground">
        <span title={formatActivityTimestamp(row.lastSeen)}>{agoLabel(Date.parse(row.lastSeen), now)}</span>
        <span
          className="whitespace-nowrap text-foreground/70 tabular-nums"
          title={`${row.runs} affected ${row.runs === 1 ? "trace" : "traces"} across ${row.sources.map(({ lens }) => lens.settings.name).join(", ")}`}
        >
          {percent ? `${percent} affected` : `${row.runs} ${row.runs === 1 ? "trace" : "traces"}`}
        </span>
      </div>
    </Inspector.Row>
  );
}

function FindingList({ rows, now }: { rows: readonly InboxRow[]; now: number }) {
  const groups = [...new Set(rows.map((row) => dayGroup(row.lastSeen, now)))];
  return (
    <div role="grid" aria-label="Findings" className="min-h-0 flex-1 overflow-y-auto pb-2">
      {groups.map((group) => (
        <div role="rowgroup" key={group} aria-label={`${group} findings`}>
          <div role="row" className="sticky top-0 z-raised bg-background px-4 pt-2 pb-1 text-xs text-muted-foreground">
            <span role="columnheader">{group}</span>
          </div>
          <div className="space-y-1 pt-1">
            {rows
              .filter((row) => dayGroup(row.lastSeen, now) === group)
              .map((row) => (
                <FindingRow key={row.key} row={row} now={now} />
              ))}
          </div>
        </div>
      ))}
    </div>
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
  const { close } = useInspector<InboxRow>();
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
          sampledRuns={inboxSampledRuns(row)}
          readOnly={readOnly}
          busy={busy}
          onOpenEvidence={setEvidence}
          onReview={(status, reason) => onReview(row, status, reason)}
          onClose={close}
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
      <div className="flex min-h-0 flex-1 flex-col overflow-hidden rounded-lg border bg-background">
        {(list.error || update.error) && (
          <InvestigationError
            message={(list.error ?? update.error)!.message}
            refresh={() => {
              update.reset();
              void list.refetch();
            }}
          />
        )}
        <div className="flex min-h-0 flex-1">
          <section
            aria-label="Findings list"
            className={`flex min-h-0 w-full flex-col border-r md:w-[22rem] md:shrink-0 lg:w-[28rem] ${selected ? "hidden md:flex" : "flex"}`}
          >
            <div className="flex shrink-0 items-center gap-2 px-3 py-2">
              <FilterSelect label="Filter by agent" value={filters.agent} items={agents} onChange={filters.setAgent} />
              <FilterSelect
                label="Filter by priority"
                value={filters.priority}
                items={PRIORITIES}
                onChange={filters.setPriority}
              />
            </div>
            <FindingList rows={rows} now={now} />
            {!rows.length && !list.error && (
              <p className="px-4 py-16 text-center text-xs text-muted-foreground">
                {all.length
                  ? "No findings match these filters."
                  : "No open findings yet. New problems show up here as soon as an investigation spots them."}
              </p>
            )}
            <footer className="flex h-8 shrink-0 items-center border-t px-3 text-xs text-muted-foreground">
              {rows.length} {rows.length === 1 ? "finding" : "findings"}
              {rows.length !== all.length && ` of ${all.length}`}
            </footer>
          </section>
          {selected ? (
            <aside
              aria-label="Finding details"
              data-testid="finding-panel"
              className="flex min-h-0 min-w-0 flex-1 flex-col"
            >
              <InboxDetail row={selected} readOnly={readOnly} busy={update.isPending} onReview={review} />
            </aside>
          ) : (
            <div className="hidden min-w-0 flex-1 items-center justify-center p-8 text-sm text-muted-foreground md:flex">
              {rows.length ? "Select a finding to see how often it happens and where." : null}
            </div>
          )}
        </div>
      </div>
    </Inspector.Root>
  );
}
