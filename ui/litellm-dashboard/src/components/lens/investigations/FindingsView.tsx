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
  inboxSummary,
  type InboxRow,
  type Priority,
} from "../model/inbox";
import type { Finding } from "../model/types";
import { useEvidenceRoute, useInboxFilters, useIssueRoute } from "../route";
import { FINDING_PANEL_WIDTH_KEY } from "../storage";
import { EvidenceView } from "./Evidence";
import { FindingDetails } from "./FindingDetails";
import { InvestigationError, InvestigationsLoading } from "./InvestigationStates";
import { PRIORITY_LABEL, PRIORITY_ORDER, PriorityDot } from "./PriorityMark";
import { Panel } from "../ui/Panel";
import { StatCell, StatStrip } from "../ui/StatStrip";

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
      <SelectTrigger
        size="sm"
        className="h-7 min-w-0 flex-1 border-transparent bg-muted/60 text-xs shadow-none hover:bg-muted"
        aria-label={label}
      >
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
          className="mx-2 block cursor-pointer space-y-1.5 rounded-md px-2 py-2.5 transition-[background-color] duration-150 outline-none hover:bg-muted/60 focus-visible:ring-2 focus-visible:ring-ring/50 data-[state=selected]:bg-muted"
        />
      }
    >
      <div role="gridcell" className="line-clamp-2 text-sm leading-snug text-pretty text-foreground">
        {row.title}
      </div>
      <div className="flex items-center gap-3 text-xs text-muted-foreground">
        <span className="shrink-0 tabular-nums" title={formatActivityTimestamp(row.lastSeen)}>
          {agoLabel(Date.parse(row.lastSeen), now)}
        </span>
        <span aria-hidden="true" className="h-1 min-w-0 flex-1 overflow-hidden rounded-full bg-muted">
          {frequency.total > 0 && (
            <span
              className="block h-full rounded-full bg-[#2b3fd6]"
              style={{ width: `${Math.max(2, (frequency.affected / frequency.total) * 100)}%` }}
            />
          )}
        </span>
        <span
          className="shrink-0 whitespace-nowrap text-foreground/75 tabular-nums"
          title={`${row.runs} affected ${row.runs === 1 ? "trace" : "traces"} across ${row.sources.map(({ lens }) => lens.settings.name).join(", ")}`}
        >
          {percent ? `${percent} affected` : `${row.runs} ${row.runs === 1 ? "trace" : "traces"}`}
        </span>
      </div>
    </Inspector.Row>
  );
}

function FindingList({ rows, now }: { rows: readonly InboxRow[]; now: number }) {
  const groups = PRIORITY_ORDER.map((priority) => ({
    priority,
    rows: rows.filter((row) => row.priority === priority),
  })).filter((group) => group.rows.length > 0);
  return (
    <div role="grid" aria-label="Findings" className="min-h-0 flex-1 overflow-y-auto overscroll-contain pb-2">
      {groups.map((group) => (
        <div role="rowgroup" key={group.priority} aria-label={`${PRIORITY_LABEL[group.priority]} priority findings`}>
          <div
            role="row"
            className="sticky top-0 z-raised flex items-center gap-2 bg-card/95 px-4 pt-3 pb-1.5 text-xs font-medium text-muted-foreground backdrop-blur"
          >
            <PriorityDot priority={group.priority} />
            <span role="columnheader">{PRIORITY_LABEL[group.priority]} priority</span>
            <span className="ml-auto tabular-nums">{group.rows.length}</span>
          </div>
          <div className="space-y-0.5">
            {group.rows.map((row) => (
              <FindingRow key={row.key} row={row} now={now} />
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

function FindingStats({ rows, now }: { rows: readonly InboxRow[]; now: number }) {
  const summary = inboxSummary(rows, now);
  return (
    <StatStrip>
      <StatCell
        label="Open findings"
        value={summary.open.toLocaleString()}
        hint={`${summary.newThisWeek.toLocaleString()} seen this week`}
      />
      <StatCell
        label="High priority"
        value={
          <span className={summary.high > 0 ? "text-destructive" : undefined}>{summary.high.toLocaleString()}</span>
        }
        hint={
          summary.open > 0 ? `${Math.round((summary.high / summary.open) * 100)}% of open findings` : "Nothing open"
        }
      />
      <StatCell
        label="Traces affected"
        value={summary.affectedTraces.toLocaleString()}
        hint="Distinct traces behind open findings"
      />
      <StatCell
        label="Agents"
        value={summary.agents.toLocaleString()}
        hint={summary.agents === 1 ? "Has open findings" : "Have open findings"}
      />
    </StatStrip>
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
      <div className="flex min-h-0 flex-1 flex-col gap-3">
        <FindingStats rows={all} now={now} />
        <Panel className="flex-1">
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
              <div className="flex shrink-0 items-center gap-2 border-b px-3 py-2">
                <FilterSelect
                  label="Filter by agent"
                  value={filters.agent}
                  items={agents}
                  onChange={filters.setAgent}
                />
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
              <footer className="flex h-9 shrink-0 items-center border-t px-3 text-xs text-muted-foreground">
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
        </Panel>
      </div>
    </Inspector.Root>
  );
}
