"use client";

import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { ALL_AGENTS, inboxAgents, inboxFinding, sampledExecutions, type InboxRow, type Priority } from "../model/inbox";
import type { Finding } from "../model/types";
import { useEvidenceRoute, useInboxFilters } from "../route";
import { EvidenceView } from "./Evidence";
import { FindingDetails } from "./FindingDetails";

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

export function InboxFilters({ rows }: { rows: readonly InboxRow[] }) {
  const filters = useInboxFilters();
  const agents = [
    { value: ALL_AGENTS, label: "All agents" },
    ...inboxAgents(rows).map((agent) => ({ value: agent, label: agent })),
  ];
  return (
    <div className="flex min-h-10 shrink-0 flex-wrap items-center gap-2 border-b px-3 py-1.5">
      <FilterSelect label="Filter by agent" value={filters.agent} items={agents} onChange={filters.setAgent} />
      <FilterSelect
        label="Filter by priority"
        value={filters.priority}
        items={PRIORITIES}
        onChange={filters.setPriority}
      />
    </div>
  );
}

export function InboxDetail({
  row,
  readOnly,
  busy,
  reason,
  onReview,
}: {
  row: InboxRow;
  readOnly: boolean;
  busy: boolean;
  reason?: string;
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
          finding={{ ...inboxFinding(row), ...(reason !== undefined ? { reason } : {}) }}
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
