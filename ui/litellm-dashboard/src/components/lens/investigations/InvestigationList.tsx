"use client";

import {
  getCoreRowModel,
  getExpandedRowModel,
  useReactTable,
  type CellContext,
  type ColumnDef,
  type TableOptions,
} from "@tanstack/react-table";
import { createContext, useContext, type ReactNode } from "react";
import { ChevronRight, CircleDot, Pencil, Play } from "lucide-react";

import { useMediaQuery } from "usehooks-ts";
import { groupBy } from "es-toolkit";
import { useNow } from "@/hooks/useNow";
import { Inspector } from "@/components/shared/Inspector";
import { InspectorTable } from "@/components/shared/InspectorTable";
import { formatActivityTimestamp } from "@/utils/activityTimestamp";
import { cn } from "@/lib/cva.config";
import { agoLabel, scopeLabel } from "../model/format";

import { filterInbox, findingKey, inboxFinding, openFindings, scheduleLabel, type InboxRow } from "../model/inbox";
import { lensStatus } from "../model/status";
import { SearchBox } from "@/components/shared/search/SearchBox";
import { itemValues } from "@/components/shared/search/valueSource";
import { type Finding, type Lens } from "../model/types";
import { useInboxFilters, useLensRoute, useListSearchRoute } from "../route";
import { FINDING_PANEL_WIDTH_KEY } from "../storage";
import { filterInvestigations, INVESTIGATION_INDEX, INVESTIGATION_QUERY } from "./investigationQuery";
import { InboxFilters } from "./FindingInbox";

const ACTION =
  "inline-flex size-7 items-center justify-center rounded text-muted-foreground hover:bg-muted hover:text-foreground disabled:cursor-not-allowed disabled:opacity-50";
const INVESTIGATION_HEIGHT = 36;

/** One row of the list: an investigation, or an open finding shown under the investigation that owns it. */
export type InvestigationRow =
  | { readonly kind: "investigation"; readonly lens: Lens }
  | { readonly kind: "finding"; readonly lens: Lens; readonly finding: Finding; readonly inbox?: InboxRow };

export const findingRow = (inbox: InboxRow): InvestigationRow => ({
  kind: "finding",
  lens: inbox.sources[0].lens,
  finding: inboxFinding(inbox),
  inbox,
});

export const investigationRowKey = (row: InvestigationRow): string =>
  row.kind === "investigation"
    ? `investigation:${row.lens.id}`
    : `finding:${row.inbox?.key ?? findingKey(row.lens, row.finding)}`;

interface ListContextValue {
  readonly now: number;
  readonly connected: boolean;
  readonly readOnly: boolean;
  readonly demo: boolean;
  readonly onEdit: (id: string) => void;
  readonly onRunNow: (id: string) => void;
}

const ListContext = createContext<ListContextValue | null>(null);

function useList(): ListContextValue {
  const value = useContext(ListContext);
  if (value === null) throw new Error("Investigation cells must be rendered inside InvestigationList");
  return value;
}

type Cell = CellContext<InvestigationRow, unknown>;

const ROW_LABEL = { investigation: "Investigation details", finding: "Finding details" } as const;

function NameCell({ row }: Cell) {
  const item = row.original;
  if (item.kind === "finding")
    return (
      <span className="flex min-w-0 items-center gap-2" title={item.finding.suggestion || undefined}>
        <InspectorTable.Indent row={row} className="h-12" />
        <CircleDot
          aria-hidden="true"
          className={cn(
            "size-3.5 shrink-0",
            item.finding.priority === "high" ? "text-destructive" : "text-muted-foreground",
          )}
        />
        <span className="min-w-0">
          <span className="block truncate text-foreground">{item.finding.title}</span>
          <span className="text-xs text-muted-foreground">
            {item.finding.priority ?? "medium"} priority · {item.finding.occurrences.length}{" "}
            {item.finding.occurrences.length === 1 ? "run" : "runs"}
          </span>
          <span
            className="block truncate text-xs text-muted-foreground"
            title={item.inbox?.sources.map(({ lens }) => lens.settings.name).join(", ")}
          >
            {item.inbox?.sources.map(({ lens }) => lens.settings.name).join(", ")}
          </span>
        </span>
      </span>
    );
  return (
    <span className="flex min-w-0 items-center gap-2">
      <InspectorTable.Indent
        row={row}
        toggleLabel={(expanded) => `${expanded ? "Hide" : "Show"} findings for ${item.lens.settings.name}`}
      />
      <span className="flex min-w-0 flex-col">
        <span className="truncate font-medium text-foreground">{item.lens.settings.name}</span>
        <span className="truncate text-xs text-muted-foreground md:hidden">{scopeLabel(item.lens.settings)}</span>
      </span>
    </span>
  );
}

function AgentCell({ row: { original: item } }: Cell) {
  if (item.kind === "finding")
    return <span className="block truncate text-muted-foreground">{item.inbox?.agents.join(", ")}</span>;
  return <span className="block truncate text-muted-foreground">{scopeLabel(item.lens.settings)}</span>;
}

function ScheduleCell({ row: { original: item } }: Cell) {
  const { now } = useList();
  if (item.kind === "finding") {
    const names = [...new Set(item.inbox?.sources.map(({ lens }) => lens.settings.name))].join(", ");
    return (
      <span className="block truncate text-muted-foreground" title={names}>
        {names}
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1.5 whitespace-nowrap text-muted-foreground">
      <span
        aria-hidden="true"
        className={cn("size-1.5 rounded-full", item.lens.settings.enabled ? "bg-info" : "bg-muted-foreground/40")}
      />
      {scheduleLabel(item.lens, now)}
    </span>
  );
}

function StatusCell({ row: { original: item } }: Cell) {
  const { connected, now } = useList();
  if (item.kind === "finding")
    return (
      <span className="text-muted-foreground" title={formatActivityTimestamp(item.finding.last_seen)}>
        {item.finding.occurrences.length} {item.finding.occurrences.length === 1 ? "run" : "runs"} ·{" "}
        {agoLabel(Date.parse(item.finding.last_seen), now)}
      </span>
    );
  const latest = item.lens.jobs[0];
  return (
    <span
      className="flex items-center gap-1.5 truncate"
      title={latest ? formatActivityTimestamp(latest.created_at) : undefined}
    >
      <span className={cn(latest?.status === "failed" ? "text-destructive" : "text-muted-foreground")}>
        {lensStatus(item.lens, connected)}
      </span>
      {latest && <span className="text-muted-foreground">· {agoLabel(Date.parse(latest.created_at), now)}</span>}
    </span>
  );
}

function FindingCount({ row: { original: item } }: Cell) {
  if (item.kind === "finding") return null;
  const count = openFindings(item.lens).length;
  if (count === 0) return null;
  return (
    <span
      title={`${count} open ${count === 1 ? "finding" : "findings"}`}
      className="inline-flex min-w-5 justify-center rounded-full bg-muted px-1.5 font-mono text-xs tabular-nums text-foreground"
    >
      {count}
    </span>
  );
}

function ActionsCell({ row: { original: item } }: Cell) {
  const { readOnly, demo, onEdit, onRunNow } = useList();
  if (item.kind === "finding")
    return <ChevronRight aria-hidden="true" className="mr-2 ml-auto size-3.5 text-muted-foreground/60" />;
  if (readOnly && !demo) return null;
  const { id, settings } = item.lens;
  return (
    <span className="flex items-center justify-end gap-0.5">
      <button
        type="button"
        aria-label={`Run ${settings.name} now`}
        title={demo ? "Turn off Demo data to run an investigation" : "Run now"}
        disabled={readOnly}
        onClick={(event) => {
          event.stopPropagation();
          onRunNow(id);
        }}
        className={ACTION}
      >
        <Play className="size-3.5" />
      </button>
      <button
        type="button"
        aria-label={`Edit ${settings.name}`}
        title={demo ? "Turn off Demo data to edit an investigation" : "Edit"}
        disabled={readOnly}
        onClick={(event) => {
          event.stopPropagation();
          onEdit(id);
        }}
        className={cn(ACTION, "text-muted-foreground/60 group-hover:text-muted-foreground")}
      >
        <Pencil className="size-3.5" />
      </button>
    </span>
  );
}

const COLUMNS: ColumnDef<InvestigationRow>[] = [
  { id: "name", header: "Investigation", cell: NameCell },
  { id: "agent", size: 160, header: "Agent", cell: AgentCell },
  { id: "schedule", size: 180, header: "Schedule", cell: ScheduleCell },
  { id: "status", size: 200, header: "Last run", cell: StatusCell },
  { id: "findings", size: 64, header: "Open", cell: FindingCount, meta: { numeric: true } },
  {
    id: "actions",
    size: 76,
    header: "Actions",
    cell: ActionsCell,
    meta: { headerClassName: "sr-only", className: "pl-0 pr-3" },
  },
];

export interface InvestigationListProps {
  readonly lenses: readonly Lens[];
  readonly inbox: readonly InboxRow[];
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

export function InvestigationList({
  lenses,
  inbox,
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
  const { demo } = useLensRoute();
  const filters = useInboxFilters();
  const now = useNow(15000);
  const desktop = useMediaQuery("(min-width: 768px)");
  const shown = filterInvestigations([...lenses], search);
  const visibleIds = new Set(shown.map(({ id }) => id));
  const findings = filterInbox(inbox, filters);
  const shownFindings = findings.filter((finding) => finding.sources.some(({ lens }) => visibleIds.has(lens.id)));
  const findingsByOwner = groupBy(
    shownFindings,
    (finding) => finding.sources.find(({ lens }) => visibleIds.has(lens.id))?.lens.id ?? "",
  );
  const filtersActive = filters.agent !== "all" || filters.priority !== "all";
  const noFindingsMatch = shown.length > 0 && shownFindings.length === 0;
  const tableOptions: TableOptions<InvestigationRow> = {
    data: shown.map((lens): InvestigationRow => ({ kind: "investigation", lens })),
    columns: COLUMNS,
    defaultColumn: { size: undefined },
    state: { columnVisibility: { agent: desktop, schedule: desktop, status: desktop } },
    getRowId: investigationRowKey,
    getSubRows: (row) =>
      row.kind === "investigation" ? (findingsByOwner[row.lens.id] ?? []).map(findingRow) : undefined,
    initialState: { expanded: true },
    autoResetAll: false,
    getCoreRowModel: getCoreRowModel(),
    getExpandedRowModel: getExpandedRowModel(),
  };
  const table = useReactTable(tableOptions);
  const rows = table.getRowModel().rows.map((row) => row.original);
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
      <div className="flex min-h-0 flex-1 flex-col overflow-hidden bg-card">
        <div className="flex h-10 shrink-0 items-stretch border-b border-border bg-card">
          <SearchBox.Root
            className="h-full min-w-0"
            language={INVESTIGATION_QUERY}
            values={itemValues(INVESTIGATION_INDEX, lenses)}
            value={search}
            onValueChange={setSearch}
            label="Search investigations"
          >
            <SearchBox.Input
              className="h-full rounded-none border-0 px-3 focus-within:bg-muted/40 focus-within:ring-0 dark:bg-transparent"
              placeholder="Search investigations"
            />
            <SearchBox.Suggestions />
          </SearchBox.Root>
          {actions && <div className="flex shrink-0 items-stretch">{actions}</div>}
        </div>
        <InboxFilters rows={inbox} />
        <ListContext.Provider value={{ now, connected, readOnly, demo, onEdit, onRunNow }}>
          <InspectorTable.Root table={table}>
            <InspectorTable.Grid aria-label="Investigations" className="text-xs md:min-w-[860px]">
              <InspectorTable.Header />
              <InspectorTable.Body<InvestigationRow>
                rowHeight={(row) => (desktop && row.original.kind === "investigation" ? INVESTIGATION_HEIGHT : 48)}
              >
                {(row) => (
                  <InspectorTable.Row
                    row={row}
                    item={row.original}
                    tabIndex={0}
                    aria-label={
                      row.original.kind === "finding" ? row.original.finding.title : row.original.lens.settings.name
                    }
                    className={cn("group h-12", row.original.kind === "investigation" && "md:h-9")}
                  />
                )}
              </InspectorTable.Body>
            </InspectorTable.Grid>
            {!shown.length && (
              <div className="py-16 text-center text-xs text-muted-foreground">
                No investigations match your search.
              </div>
            )}
            {filtersActive && noFindingsMatch && (
              <p className="px-4 py-8 text-center text-xs text-muted-foreground">No findings match these filters.</p>
            )}
          </InspectorTable.Root>
        </ListContext.Provider>
        <footer className="flex h-8 shrink-0 items-center border-t bg-muted/30 px-3 text-xs text-muted-foreground">
          {shown.length} {shown.length === 1 ? "investigation" : "investigations"} ·{" "}
          {shown.filter((lens) => lens.settings.enabled).length} watching
          {" · "}
          {shownFindings.length} {shownFindings.length === 1 ? "finding" : "findings"}
        </footer>
      </div>
      <Inspector.Panel label={ROW_LABEL[noun]} testId="investigation-panel">
        {children}
      </Inspector.Panel>
    </Inspector.Root>
  );
}
