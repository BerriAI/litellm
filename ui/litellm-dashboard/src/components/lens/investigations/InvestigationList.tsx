"use client";

import {
  getCoreRowModel,
  useReactTable,
  type CellContext,
  type ColumnDef,
  type TableOptions,
} from "@tanstack/react-table";
import { createContext, useContext, type ReactNode } from "react";
import { ChevronRight, Pencil, Play } from "lucide-react";

import { useMediaQuery } from "usehooks-ts";
import { useNow } from "@/hooks/useNow";
import { Inspector } from "@/components/shared/Inspector";
import { Panel } from "../ui/Panel";
import { StatCell, StatStrip } from "../ui/StatStrip";
import { InspectorTable } from "@/components/shared/InspectorTable";
import { formatActivityTimestamp } from "@/utils/activityTimestamp";
import { cn } from "@/lib/cva.config";
import { agoLabel, scopeLabel } from "../model/format";

import { findingKey, openFindings, scheduleLabel } from "../model/inbox";
import { investigationSummary, lensStatus } from "../model/status";
import { SearchBox } from "@/components/shared/search/SearchBox";
import { itemValues } from "@/components/shared/search/valueSource";
import { type Finding, type Lens } from "../model/types";
import { useLensRoute, useListSearchRoute } from "../route";
import { FINDING_PANEL_WIDTH_KEY } from "../storage";
import { filterInvestigations, INVESTIGATION_INDEX, INVESTIGATION_QUERY } from "./investigationQuery";

const ACTION =
  "inline-flex size-7 items-center justify-center rounded text-muted-foreground hover:bg-muted hover:text-foreground disabled:cursor-not-allowed disabled:opacity-50";
const INVESTIGATION_HEIGHT = 36;

/** One row of the list: an investigation, or an open finding shown under the investigation that owns it. */
export type InvestigationRow =
  | { readonly kind: "investigation"; readonly lens: Lens }
  | { readonly kind: "finding"; readonly lens: Lens; readonly finding: Finding };

export const investigationRowKey = (row: InvestigationRow): string =>
  row.kind === "investigation" ? `investigation:${row.lens.id}` : `finding:${findingKey(row.lens, row.finding)}`;

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

function NameCell({ row: { original: item } }: Cell) {
  return (
    <span className="flex min-w-0 flex-col">
      <span className="truncate font-medium text-foreground">{item.lens.settings.name}</span>
      <span className="truncate text-xs text-muted-foreground md:hidden">{scopeLabel(item.lens.settings)}</span>
    </span>
  );
}

function AgentCell({ row: { original: item } }: Cell) {
  return <span className="block truncate text-muted-foreground">{scopeLabel(item.lens.settings)}</span>;
}

function ScheduleCell({ row: { original: item } }: Cell) {
  const { now } = useList();
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
    <span title={`${count} open ${count === 1 ? "finding" : "findings"}`} className="tabular-nums text-foreground">
      {count}
    </span>
  );
}

function InvestigationStats({ lenses }: { lenses: readonly Lens[] }) {
  const summary = investigationSummary(lenses);
  const paused = summary.total - summary.watching;
  return (
    <StatStrip>
      <StatCell
        label="Watching"
        value={summary.watching.toLocaleString()}
        hint={
          paused > 0
            ? `${paused.toLocaleString()} paused of ${summary.total.toLocaleString()}`
            : "All investigations on"
        }
      />
      <StatCell
        label="Open findings"
        value={summary.openFindings.toLocaleString()}
        hint={`Across ${summary.total.toLocaleString()} ${summary.total === 1 ? "investigation" : "investigations"}`}
      />
      <StatCell
        label="Failed last run"
        value={
          <span className={summary.failedLastRun > 0 ? "text-destructive" : undefined}>
            {summary.failedLastRun.toLocaleString()}
          </span>
        }
        hint={summary.failedLastRun > 0 ? "Open one to see why" : "Every last run finished"}
      />
      <StatCell
        label="Analysis spend"
        value={`$${summary.spentThisMonth.toFixed(2)}`}
        hint="This month, all investigations"
      />
    </StatStrip>
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
  { id: "schedule", size: 190, header: "Schedule", cell: ScheduleCell },
  { id: "status", size: 230, header: "Last run", cell: StatusCell },
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
  const now = useNow(15000);
  const desktop = useMediaQuery("(min-width: 768px)");
  const shown = filterInvestigations([...lenses], search);
  const tableOptions: TableOptions<InvestigationRow> = {
    data: shown.map((lens): InvestigationRow => ({ kind: "investigation", lens })),
    columns: COLUMNS,
    defaultColumn: { size: undefined },
    state: { columnVisibility: { agent: desktop, schedule: desktop, status: desktop } },
    getRowId: investigationRowKey,
    autoResetAll: false,
    getCoreRowModel: getCoreRowModel(),
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
      <div className="flex min-h-0 flex-1 flex-col gap-3">
        <InvestigationStats lenses={lenses} />
        <Panel className="flex-1">
          <div className="flex min-h-12 shrink-0 flex-wrap items-center gap-2 border-b border-border px-3 py-2">
            <SearchBox.Root
              className="sm:max-w-96"
              language={INVESTIGATION_QUERY}
              values={itemValues(INVESTIGATION_INDEX, lenses)}
              value={search}
              onValueChange={setSearch}
              label="Search investigations"
            >
              <SearchBox.Input className="rounded-md" placeholder="Search investigations" />
              <SearchBox.Suggestions />
            </SearchBox.Root>
            {actions && <div className="ml-auto flex items-center gap-2">{actions}</div>}
          </div>
          <ListContext.Provider value={{ now, connected, readOnly, demo, onEdit, onRunNow }}>
            <InspectorTable.Root table={table}>
              <InspectorTable.Grid aria-label="Investigations" className="text-sm md:min-w-[860px]">
                <InspectorTable.Header />
                <InspectorTable.Body<InvestigationRow> rowHeight={() => (desktop ? INVESTIGATION_HEIGHT : 48)}>
                  {(row) => (
                    <InspectorTable.Row
                      row={row}
                      item={row.original}
                      tabIndex={0}
                      aria-label={
                        row.original.kind === "finding" ? row.original.finding.title : row.original.lens.settings.name
                      }
                      className="group h-12 md:h-9"
                    />
                  )}
                </InspectorTable.Body>
              </InspectorTable.Grid>
              {!shown.length && (
                <div className="py-16 text-center text-xs text-muted-foreground">
                  No investigations match your search.
                </div>
              )}
            </InspectorTable.Root>
          </ListContext.Provider>
          <footer className="flex h-9 shrink-0 items-center border-t px-3 text-xs text-muted-foreground">
            {shown.length} {shown.length === 1 ? "investigation" : "investigations"} ·{" "}
            {shown.filter((lens) => lens.settings.enabled).length} watching
          </footer>
        </Panel>
      </div>
      <Inspector.Panel label={ROW_LABEL[noun]} testId="investigation-panel">
        {children}
      </Inspector.Panel>
    </Inspector.Root>
  );
}
