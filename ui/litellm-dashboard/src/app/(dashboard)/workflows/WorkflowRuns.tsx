import React, { useState, useEffect, useCallback, useMemo } from "react";
import { skipToken, useQuery, type UseQueryOptions } from "@tanstack/react-query";
import { ArrowLeft, ChevronDown, RefreshCw } from "lucide-react";
import { functionalUpdate, type ColumnDef } from "@tanstack/react-table";
import { parseAsString, useQueryState } from "nuqs";
import { getGlobalLitellmHeaderName, proxyBaseUrl } from "@/components/networking";
import {
  DataTable,
  DataTableFilterDrawer,
  DataTableFilterField,
  DataTableToolbar,
  usePersistedColumnVisibility,
  useUrlTableState,
  type UrlTableStateOptions,
} from "@/components/shared/DataTable";
import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Sheet, SheetContent, SheetDescription, SheetTitle } from "@/components/ui/sheet";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { cn } from "@/lib/cva.config";

interface WorkflowRunsProps {
  accessToken: string | null;
}

type RunStatus = "pending" | "running" | "paused" | "completed" | "failed";

interface RunMetadata {
  title?: string;
  state?: string;
  pr_url?: string;
  worktree_path?: string;
  plan_text?: string;
  grill_session_id?: string;
  session_id?: string;
  [key: string]: unknown;
}

interface WorkflowRun {
  run_id: string;
  status: RunStatus;
  workflow_type: string;
  created_at: string;
  metadata?: RunMetadata | null;
}

interface WorkflowRunEvent {
  event_id: string;
  event_type: string;
  step_name: string;
  sequence_number: number;
  created_at: string;
  data?: Record<string, unknown> | null;
}

interface WorkflowRunMessage {
  message_id: string;
  role: string;
  content: string;
  sequence_number: number;
  created_at: string;
}

interface WorkflowRunDetail {
  events: WorkflowRunEvent[];
  messages: WorkflowRunMessage[];
}

type RunFilterColumn = "status" | "workflow_type";

const TABLE_STATE_OPTIONS: UrlTableStateOptions<RunFilterColumn> = {
  sortFields: [],
  defaultSort: { id: "created_at", desc: true },
  defaultPageSize: 50,
  filterColumns: ["status", "workflow_type"],
  urlKeys: { filter_workflow_type: "filter_type" },
};

const RUN_PARAM = parseAsString.withOptions({ history: "push" });

// ── design tokens ─────────────────────────────────────────────────────────────

const STATUS_DOT: Record<RunStatus, string> = {
  pending: "bg-border",
  running: "bg-info",
  paused: "bg-warning",
  completed: "bg-success",
  failed: "bg-destructive",
};

const RUN_STATUS_OPTIONS: readonly RunStatus[] = ["pending", "running", "paused", "completed", "failed"];
const STATUS_LABELS: Record<RunStatus, string> = {
  pending: "Pending",
  running: "Running",
  paused: "Paused",
  completed: "Completed",
  failed: "Failed",
};

const isRunStatus = (value: string): value is RunStatus => RUN_STATUS_OPTIONS.some((status) => status === value);

const stateLabel = (state: string): string => (isRunStatus(state) ? STATUS_LABELS[state] : state);

const EVENT_COLOR: Record<string, { bar: string; text: string }> = {
  "step.started": { bar: "border-success/30 bg-success/10", text: "text-success" },
  "step.failed": { bar: "border-destructive/30 bg-destructive/10", text: "text-destructive" },
  "hook.waiting": { bar: "border-warning/30 bg-warning/10", text: "text-warning" },
  "hook.received": { bar: "border-info/30 bg-info/10", text: "text-info" },
};

function eventStyle(type: string) {
  return EVENT_COLOR[type] ?? { bar: "border-border bg-muted", text: "text-muted-foreground" };
}

// ── helpers ───────────────────────────────────────────────────────────────────

function timeAgo(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  if (isNaN(diff)) return iso;
  const s = Math.floor(diff / 1000);
  if (s < 60) return `${s}s ago`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}

function fmtDuration(ms: number): string {
  if (ms < 0) return "";
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

function runTitle(run: WorkflowRun): string {
  const t = run.metadata?.title;
  if (t) return String(t);
  return run.workflow_type ?? run.run_id.slice(0, 8);
}

function shortId(id: string): string {
  return id.slice(0, 8);
}

function displayedState(run: WorkflowRun): string {
  return run.metadata?.state ?? run.status;
}

function bySequence<T extends { sequence_number: number }>(items: readonly T[] | undefined): T[] {
  return [...(items ?? [])].sort((a, b) => a.sequence_number - b.sequence_number);
}

async function fetchRunDetail(accessToken: string, runId: string, signal: AbortSignal): Promise<WorkflowRunDetail> {
  const runUrl = `${proxyBaseUrl ?? ""}/v1/workflows/runs/${encodeURIComponent(runId)}`;
  const init = { headers: { [getGlobalLitellmHeaderName()]: `Bearer ${accessToken}` }, signal };
  const [evRes, msgRes] = await Promise.all([fetch(`${runUrl}/events`, init), fetch(`${runUrl}/messages`, init)]);
  const evData: { events?: WorkflowRunEvent[] } = evRes.ok ? await evRes.json() : {};
  const msgData: { messages?: WorkflowRunMessage[] } = msgRes.ok ? await msgRes.json() : {};
  return { events: bySequence(evData.events), messages: bySequence(msgData.messages) };
}

// ── status dot ────────────────────────────────────────────────────────────────

const StatusDot: React.FC<{ status: RunStatus; className?: string }> = ({ status, className }) => (
  <span className={cn("inline-block flex-none rounded-full", STATUS_DOT[status] ?? "bg-border", className)} />
);

// ── truncated text value ──────────────────────────────────────────────────────

const TRUNCATE_AT = 120;

const TruncatedValue: React.FC<{ value: string }> = ({ value }) => {
  const [expanded, setExpanded] = useState(false);
  if (value.length <= TRUNCATE_AT) {
    return <span className="break-all text-foreground">{value}</span>;
  }
  return (
    <span className="break-all text-foreground">
      {expanded ? value : value.slice(0, TRUNCATE_AT) + "…"}
      <Button variant="link" size="xs" className="h-auto px-1 py-0 text-[11px]" onClick={() => setExpanded((e) => !e)}>
        {expanded ? "less" : "more"}
      </Button>
    </span>
  );
};

// ── metadata card ─────────────────────────────────────────────────────────────

const MetadataCard: React.FC<{ run: WorkflowRun }> = ({ run }) => {
  const meta = run.metadata ?? {};

  const primaryFields: { key: string; label: string }[] = [
    { key: "state", label: "state" },
    { key: "worktree_path", label: "worktree" },
    { key: "grill_session_id", label: "grill session" },
    { key: "session_id", label: "session" },
  ];

  const primaryKeys = new Set(["title", ...primaryFields.map((f) => f.key)]);
  const extraEntries = Object.entries(meta).filter(
    ([k, v]) => !primaryKeys.has(k) && v !== null && v !== undefined && v !== "",
  );

  return (
    <div className="mb-4 overflow-hidden rounded-lg border">
      {/* title bar */}
      <div className="flex items-center gap-2.5 border-b px-5 py-3.5">
        <StatusDot status={run.status} className="size-2.5" />
        <span className="flex-1 text-sm font-semibold text-foreground">{runTitle(run)}</span>
        <span className="rounded bg-muted px-2 py-0.5 font-mono text-[11px] text-muted-foreground">
          {shortId(run.run_id)}
        </span>
        <span className="rounded bg-muted px-2 py-0.5 text-[11px] text-muted-foreground">{run.workflow_type}</span>
      </div>

      {/* key fields grid */}
      <div className="grid grid-cols-[repeat(auto-fill,minmax(220px,1fr))] gap-x-6 gap-y-2 px-5 py-3 font-mono text-xs">
        <FieldPair label="status">
          <span className="capitalize text-foreground">{run.status}</span>
        </FieldPair>
        <FieldPair label="created">
          <span className="text-foreground">{timeAgo(run.created_at)}</span>
        </FieldPair>

        {meta.pr_url && (
          <FieldPair label="pr">
            <a
              href={String(meta.pr_url)}
              target="_blank"
              rel="noopener noreferrer"
              className="break-all text-primary underline-offset-4 hover:underline"
            >
              {String(meta.pr_url)}
            </a>
          </FieldPair>
        )}

        {primaryFields.map(({ key, label }) => {
          const v = meta[key];
          if (v === null || v === undefined || v === "") return null;
          const str = typeof v === "object" ? JSON.stringify(v) : String(v);
          return (
            <FieldPair key={key} label={label}>
              <TruncatedValue value={str} />
            </FieldPair>
          );
        })}

        {extraEntries.map(([k, v]) => {
          const str = typeof v === "object" ? JSON.stringify(v) : String(v);
          return (
            <FieldPair key={k} label={k}>
              <TruncatedValue value={str} />
            </FieldPair>
          );
        })}
      </div>
    </div>
  );
};

const FieldPair: React.FC<{ label: string; children: React.ReactNode }> = ({ label, children }) => (
  <div className="flex flex-col gap-px">
    <span className="text-[10px] uppercase tracking-[0.06em] text-muted-foreground">{label}</span>
    <span className="text-xs">{children}</span>
  </div>
);

// ── gantt timeline ────────────────────────────────────────────────────────────

const GanttTimeline: React.FC<{
  run: WorkflowRun;
  events: WorkflowRunEvent[];
}> = ({ run, events }) => {
  if (events.length === 0) {
    return <div className="py-4 font-mono text-xs text-muted-foreground">No events recorded</div>;
  }

  const runStart = new Date(run.created_at).getTime();
  const eventTimes = events.map((e) => new Date(e.created_at).getTime());
  const lastTime = Math.max(...eventTimes);
  const totalSpan = Math.max(lastTime - runStart, 1);
  const totalDur = fmtDuration(lastTime - runStart);

  return (
    <TooltipProvider delay={300}>
      <div className="font-mono text-xs">
        {/* ruler */}
        <div className="mb-0.5 grid grid-cols-[160px_minmax(0,1fr)] gap-x-3">
          <div />
          <div className="relative h-4">
            <span className="absolute left-0 text-[10px] text-muted-foreground">0</span>
            <span className="absolute left-full -translate-x-full text-[10px] text-muted-foreground">{totalDur}</span>
          </div>
        </div>

        {/* outer run bar */}
        <div className="mb-1 grid grid-cols-[160px_minmax(0,1fr)] gap-x-3">
          <div className="truncate pt-0.5 text-foreground">{runTitle(run)}</div>
          <div className="flex h-6 items-center rounded border bg-muted pl-2">
            <span className="text-[11px] text-muted-foreground">{totalDur}</span>
          </div>
        </div>

        {/* event rows */}
        <div className="grid grid-cols-[160px_minmax(0,1fr)] gap-x-3 gap-y-[3px]">
          {events.map((ev) => {
            const evTime = new Date(ev.created_at).getTime();
            const leftPct = ((evTime - runStart) / totalSpan) * 100;

            const nextIdx = events.findIndex((e) => e.sequence_number > ev.sequence_number);
            const nextTime =
              nextIdx >= 0
                ? new Date(events[nextIdx].created_at).getTime()
                : lastTime + Math.max(totalSpan * 0.12, 500);
            const widthPct = Math.max(8, ((nextTime - evTime) / totalSpan) * 100);
            const style = eventStyle(ev.event_type);
            const dur = fmtDuration(nextTime - evTime);

            return (
              <React.Fragment key={ev.event_id}>
                <div className={cn("truncate pt-0.5 pl-3", style.text)}>{ev.step_name || ev.event_type}</div>
                <div className="relative h-6">
                  <Tooltip>
                    <TooltipTrigger
                      render={
                        <div
                          className={cn(
                            "absolute h-full cursor-default gap-1.5 overflow-hidden rounded border pl-2",
                            "flex items-center",
                            style.bar,
                          )}
                          style={{
                            left: `${Math.min(leftPct, 92)}%`,
                            width: `${Math.min(widthPct, 100 - Math.min(leftPct, 92))}%`,
                          }}
                        />
                      }
                    >
                      <span className={cn("whitespace-nowrap text-[11px]", style.text)}>{ev.event_type}</span>
                      {dur && <span className="whitespace-nowrap text-[11px] text-muted-foreground">{dur}</span>}
                    </TooltipTrigger>
                    <TooltipContent className="font-mono text-[11px] leading-relaxed">
                      <div className="flex flex-col">
                        <div>
                          <span className="opacity-70">type: </span>
                          <span>{ev.event_type}</span>
                        </div>
                        <div>
                          <span className="opacity-70">step: </span>
                          {ev.step_name}
                        </div>
                        <div>
                          <span className="opacity-70">seq: </span>
                          {ev.sequence_number}
                        </div>
                        <div>
                          <span className="opacity-70">time: </span>
                          {timeAgo(ev.created_at)}
                        </div>
                        {ev.data && Object.keys(ev.data).length > 0 && (
                          <div>
                            <span className="opacity-70">data: </span>
                            {JSON.stringify(ev.data)}
                          </div>
                        )}
                      </div>
                    </TooltipContent>
                  </Tooltip>
                </div>
              </React.Fragment>
            );
          })}
        </div>
      </div>
    </TooltipProvider>
  );
};

// ── message row ───────────────────────────────────────────────────────────────

const ROLE_COLOR: Record<string, string> = {
  user: "text-info",
  assistant: "text-success",
  system: "text-violet-600",
  tool_result: "text-warning",
};

const MessageRow: React.FC<{ msg: WorkflowRunMessage }> = ({ msg }) => (
  <div className="grid grid-cols-[80px_minmax(0,1fr)] items-start gap-x-4 border-b py-2.5 font-mono text-xs">
    <span className={cn("pt-px", ROLE_COLOR[msg.role] ?? "text-muted-foreground")}>[{msg.role}]</span>
    <div>
      <span className="block whitespace-pre-wrap break-words leading-relaxed text-foreground">{msg.content}</span>
      <span className="mt-0.5 block text-[11px] text-muted-foreground">{timeAgo(msg.created_at)}</span>
    </div>
  </div>
);

// ── collapsible section ───────────────────────────────────────────────────────

const DetailSection: React.FC<{
  title: string;
  meta: React.ReactNode;
  defaultOpen?: boolean;
  children: React.ReactNode;
}> = ({ title, meta, defaultOpen = false, children }) => (
  <Collapsible defaultOpen={defaultOpen}>
    <CollapsibleTrigger className="group flex w-full items-center gap-2 px-4 py-3 text-left text-xs font-medium text-foreground hover:bg-muted/50">
      <ChevronDown className="size-3.5 -rotate-90 text-muted-foreground transition-transform group-data-[panel-open]:rotate-0" />
      <span>
        {title}
        <span className="ml-1.5 text-[11px] font-normal text-muted-foreground">{meta}</span>
      </span>
    </CollapsibleTrigger>
    <CollapsibleContent className="px-4 pb-3">{children}</CollapsibleContent>
  </Collapsible>
);

// ── run detail drawer body ────────────────────────────────────────────────────

const DrawerSpinner: React.FC = () => (
  <div className="flex justify-center py-20">
    <UiLoadingSpinner className="size-8 text-muted-foreground" />
  </div>
);

const DrawerCloseButton: React.FC<{ onClose: () => void }> = ({ onClose }) => (
  <Button
    variant="ghost"
    size="sm"
    className="px-0 text-xs font-normal text-muted-foreground hover:bg-transparent"
    onClick={onClose}
  >
    <ArrowLeft />
    close
  </Button>
);

interface RunDetailBodyProps {
  run: WorkflowRun | undefined;
  runsLoading: boolean;
  detail: WorkflowRunDetail | undefined;
  detailLoading: boolean;
  onClose: () => void;
  onRefresh: () => void;
}

const RunDetailBody: React.FC<RunDetailBodyProps> = ({
  run,
  runsLoading,
  detail,
  detailLoading,
  onClose,
  onRefresh,
}) => {
  if (run === undefined) {
    return runsLoading ? (
      <DrawerSpinner />
    ) : (
      <div className="px-7 py-6">
        <DrawerCloseButton onClose={onClose} />
        <p className="mt-4 text-sm text-muted-foreground">Workflow run not found.</p>
      </div>
    );
  }
  if (detailLoading) {
    return <DrawerSpinner />;
  }

  const events = detail?.events ?? [];
  const messages = detail?.messages ?? [];

  return (
    <div className="px-7 py-6">
      <div className="mb-4 flex items-center justify-between">
        <DrawerCloseButton onClose={onClose} />
        <Button variant="outline" size="sm" onClick={onRefresh}>
          <RefreshCw />
          Refresh
        </Button>
      </div>
      <MetadataCard run={run} />
      <div className="divide-y overflow-hidden rounded-lg border">
        <DetailSection
          title="Timeline"
          meta={
            <>
              {events.length} {events.length === 1 ? "event" : "events"}
            </>
          }
          defaultOpen
        >
          <GanttTimeline run={run} events={events} />
        </DetailSection>
        <DetailSection title="Messages" meta={messages.length}>
          {messages.length === 0 ? (
            <div className="py-3 font-mono text-xs text-muted-foreground">No messages</div>
          ) : (
            <div>
              {messages.map((msg) => (
                <MessageRow key={msg.message_id} msg={msg} />
              ))}
            </div>
          )}
        </DetailSection>
      </div>
    </div>
  );
};

// ── main component ────────────────────────────────────────────────────────────

const WorkflowRuns: React.FC<WorkflowRunsProps> = ({ accessToken }) => {
  const [runs, setRuns] = useState<WorkflowRun[]>([]);
  const [loadingRuns, setLoadingRuns] = useState(Boolean(accessToken));
  const [filtersOpen, setFiltersOpen] = useState(false);
  const { search, setSearch, pagination, onPaginationChange, columnFilters, onColumnFiltersChange } =
    useUrlTableState(TABLE_STATE_OPTIONS);
  const { columnVisibility, onColumnVisibilityChange } = usePersistedColumnVisibility("workflow-runs");
  const [runId, setRunId] = useQueryState("run", RUN_PARAM);
  const [shownRunId, setShownRunId] = useState(runId);
  if (runId !== null && runId !== shownRunId) {
    setShownRunId(runId);
  }
  const shownRun = runs.find((run) => run.run_id === shownRunId);

  const detailQueryOptions: UseQueryOptions<WorkflowRunDetail> = {
    queryKey: ["workflow-run-detail", shownRun?.run_id],
    queryFn:
      accessToken && shownRun !== undefined
        ? ({ signal }) => fetchRunDetail(accessToken, shownRun.run_id, signal)
        : skipToken,
    enabled: runId !== null,
    refetchOnWindowFocus: false,
    retry: false,
  };
  const detailQuery = useQuery(detailQueryOptions);

  const closeRun = useCallback(() => void setRunId(null), [setRunId]);

  const fetchRuns = useCallback(async () => {
    if (!accessToken) return;
    setLoadingRuns(true);
    try {
      const res = await fetch(`${proxyBaseUrl ?? ""}/v1/workflows/runs?limit=100`, {
        headers: { [getGlobalLitellmHeaderName()]: `Bearer ${accessToken}` },
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setRuns(data.runs ?? []);
    } catch (err) {
      console.error("workflow runs fetch failed:", err);
    } finally {
      setLoadingRuns(false);
    }
  }, [accessToken]);

  useEffect(() => {
    fetchRuns();
  }, [fetchRuns]);

  const statusFilterItems = useMemo(() => {
    const states = new Set([...RUN_STATUS_OPTIONS, ...runs.map(displayedState)]);
    return Object.fromEntries([...states].map((state) => [state, stateLabel(state)]));
  }, [runs]);

  const columns = useMemo<ColumnDef<WorkflowRun, unknown>[]>(
    () => [
      {
        id: "run",
        accessorFn: (row) => `${runTitle(row)} ${row.run_id}`,
        header: "Run",
        meta: { title: "Run", skeleton: "twoLine" },
        cell: ({ row }) => {
          const run = row.original;
          return (
            <div className="flex items-center gap-2">
              <StatusDot status={run.status} className="size-[7px]" />
              <div>
                <div className="text-[13px] font-medium leading-snug text-foreground">{runTitle(run)}</div>
                <div className="font-mono text-[11px] text-muted-foreground">{shortId(run.run_id)}</div>
              </div>
            </div>
          );
        },
      },
      {
        accessorKey: "workflow_type",
        header: "Type",
        meta: { title: "Type" },
        filterFn: "includesString",
        cell: ({ row }) => (
          <span className="font-mono text-xs text-muted-foreground">{row.original.workflow_type}</span>
        ),
      },
      {
        id: "status",
        accessorFn: displayedState,
        header: "Status",
        meta: { title: "Status" },
        filterFn: "equalsString",
        cell: ({ row }) => {
          const run = row.original;
          return (
            <div className="flex items-center gap-1.5">
              <StatusDot status={run.status} className="size-[7px]" />
              <span className="text-xs capitalize text-muted-foreground">{displayedState(run)}</span>
            </div>
          );
        },
      },
      {
        accessorKey: "created_at",
        header: "Created",
        meta: { title: "Created" },
        cell: ({ row }) => <span className="text-xs text-muted-foreground">{timeAgo(row.original.created_at)}</span>,
      },
    ],
    [],
  );

  return (
    <div className="w-full px-8 py-6">
      {/* page header */}
      <div className="mb-5">
        <div className="text-lg font-semibold text-foreground">Workflow Runs</div>
        <div className="mt-0.5 text-[13px] text-muted-foreground">
          Durable state tracking for agents and automated workflows
        </div>
      </div>

      <DataTable
        data={runs}
        columns={columns}
        getRowId={(run) => run.run_id}
        isLoading={loadingRuns}
        loadingMessage="Loading workflow runs…"
        noDataMessage={<div className="py-6 text-center text-[13px] text-muted-foreground">No workflow runs yet</div>}
        paginationMode="client"
        pagination={pagination}
        onPaginationChange={onPaginationChange}
        pageSizeOptions={[50, 100]}
        filterMode="client"
        columnFilters={columnFilters}
        onColumnFiltersChange={onColumnFiltersChange}
        globalFilter={search}
        onGlobalFilterChange={(updater) => setSearch(functionalUpdate(updater, search))}
        columnVisibility={columnVisibility}
        onColumnVisibilityChange={onColumnVisibilityChange}
        onRowClick={(run) => void setRunId(run.run_id)}
        size="compact"
        toolbar={(table) => (
          <>
            <DataTableToolbar
              table={table}
              searchValue={search}
              onSearchChange={setSearch}
              searchPlaceholder="Search runs…"
              onRefresh={fetchRuns}
              isRefreshing={loadingRuns}
              onOpenFilters={() => setFiltersOpen(true)}
            />
            <DataTableFilterDrawer
              table={table}
              open={filtersOpen}
              onOpenChange={setFiltersOpen}
              title="Filters"
              description="Narrow down workflow runs"
            >
              {({ get, set }) => (
                <>
                  <DataTableFilterField label="Status">
                    <Select
                      items={statusFilterItems}
                      value={(get("status") as string) || null}
                      onValueChange={(value: string | null) => set("status", value ?? "")}
                    >
                      <SelectTrigger className="w-full" data-testid="filter-status">
                        <SelectValue placeholder="All statuses" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value={null}>All statuses</SelectItem>
                        {Object.entries(statusFilterItems).map(([state, label]) => (
                          <SelectItem key={state} value={state}>
                            {label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </DataTableFilterField>
                  <DataTableFilterField label="Type">
                    <Input
                      value={(get("workflow_type") as string) ?? ""}
                      onChange={(event) => set("workflow_type", event.target.value)}
                      placeholder="Filter by type…"
                    />
                  </DataTableFilterField>
                </>
              )}
            </DataTableFilterDrawer>
          </>
        )}
      />

      {/* detail drawer */}
      <Sheet
        open={runId !== null}
        onOpenChange={(open) => {
          if (!open) closeRun();
        }}
      >
        <SheetContent
          showCloseButton={false}
          className="overflow-y-auto p-0 data-[side=right]:w-full data-[side=right]:sm:max-w-[680px]"
        >
          <SheetTitle className="sr-only">Workflow run details</SheetTitle>
          <SheetDescription className="sr-only">
            Metadata, timeline and messages for the selected workflow run
          </SheetDescription>
          <RunDetailBody
            run={shownRun}
            runsLoading={loadingRuns}
            detail={detailQuery.data}
            detailLoading={detailQuery.isFetching}
            onClose={closeRun}
            onRefresh={() => void detailQuery.refetch()}
          />
        </SheetContent>
      </Sheet>
    </div>
  );
};

export default WorkflowRuns;
