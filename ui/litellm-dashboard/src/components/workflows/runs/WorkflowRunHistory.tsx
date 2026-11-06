"use client";

import { useState } from "react";
import { useQuery, type UseQueryOptions } from "@tanstack/react-query";
import { ChevronLeft, ChevronRight, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { cn } from "@/lib/cva.config";
import { runStates, type RunStatusFilter, type WorkflowRunDataSource, type WorkflowRunPage } from "./contracts";
import { formatTimestamp, stateLabels } from "./presentation";
import { RunStatusBadge } from "./RunStatusBadge";

export interface WorkflowRunHistoryProps {
  readonly workflowId: string;
  readonly dataSource: WorkflowRunDataSource;
  readonly selectedRunId: string | null;
  readonly onSelectRun: (runId: string) => void;
  readonly pageSize?: number;
}

export function WorkflowRunHistory(props: WorkflowRunHistoryProps) {
  return <ScopedHistory key={`${props.dataSource.cacheKey}:${props.workflowId}:${props.pageSize}`} {...props} />;
}

function ScopedHistory({ workflowId, dataSource, selectedRunId, onSelectRun, pageSize = 20 }: WorkflowRunHistoryProps) {
  const [status, setStatus] = useState<RunStatusFilter>("all");
  const [cursors, setCursors] = useState<readonly (string | null)[]>([null]);
  const cursor = cursors[cursors.length - 1];
  const historyOptions: UseQueryOptions<WorkflowRunPage> = {
    queryKey: ["workflowRuns", dataSource.cacheKey, workflowId, status, cursor, pageSize],
    queryFn: ({ signal }) => {
      const request = { workflowId, cursor, status, limit: pageSize, signal };
      return dataSource.listRuns(request);
    },
    retry: false,
    refetchInterval: cursor === null ? 5000 : false,
  };
  const history = useQuery(historyOptions);
  const changeStatus = (value: RunStatusFilter | null) => {
    setStatus(value ?? "all");
    setCursors([null]);
  };
  const nextCursor = history.data?.next_cursor;
  return (
    <section aria-label="Workflow run history" className="min-w-0 overflow-hidden rounded-xl border bg-card">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b px-5 py-4">
        <div>
          <h2 className="text-base font-semibold">Run history</h2>
          <p className="mt-1 text-xs text-muted-foreground">Select a run to inspect its execution</p>
        </div>
        <div className="flex items-center gap-2">
          <Select value={status} onValueChange={changeStatus}>
            <SelectTrigger aria-label="Filter runs by status" size="sm">
              <SelectValue>{status === "all" ? "All statuses" : stateLabels[status]}</SelectValue>
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All statuses</SelectItem>
              {runStates.map((state) => (
                <SelectItem key={state} value={state}>
                  {stateLabels[state]}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button
            variant="outline"
            size="icon-sm"
            aria-label="Refresh run history"
            disabled={history.isFetching}
            onClick={() => void history.refetch()}
          >
            <RefreshCw className={cn(history.isFetching && "animate-spin")} />
          </Button>
        </div>
      </div>
      {history.isError && (
        <div
          role="alert"
          className="flex items-center justify-between gap-3 border-b bg-destructive/5 px-5 py-3 text-sm"
        >
          Could not load run history. Check your access and try again.
          <Button variant="outline" size="sm" onClick={() => void history.refetch()}>
            Retry history
          </Button>
        </div>
      )}
      <div aria-busy={history.isFetching} className="max-h-80 overflow-auto">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="pl-5">Run</TableHead>
              <TableHead>Status</TableHead>
              <TableHead>Started</TableHead>
              <TableHead className="pr-5">Snapshot</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {history.data?.runs.map((run) => (
              <TableRow key={run.run_id} data-state={selectedRunId === run.run_id ? "selected" : undefined}>
                <TableCell className="pl-5">
                  <Button
                    variant="link"
                    className="h-auto max-w-64 justify-start p-0 font-mono text-xs"
                    aria-pressed={selectedRunId === run.run_id}
                    onClick={() => onSelectRun(run.run_id)}
                  >
                    <span className="truncate">{run.run_id}</span>
                  </Button>
                </TableCell>
                <TableCell>
                  <RunStatusBadge state={run.state} />
                </TableCell>
                <TableCell className="whitespace-nowrap text-xs text-muted-foreground">
                  {formatTimestamp(run.created_at)}
                </TableCell>
                <TableCell className="max-w-48 truncate pr-5 font-mono text-xs" title={run.snapshot_id}>
                  {run.snapshot_id}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
        {history.isPending && (
          <p role="status" className="p-6 text-sm text-muted-foreground">
            Loading runs…
          </p>
        )}
        {history.data?.runs.length === 0 && (
          <p className="p-6 text-sm text-muted-foreground">No runs match this filter.</p>
        )}
      </div>
      <div className="flex items-center justify-between border-t px-5 py-3">
        <span className="text-xs text-muted-foreground">Page {cursors.length}</span>
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            disabled={cursors.length === 1 || history.isFetching}
            onClick={() => setCursors(cursors.slice(0, -1))}
          >
            <ChevronLeft />
            Previous
          </Button>
          <Button
            variant="outline"
            size="sm"
            disabled={!nextCursor || history.isFetching || history.isError}
            onClick={() => {
              if (nextCursor) setCursors([...cursors, nextCursor]);
            }}
          >
            Next
            <ChevronRight />
          </Button>
        </div>
      </div>
    </section>
  );
}
