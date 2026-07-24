"use client";

import React, { useState } from "react";
import moment from "moment";
import { AlertCircle, ScrollText } from "lucide-react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { uiSpendLogDetailsCall, uiSpendLogsCall } from "../networking";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import type { LogEntry } from "../view_logs/columns";

const LOGS_QUERY_KEY = "chat-user-logs";
const PAGE_SIZE = 50;

interface Props {
  accessToken: string;
  userId: string;
}

type TimeRange = "24h" | "7d" | "30d";

const TIME_RANGE_OPTIONS: { value: TimeRange; label: string }[] = [
  { value: "24h", label: "24h" },
  { value: "7d", label: "7d" },
  { value: "30d", label: "30d" },
];

function getStartMoment(range: TimeRange): moment.Moment {
  if (range === "24h") return moment().subtract(24, "hours");
  if (range === "7d") return moment().subtract(7, "days");
  return moment().subtract(30, "days");
}

interface PaginatedLogs {
  data: LogEntry[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

interface LogDetails {
  messages?: unknown;
  response?: unknown;
  proxy_server_request?: unknown;
}

function formatTokens(n: number): string {
  return (n ?? 0).toLocaleString();
}

function formatCost(spend: number): string {
  const value = spend ?? 0;
  if (value === 0) return "$0";
  if (value < 0.01) return `$${value.toFixed(6)}`;
  return `$${value.toFixed(4)}`;
}

function durationMs(row: LogEntry): number | null {
  if (row.request_duration_ms != null) return row.request_duration_ms;
  if (row.startTime && row.endTime) return Date.parse(row.endTime) - Date.parse(row.startTime);
  return null;
}

function formatDuration(row: LogEntry): string {
  const ms = durationMs(row);
  if (ms == null || Number.isNaN(ms)) return "-";
  return `${(ms / 1000).toFixed(2)}s`;
}

function StatusBadge({ status }: { status?: string }) {
  const isFailure = status === "failure";
  return (
    <span
      className={`inline-flex items-center gap-1.5 text-xs ${
        isFailure ? "text-red-600 dark:text-red-400" : "text-emerald-600 dark:text-emerald-400"
      }`}
    >
      <span className={`h-1.5 w-1.5 rounded-full ${isFailure ? "bg-red-500" : "bg-emerald-500"}`} />
      {isFailure ? "Failure" : "Success"}
    </span>
  );
}

function JsonBlock({ value }: { value: unknown }) {
  if (value == null || value === "") {
    return <p className="m-0 text-xs text-muted-foreground">Not available</p>;
  }
  const text = typeof value === "string" ? value : JSON.stringify(value, null, 2);
  return (
    <pre className="m-0 max-h-64 overflow-auto whitespace-pre-wrap break-words rounded-md border bg-muted/50 p-3 font-mono text-xs">
      {text}
    </pre>
  );
}

function LogsSkeleton() {
  return (
    <div className="overflow-hidden rounded-lg border">
      <div className="flex flex-col gap-px">
        {[...Array(8)].map((_, i) => (
          <div key={i} className="flex items-center gap-4 p-3">
            <Skeleton className="h-4 w-32" />
            <Skeleton className="h-4 w-40" />
            <Skeleton className="h-4 w-20" />
            <Skeleton className="h-4 w-16" />
          </div>
        ))}
      </div>
    </div>
  );
}

function LogsEmpty() {
  return (
    <div className="rounded-lg border border-dashed py-12 text-center text-sm text-muted-foreground">
      <ScrollText className="mx-auto mb-3 h-6 w-6 text-muted-foreground/50" />
      No logs for this period
    </div>
  );
}

function LogsError({ onRetry }: { onRetry: () => void }) {
  return (
    <div className="flex flex-col items-center gap-3 rounded-lg border border-dashed py-12 text-center text-sm text-muted-foreground">
      <AlertCircle className="h-6 w-6 text-destructive/70" />
      Failed to load your logs
      <Button variant="outline" size="sm" onClick={onRetry}>
        Retry
      </Button>
    </div>
  );
}

function LogsTable({ rows, onRowClick }: { rows: LogEntry[]; onRowClick: (row: LogEntry) => void }) {
  return (
    <div className="overflow-hidden rounded-lg border">
      <Table>
        <TableHeader>
          <TableRow className="bg-muted/50">
            <TableHead className="text-[11px] font-medium uppercase tracking-wide">Time</TableHead>
            <TableHead className="text-[11px] font-medium uppercase tracking-wide">Model</TableHead>
            <TableHead className="text-[11px] font-medium uppercase tracking-wide">Status</TableHead>
            <TableHead className="text-right text-[11px] font-medium uppercase tracking-wide">Tokens</TableHead>
            <TableHead className="text-right text-[11px] font-medium uppercase tracking-wide">Duration</TableHead>
            <TableHead className="text-right text-[11px] font-medium uppercase tracking-wide">Cost</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((row) => (
            <TableRow key={row.request_id} className="cursor-pointer" onClick={() => onRowClick(row)}>
              <TableCell className="whitespace-nowrap text-xs text-muted-foreground">
                {moment(row.startTime).format("MMM D, HH:mm:ss")}
              </TableCell>
              <TableCell className="text-sm">{row.model || "-"}</TableCell>
              <TableCell>
                <StatusBadge status={row.status} />
              </TableCell>
              <TableCell className="text-right text-sm tabular-nums">{formatTokens(row.total_tokens)}</TableCell>
              <TableCell className="text-right text-sm tabular-nums text-muted-foreground">
                {formatDuration(row)}
              </TableCell>
              <TableCell className="text-right text-sm tabular-nums">{formatCost(row.spend)}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}

function LogDetailDialog({
  log,
  details,
  isLoading,
  onClose,
}: {
  log: LogEntry | null;
  details: LogDetails | undefined;
  isLoading: boolean;
  onClose: () => void;
}) {
  return (
    <Dialog open={!!log} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle>Request details</DialogTitle>
          <DialogDescription className="break-all font-mono text-xs">{log?.request_id}</DialogDescription>
        </DialogHeader>
        {log && (
          <div className="flex flex-col gap-4">
            <div className="grid grid-cols-2 gap-3">
              <div className="rounded-md border bg-card p-3">
                <div className="mb-0.5 text-xs text-muted-foreground">Model</div>
                <div className="text-sm text-foreground">{log.model || "-"}</div>
              </div>
              <div className="rounded-md border bg-card p-3">
                <div className="mb-0.5 text-xs text-muted-foreground">Cost</div>
                <div className="text-sm text-foreground">{formatCost(log.spend)}</div>
              </div>
              <div className="rounded-md border bg-card p-3">
                <div className="mb-0.5 text-xs text-muted-foreground">Tokens</div>
                <div className="text-sm text-foreground">
                  {formatTokens(log.total_tokens)} ({formatTokens(log.prompt_tokens)} in /{" "}
                  {formatTokens(log.completion_tokens)} out)
                </div>
              </div>
              <div className="rounded-md border bg-card p-3">
                <div className="mb-0.5 text-xs text-muted-foreground">Duration</div>
                <div className="text-sm text-foreground">{formatDuration(log)}</div>
              </div>
            </div>

            <div className="flex flex-col gap-1.5">
              <div className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Request</div>
              {isLoading ? (
                <Skeleton className="h-16 w-full" />
              ) : (
                <JsonBlock value={details?.proxy_server_request ?? details?.messages} />
              )}
            </div>
            <div className="flex flex-col gap-1.5">
              <div className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Response</div>
              {isLoading ? <Skeleton className="h-16 w-full" /> : <JsonBlock value={details?.response} />}
            </div>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}

const LogsPanel: React.FC<Props> = ({ accessToken, userId }) => {
  const [timeRange, setTimeRange] = useState<TimeRange>("24h");
  const [page, setPage] = useState(1);
  const [selectedLog, setSelectedLog] = useState<LogEntry | null>(null);

  const startDate = getStartMoment(timeRange).utc().format("YYYY-MM-DD HH:mm:ss");
  const endDate = moment().utc().format("YYYY-MM-DD HH:mm:ss");

  const logsCallOptions = {
    accessToken,
    start_date: startDate,
    end_date: endDate,
    page,
    page_size: PAGE_SIZE,
    params: { user_id: userId, sort_by: "startTime", sort_order: "desc" as const },
  };
  const logsQueryOptions = {
    queryKey: [LOGS_QUERY_KEY, accessToken, userId, timeRange, page],
    queryFn: () => uiSpendLogsCall(logsCallOptions),
    enabled: !!accessToken && !!userId,
    placeholderData: keepPreviousData,
  };
  const { data, isLoading, isError, refetch } = useQuery(logsQueryOptions);

  const logs = data as PaginatedLogs | undefined;
  const rows = logs?.data ?? [];
  const totalPages = logs?.total_pages ?? 0;
  const total = logs?.total ?? 0;

  const detailStartDate = selectedLog ? moment(selectedLog.startTime).utc().format("YYYY-MM-DD HH:mm:ss") : "";
  const { data: detailData, isLoading: isDetailLoading } = useQuery({
    queryKey: [LOGS_QUERY_KEY, "detail", accessToken, selectedLog?.request_id, selectedLog?.startTime],
    queryFn: () => uiSpendLogDetailsCall(accessToken, selectedLog!.request_id, detailStartDate),
    enabled: !!accessToken && !!selectedLog,
  });
  const details = detailData as LogDetails | undefined;

  const renderBody = () => {
    if (isLoading) return <LogsSkeleton />;
    if (isError) return <LogsError onRetry={() => refetch()} />;
    if (rows.length === 0) return <LogsEmpty />;
    return (
      <>
        <LogsTable rows={rows} onRowClick={setSelectedLog} />
        <div className="mt-3 flex items-center justify-between">
          <p className="m-0 text-xs text-muted-foreground">
            {total.toLocaleString()} request{total === 1 ? "" : "s"}
            {totalPages > 1 ? ` · Page ${page} of ${totalPages}` : ""}
          </p>
          {totalPages > 1 && (
            <div className="flex gap-1">
              <Button variant="outline" size="sm" disabled={page <= 1} onClick={() => setPage((p) => p - 1)}>
                Previous
              </Button>
              <Button variant="outline" size="sm" disabled={page >= totalPages} onClick={() => setPage((p) => p + 1)}>
                Next
              </Button>
            </div>
          )}
        </div>
      </>
    );
  };

  return (
    <div className="w-full">
      <div className="mb-4 flex items-center justify-between">
        <div>
          <h2 className="mb-0.5 text-base font-semibold tracking-tight text-foreground">Your Logs</h2>
          <p className="m-0 text-sm text-muted-foreground">Request logs for your account only</p>
        </div>
        <div className="flex gap-1">
          {TIME_RANGE_OPTIONS.map((opt) => (
            <Button
              key={opt.value}
              variant={timeRange === opt.value ? "default" : "outline"}
              size="sm"
              onClick={() => {
                setTimeRange(opt.value);
                setPage(1);
              }}
            >
              {opt.label}
            </Button>
          ))}
        </div>
      </div>

      {renderBody()}

      <LogDetailDialog
        log={selectedLog}
        details={details}
        isLoading={isDetailLoading}
        onClose={() => setSelectedLog(null)}
      />
    </div>
  );
};

export default LogsPanel;
