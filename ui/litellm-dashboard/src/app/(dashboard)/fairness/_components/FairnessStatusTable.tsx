"use client";

import { useQuery } from "@tanstack/react-query";
import * as React from "react";

import { Badge } from "@/components/ui/badge";
import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { fetchClient } from "@/lib/http/api";

import { FAIRNESS_STATUS_QUERY_KEY } from "./FairnessSettingsForm";
import type { FairnessStatusResponse, ModelFairnessStatus, WorkloadClassStatus } from "./schema";

const REFRESH_INTERVAL_MS = 5_000;

const defaultFetchStatus = async (): Promise<FairnessStatusResponse> => {
  const { data } = await fetchClient.GET("/fairness/status");
  if (data === undefined) throw new Error("Failed to load fairness status");
  return data;
};

const percent = (fraction: number): string => `${Math.round(fraction * 100)}%`;
const seconds = (value: number): string => `${Math.round(value * 100) / 100}s`;
const limit = (value: number | null): string => (value === null ? "no limit" : value.toLocaleString());

export const notServedTotal = (row: WorkloadClassStatus): number =>
  row.rejected_capacity_total + row.rejected_queue_full_total + row.rejected_deadline_total + row.disconnected_total;

const ClassRow = ({ row }: { row: WorkloadClassStatus }) => (
  <TableRow data-testid={`class-row-${row.name}`}>
    <TableCell className="font-medium">{row.name}</TableCell>
    <TableCell>
      {percent(row.reserved_share)}
      <span className="block text-xs text-muted-foreground">
        {limit(row.reserved_rpm)} rpm / {limit(row.reserved_tpm)} tpm
      </span>
    </TableCell>
    <TableCell>
      {row.current_requests.toLocaleString()} req / {row.current_tokens.toLocaleString()} tok
    </TableCell>
    <TableCell>
      {row.queue_depth}
      <span className="block text-xs text-muted-foreground">deadline {seconds(row.max_queue_wait_seconds)}</span>
    </TableCell>
    <TableCell>
      {row.queued_total.toLocaleString()} queued, {row.admitted_after_wait_total.toLocaleString()} admitted
      <span className="block text-xs text-muted-foreground">avg wait {seconds(row.avg_queue_wait_seconds)}</span>
    </TableCell>
    <TableCell>
      {notServedTotal(row).toLocaleString()}
      <span className="block text-xs text-muted-foreground">
        {`${row.rejected_capacity_total} capacity, ${row.rejected_queue_full_total} queue full, ` +
          `${row.rejected_deadline_total} deadline, ${row.disconnected_total} disconnected`}
      </span>
    </TableCell>
  </TableRow>
);

const ModelSection = ({ model, threshold }: { model: ModelFairnessStatus; threshold: number }) => (
  <div className="flex flex-col gap-2" data-testid={`model-status-${model.model_group}`}>
    <div className="flex flex-wrap items-center gap-2">
      <p className="text-sm font-medium">{model.model_group}</p>
      <Badge variant={model.enforcing_reservations ? "destructive" : "secondary"}>
        {model.enforcing_reservations ? "saturated, reservations enforced" : "borrowing allowed"}
      </Badge>
      <span className="text-xs text-muted-foreground">
        {percent(model.saturation)} of {limit(model.rpm)} rpm / {limit(model.tpm)} tpm (threshold {percent(threshold)}
        ), {model.current_requests.toLocaleString()} req / {model.current_tokens.toLocaleString()} tok in window
      </span>
    </div>
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Class</TableHead>
          <TableHead>Reserved</TableHead>
          <TableHead>In window</TableHead>
          <TableHead>Queue depth</TableHead>
          <TableHead>Waited</TableHead>
          <TableHead>Not served</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {model.classes.map((row) => (
          <ClassRow key={row.name} row={row} />
        ))}
      </TableBody>
    </Table>
  </div>
);

const StatusCard = ({ action, children }: { action?: React.ReactNode; children: React.ReactNode }) => (
  <Card>
    <CardHeader>
      <CardTitle>Live status</CardTitle>
      <CardDescription>
        Per-model saturation and per-class usage, queue depth, wait time and rejections. Counters cover the current
        rate-limit window; wait and rejection totals cover the last few minutes.
      </CardDescription>
      {action !== undefined && <CardAction>{action}</CardAction>}
    </CardHeader>
    <CardContent className="flex flex-col gap-6">{children}</CardContent>
  </Card>
);

export interface FairnessStatusTableProps {
  fetchStatus?: () => Promise<FairnessStatusResponse>;
  refreshIntervalMs?: number | false;
}

export const FairnessStatusTable = ({
  fetchStatus = defaultFetchStatus,
  refreshIntervalMs = REFRESH_INTERVAL_MS,
}: FairnessStatusTableProps) => {
  const { data, isPending, isError } = useQuery({
    queryKey: FAIRNESS_STATUS_QUERY_KEY,
    queryFn: fetchStatus,
    refetchInterval: refreshIntervalMs,
  });

  if (isPending) {
    return (
      <StatusCard>
        <Skeleton className="h-40 w-full" />
      </StatusCard>
    );
  }

  if (isError) {
    return (
      <StatusCard>
        <p role="alert">Could not load the fairness status.</p>
      </StatusCard>
    );
  }

  const badge = data.limiter_active ? (
    <Badge>limiter active</Badge>
  ) : (
    <Badge variant="outline">{data.enabled ? "limiter not registered" : "disabled"}</Badge>
  );

  return (
    <StatusCard action={badge}>
      {data.models.length === 0 ? (
        <p className="text-sm text-muted-foreground">
          {data.limiter_active
            ? "No model groups configured. Fairness only applies to model groups that declare an RPM or TPM limit."
            : "Enable fairness and save the settings to start collecting per-model status."}
        </p>
      ) : (
        data.models.map((model) => (
          <ModelSection key={model.model_group} model={model} threshold={data.saturation_threshold} />
        ))
      )}
    </StatusCard>
  );
};
