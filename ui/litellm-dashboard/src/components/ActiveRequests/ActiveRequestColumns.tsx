"use client";

import type { ColumnDef } from "@tanstack/react-table";
import { useEffect, useState } from "react";
import { DataTableSortHeader } from "@/components/shared/DataTable";
import { Badge } from "@/components/ui/badge";
import type { ActiveRequest } from "./activeRequestsApi";

export const formatAge = (startedAt: number, now: number) => {
  const seconds = Math.max(0, Math.floor(now / 1000 - startedAt));
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds % 60;
  if (minutes < 60) return `${minutes}m ${remainder}s`;
  return `${Math.floor(minutes / 60)}h ${minutes % 60}m`;
};

const Dash = () => <span className="text-muted-foreground">—</span>;

const text = (value?: string | null) => (value ? <span>{value}</span> : <Dash />);

const AGE_WARNING_SECONDS = 60;

const AgeCell = ({ startedAt }: { startedAt: number }) => {
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    const interval = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(interval);
  }, []);

  const stale = now / 1000 - startedAt > AGE_WARNING_SECONDS;
  return (
    <Badge
      variant={stale ? "outline" : "secondary"}
      className={stale ? "border-amber-500/50 text-amber-700 dark:text-amber-400" : undefined}
    >
      {formatAge(startedAt, now)}
    </Badge>
  );
};

export const activeRequestColumns: ColumnDef<ActiveRequest>[] = [
  {
    id: "age",
    accessorKey: "started_at",
    meta: { title: "Age" },
    header: ({ column }) => <DataTableSortHeader column={column} title="Age" />,
    size: 110,
    enableSorting: true,
    sortingFn: (left, right) => right.original.started_at - left.original.started_at,
    cell: ({ row }) => <AgeCell startedAt={row.original.started_at} />,
  },
  {
    id: "model",
    accessorFn: (row) => row.model ?? "",
    meta: { title: "Model" },
    header: ({ column }) => <DataTableSortHeader column={column} title="Model" />,
    size: 190,
    enableSorting: true,
    cell: ({ row }) => text(row.original.model),
  },
  {
    id: "type",
    accessorFn: (row) => row.call_type ?? "",
    meta: { title: "Type" },
    header: "Type",
    size: 160,
    enableSorting: false,
    cell: ({ row }) => (
      <div className="flex items-center gap-2">
        {text(row.original.call_type)}
        {row.original.streaming && <Badge variant="outline">stream</Badge>}
      </div>
    ),
  },
  {
    id: "end_user_id",
    accessorFn: (row) => row.end_user_id ?? "",
    meta: { title: "End User" },
    header: ({ column }) => <DataTableSortHeader column={column} title="End User" />,
    size: 170,
    enableSorting: true,
    cell: ({ row }) => text(row.original.end_user_id),
  },
  {
    id: "user",
    accessorFn: (row) => row.user_id ?? "",
    meta: { title: "User" },
    header: ({ column }) => <DataTableSortHeader column={column} title="User" />,
    size: 220,
    enableSorting: true,
    cell: ({ row }) => (
      <div className="flex flex-col">
        {text(row.original.user_id)}
        {row.original.user_email && <span className="text-xs text-muted-foreground">{row.original.user_email}</span>}
      </div>
    ),
  },
  {
    id: "organization",
    accessorFn: (row) => row.organization_alias ?? row.organization_id ?? "",
    meta: { title: "Organization" },
    header: "Organization",
    size: 170,
    enableSorting: false,
    cell: ({ row }) => text(row.original.organization_alias || row.original.organization_id),
  },
  {
    id: "project",
    accessorFn: (row) => row.project_alias ?? row.project_id ?? "",
    meta: { title: "Project" },
    header: "Project",
    size: 170,
    enableSorting: false,
    cell: ({ row }) => text(row.original.project_alias || row.original.project_id),
  },
  {
    id: "team",
    accessorFn: (row) => row.team_alias ?? row.team_id ?? "",
    meta: { title: "Team" },
    header: "Team",
    size: 170,
    enableSorting: false,
    cell: ({ row }) => text(row.original.team_alias || row.original.team_id),
  },
  {
    id: "key",
    accessorFn: (row) => row.key_alias ?? row.key_fingerprint ?? "",
    meta: { title: "Key" },
    header: "Key",
    size: 170,
    enableSorting: false,
    cell: ({ row }) => text(row.original.key_alias || row.original.key_fingerprint),
  },
  {
    id: "route",
    accessorFn: (row) => row.route ?? "",
    meta: { title: "Route" },
    header: "Route",
    size: 180,
    enableSorting: false,
    cell: ({ row }) => text(row.original.route),
  },
  {
    id: "pod",
    accessorFn: (row) => row.pod ?? "",
    meta: { title: "Pod" },
    header: "Pod",
    size: 190,
    enableSorting: false,
    cell: ({ row }) => text(row.original.pod),
  },
  {
    id: "request_id",
    accessorKey: "request_id",
    meta: { title: "Request ID" },
    header: "Request ID",
    size: 260,
    enableSorting: false,
    cell: ({ row }) => <span className="font-mono text-xs">{row.original.request_id || "—"}</span>,
  },
];
