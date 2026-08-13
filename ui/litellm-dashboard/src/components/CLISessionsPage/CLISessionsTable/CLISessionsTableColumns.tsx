"use client";

import { ColumnDef } from "@tanstack/react-table";

import type { CLISessionResponse } from "@/app/(dashboard)/hooks/cliSessions/useCLISessions";
import { DataTableSortHeader } from "@/components/shared/DataTable";
import { DateCell, IdCell } from "@/components/shared/table_cells";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";

function RevokeCell({
  session,
  onRevoke,
  isRevoking,
}: {
  session: CLISessionResponse;
  onRevoke: (sessionId: string) => void;
  isRevoking: boolean;
}) {
  if (session.revoked_at) {
    return (
      <span className="text-xs text-muted-foreground" title={session.revoked_by ?? undefined}>
        Revoked
      </span>
    );
  }

  return (
    <AlertDialog>
      <AlertDialogTrigger
        render={
          <Button variant="outline" size="sm" disabled={isRevoking}>
            Revoke
          </Button>
        }
      />
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>Revoke this CLI session?</AlertDialogTitle>
          <AlertDialogDescription>
            {session.user_id} stops authenticating with this session within one cache interval. Running `lite login`
            again starts a new one.
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>Cancel</AlertDialogCancel>
          <AlertDialogAction variant="destructive" onClick={() => onRevoke(session.session_id)}>
            Revoke
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}

export const getCLISessionsTableColumns = (
  onRevoke: (sessionId: string) => void,
  isRevoking: boolean,
  canRevoke: boolean,
): ColumnDef<CLISessionResponse>[] => [
  {
    id: "session_id",
    accessorKey: "session_id",
    meta: { title: "Session ID" },
    header: "Session ID",
    size: 160,
    enableSorting: false,
    cell: ({ row }) => <IdCell value={row.original.session_id} variant="plain" />,
  },
  {
    id: "user_id",
    accessorKey: "user_id",
    meta: { title: "User" },
    header: "User",
    size: 160,
    enableSorting: false,
    cell: ({ row }) => <IdCell value={row.original.user_id} variant="plain" />,
  },
  {
    id: "team_id",
    accessorKey: "team_id",
    meta: { title: "Team" },
    header: "Team",
    size: 140,
    enableSorting: false,
    cell: ({ row }) =>
      row.original.team_id ? (
        <IdCell value={row.original.team_id} variant="plain" />
      ) : (
        <span className="text-muted-foreground">-</span>
      ),
  },
  {
    id: "created_at",
    accessorKey: "created_at",
    meta: { title: "Issued" },
    header: ({ column }) => <DataTableSortHeader column={column} title="Issued" />,
    size: 130,
    enableSorting: true,
    cell: ({ row }) => <DateCell value={row.original.created_at} />,
  },
  {
    id: "expires_at",
    accessorKey: "expires_at",
    meta: { title: "Expires" },
    header: ({ column }) => <DataTableSortHeader column={column} title="Expires" />,
    size: 130,
    enableSorting: true,
    cell: ({ row }) => <DateCell value={row.original.expires_at} />,
  },
  {
    id: "status",
    meta: { title: "Status" },
    header: "Status",
    size: 90,
    enableSorting: false,
    cell: ({ row }) =>
      row.original.revoked_at ? (
        <Badge variant="destructive">Revoked</Badge>
      ) : (
        <Badge variant="secondary">Active</Badge>
      ),
  },
  ...(canRevoke
    ? [
        {
          id: "actions",
          meta: { title: "Actions" },
          header: "",
          size: 90,
          enableSorting: false,
          cell: ({ row }) => <RevokeCell session={row.original} onRevoke={onRevoke} isRevoking={isRevoking} />,
        } satisfies ColumnDef<CLISessionResponse>,
      ]
    : []),
];
