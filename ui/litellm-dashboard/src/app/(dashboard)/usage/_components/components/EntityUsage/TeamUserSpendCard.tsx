import { useQuery } from "@tanstack/react-query";
import type { ColumnDef } from "@tanstack/react-table";
import { Download, Users } from "lucide-react";
import React, { useMemo } from "react";

import { teamSpendByUserCall } from "@/components/networking";
import { DataTable } from "@/components/shared/DataTable";
import { MoneyCell } from "@/components/shared/table_cells";
import { Button } from "@/components/ui/button";

import { Panel } from "../overview/Primitives";

import {
  buildTeamUserSpendCsv,
  downloadCsv,
  sortBySpendDesc,
  teamLabel,
  teamUserSpendCsvFileName,
  teamUserSpendRowId,
  userLabel,
  type TeamUserSpendRow,
} from "./teamUserSpend";

interface TeamUserSpendCardProps {
  accessToken: string | null;
  startTime: Date | null;
  endTime: Date | null;
  teamIds: string[];
}

const QUIET_HEADER = { headerClassName: "font-normal" };

const columns: ColumnDef<TeamUserSpendRow>[] = [
  { header: "Team", accessorFn: teamLabel, id: "team", meta: QUIET_HEADER, cell: ({ row }) => teamLabel(row.original) },
  {
    header: "User",
    accessorFn: userLabel,
    id: "user",
    meta: QUIET_HEADER,
    cell: ({ row }) => <span className="font-medium text-foreground">{userLabel(row.original)}</span>,
  },
  {
    header: "Spend",
    accessorKey: "spend",
    meta: { numeric: true, ...QUIET_HEADER },
    cell: ({ row }) => <MoneyCell value={row.original.spend} decimals={4} />,
  },
  {
    header: "Requests",
    accessorKey: "api_requests",
    meta: { numeric: true, ...QUIET_HEADER },
    cell: ({ row }) => row.original.api_requests.toLocaleString(),
  },
  {
    header: "Successful",
    accessorKey: "successful_requests",
    meta: { numeric: true, className: "text-success", ...QUIET_HEADER },
    cell: ({ row }) => row.original.successful_requests.toLocaleString(),
  },
  {
    header: "Failed",
    accessorKey: "failed_requests",
    meta: { numeric: true, className: "text-destructive", ...QUIET_HEADER },
    cell: ({ row }) => row.original.failed_requests.toLocaleString(),
  },
  {
    header: "Tokens",
    accessorKey: "total_tokens",
    meta: { numeric: true, ...QUIET_HEADER },
    cell: ({ row }) => row.original.total_tokens.toLocaleString(),
  },
];

const TeamUserSpendCard: React.FC<TeamUserSpendCardProps> = ({ accessToken, startTime, endTime, teamIds }) => {
  const hasTeams = teamIds.length > 0;
  const { data, isLoading } = useQuery({
    queryKey: ["teamSpendByUser", startTime?.toISOString(), endTime?.toISOString(), teamIds],
    queryFn: () =>
      accessToken && startTime && endTime ? teamSpendByUserCall(accessToken, startTime, endTime, teamIds) : null,
    enabled: Boolean(accessToken && startTime && endTime) && hasTeams,
  });
  const rows = useMemo(() => sortBySpendDesc(data?.results ?? []), [data]);

  return (
    <Panel
      icon={Users}
      title="Spend Per User Within Team"
      action={
        <Button
          variant="outline"
          size="xs"
          disabled={!data || rows.length === 0}
          onClick={() => data && downloadCsv(buildTeamUserSpendCsv(data), teamUserSpendCsvFileName(data))}
        >
          <Download />
          Download CSV
        </Button>
      }
    >
      <p className="mb-3 text-xs text-muted-foreground">
        Attributed per request from spend logs, so it includes JWT/SSO traffic that does not use a virtual key
      </p>
      <DataTable
        columns={columns}
        data={rows}
        getRowId={teamUserSpendRowId}
        isLoading={isLoading}
        maxBodyHeight={320}
        noDataMessage={teamIds.length === 0 ? "Select a team to see spend per user" : "No user spend in this range"}
        size="compact"
      />
    </Panel>
  );
};

export default TeamUserSpendCard;
