import type { ColumnDef } from "@tanstack/react-table";
import { Tags } from "lucide-react";
import React, { useMemo } from "react";

import { DataTable } from "@/components/shared/DataTable";
import { MoneyCell } from "@/components/shared/table_cells";
import { toDailyData, type DailyActivityRequest } from "@/components/UsagePage/dailyActivityApi";

import { useAggregatedDailyActivity } from "../../hooks/useAggregatedDailyActivity";
import { Panel } from "../overview/Primitives";

import { ENTITY_API } from "./entityFetchFns";
import { teamTagSpendRows, type TeamTagSpendRow } from "./teamTagSpend";

interface TeamTagSpendCardProps {
  accessToken: string | null;
  startTime: Date | null;
  endTime: Date | null;
  teamId: string;
}

const QUIET_HEADER = { headerClassName: "font-normal" };

const columns: ColumnDef<TeamTagSpendRow>[] = [
  {
    header: "Tag",
    accessorKey: "tag",
    meta: QUIET_HEADER,
    cell: ({ row }) => <span className="font-medium text-foreground">{row.original.tag}</span>,
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

const TeamTagSpendCard: React.FC<TeamTagSpendCardProps> = ({ accessToken, startTime, endTime, teamId }) => {
  const tagSpendRequest: DailyActivityRequest = {
    accessToken: accessToken as string,
    startTime: startTime as Date,
    endTime: endTime as Date,
    entityIds: [teamId],
    groupBy: "tag",
  };
  const { data, loading, failed } = useAggregatedDailyActivity({
    fetch: () => ENTITY_API.team.aggregated(tagSpendRequest),
    enabled: Boolean(accessToken && startTime && endTime),
    deps: [accessToken, startTime, endTime, teamId],
  });
  const rows = useMemo(() => teamTagSpendRows(toDailyData(data)), [data]);

  return (
    <Panel icon={Tags} title="Spend Per Tag" testId="team-tag-spend-card">
      <p className="mb-3 text-xs text-muted-foreground">
        A request with several tags counts once per tag, so the rows can add up to more than the team total
      </p>
      <DataTable
        columns={columns}
        data={rows}
        getRowId={(row) => row.tag}
        isLoading={loading}
        maxBodyHeight={280}
        noDataMessage={
          failed ? "Could not load tag spend. Reload the page to try again" : "No tagged spend in this range"
        }
        size="compact"
      />
    </Panel>
  );
};

export default TeamTagSpendCard;
