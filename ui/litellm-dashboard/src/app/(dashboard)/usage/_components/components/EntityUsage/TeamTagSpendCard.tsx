import type { ColumnDef } from "@tanstack/react-table";
import { Tags } from "lucide-react";
import React, { useEffect, useMemo, useState } from "react";

import { DataTable, DataTablePagination } from "@/components/shared/DataTable";
import { MoneyCell } from "@/components/shared/table_cells";
import type {
  DailyActivityTagPageResponse,
  DailyActivityTagSpendRow,
} from "@/components/UsagePage/dailyActivityApi";

import { Panel } from "../overview/Primitives";

import { ENTITY_API } from "./entityFetchFns";

interface TeamTagSpendCardProps {
  accessToken: string | null;
  startTime: Date | null;
  endTime: Date | null;
  teamId: string;
}

interface TagPageState {
  scope: object;
  page: number;
  response: DailyActivityTagPageResponse | null;
  failed: boolean;
}

interface TeamTagSpendPage {
  page: number;
  loading: boolean;
  failed: boolean;
  rows: DailyActivityTagSpendRow[];
  totalTags: number;
  onPageChange: (nextPage: number) => void;
}

const QUIET_HEADER = { headerClassName: "font-normal" };
const PAGE_SIZE = 50;

const columns: ColumnDef<DailyActivityTagSpendRow>[] = [
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

const useTeamTagSpendPage = ({
  accessToken,
  startTime,
  endTime,
  teamId,
}: TeamTagSpendCardProps): TeamTagSpendPage => {
  const startTimestamp = startTime?.getTime();
  const endTimestamp = endTime?.getTime();
  const requestScope = useMemo(
    () => ({ hasAccessToken: accessToken !== null, endTimestamp, startTimestamp, teamId }),
    [accessToken, endTimestamp, startTimestamp, teamId],
  );
  const [pageState, setPageState] = useState<{ scope: object; page: number }>({ scope: {}, page: 0 });
  const [tagPageState, setTagPageState] = useState<TagPageState | null>(null);
  const page = pageState.scope === requestScope ? pageState.page : 0;
  const currentScopeState = tagPageState?.scope === requestScope ? tagPageState : null;
  const currentPageState = currentScopeState?.page === page ? currentScopeState : null;
  const enabled = Boolean(accessToken && startTimestamp !== undefined && endTimestamp !== undefined);

  useEffect(() => {
    if (!accessToken || startTimestamp === undefined || endTimestamp === undefined) return;

    const controller = new AbortController();
    const tagPageRequest = {
      accessToken,
      startTime: new Date(startTimestamp),
      endTime: new Date(endTimestamp),
      teamId,
      offset: page * PAGE_SIZE,
      limit: PAGE_SIZE,
      signal: controller.signal,
    };
    void ENTITY_API.team.tagPage(tagPageRequest)
      .then((response) => {
        if (controller.signal.aborted) return;
        const nextPageState: TagPageState = { scope: requestScope, page, response, failed: false };
        setTagPageState(nextPageState);
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted) return;
        console.error("Team tag spend page request failed:", error);
        setTagPageState((current) => ({
          scope: requestScope,
          page,
          response: current?.scope === requestScope ? current.response : null,
          failed: true,
        }));
      });
    return () => controller.abort();
  }, [accessToken, endTimestamp, page, requestScope, startTimestamp, teamId]);

  const loading = enabled && currentPageState === null;
  const failed = currentPageState?.failed ?? false;
  const rows = failed ? [] : (currentPageState?.response?.tags ?? []);
  const totalTags = currentScopeState?.response?.total_tags ?? 0;
  const onPageChange = (nextPage: number) => setPageState({ scope: requestScope, page: nextPage });

  return { page, loading, failed, rows, totalTags, onPageChange };
};

const TeamTagSpendCard: React.FC<TeamTagSpendCardProps> = (props) => {
  const { page, loading, failed, rows, totalTags, onPageChange } = useTeamTagSpendPage(props);
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
      {totalTags > 0 && (
        <DataTablePagination
          page={page}
          pageSize={PAGE_SIZE}
          rowCount={totalTags}
          onPageChange={onPageChange}
          onPageSizeChange={() => undefined}
          pageSizeOptions={[PAGE_SIZE]}
          isLoading={loading}
        />
      )}
    </Panel>
  );
};

export default TeamTagSpendCard;
