import type { ColumnDef } from "@tanstack/react-table";
import { Tags } from "lucide-react";
import React, { useEffect, useMemo, useState } from "react";

import { DataTable, DataTablePagination, DEFAULT_PAGE_SIZE_OPTIONS } from "@/components/shared/DataTable";
import { MoneyCell } from "@/components/shared/table_cells";
import type { DailyActivityTagPageResponse, DailyActivityTagSpendRow } from "@/components/UsagePage/dailyActivityApi";

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
  pageKey: string;
  response: DailyActivityTagPageResponse | null;
  failed: boolean;
}

interface TeamTagSpendPage {
  page: number;
  pageSize: number;
  loading: boolean;
  failed: boolean;
  rows: DailyActivityTagSpendRow[];
  totalTags: number;
  onPageChange: (nextPage: number) => void;
  onPageSizeChange: (nextPageSize: number) => void;
}

const QUIET_HEADER = { headerClassName: "font-normal" };
const DEFAULT_PAGE_SIZE = 50;

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

const useTeamTagSpendPage = ({ accessToken, startTime, endTime, teamId }: TeamTagSpendCardProps): TeamTagSpendPage => {
  const startTimestamp = startTime?.getTime();
  const endTimestamp = endTime?.getTime();
  const requestScope = useMemo(
    () => ({ hasAccessToken: accessToken !== null, endTimestamp, startTimestamp, teamId }),
    [accessToken, endTimestamp, startTimestamp, teamId],
  );
  const [pageState, setPageState] = useState<{ scope: object; pageSize: number; page: number }>({
    scope: {},
    pageSize: DEFAULT_PAGE_SIZE,
    page: 0,
  });
  const [tagPageState, setTagPageState] = useState<TagPageState | null>(null);
  const pageSize = pageState.pageSize;
  const page = pageState.scope === requestScope ? pageState.page : 0;
  const pageKey = `${page}:${pageSize}`;
  const currentScopeState = tagPageState?.scope === requestScope ? tagPageState : null;
  const currentPageState = currentScopeState?.pageKey === pageKey ? currentScopeState : null;
  const enabled = Boolean(accessToken && startTimestamp !== undefined && endTimestamp !== undefined);

  useEffect(() => {
    if (!accessToken || startTimestamp === undefined || endTimestamp === undefined) return;

    const controller = new AbortController();
    const tagPageRequest = {
      accessToken,
      startTime: new Date(startTimestamp),
      endTime: new Date(endTimestamp),
      teamId,
      offset: page * pageSize,
      limit: pageSize,
      signal: controller.signal,
    };
    void ENTITY_API.team
      .tagPage(tagPageRequest)
      .then((response) => {
        if (controller.signal.aborted) return;
        const nextPageState: TagPageState = { scope: requestScope, pageKey, response, failed: false };
        setTagPageState(nextPageState);
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted) return;
        console.error("Team tag spend page request failed:", error);
        setTagPageState((current) => ({
          scope: requestScope,
          pageKey,
          response: current?.scope === requestScope ? current.response : null,
          failed: true,
        }));
      });
    return () => controller.abort();
  }, [accessToken, endTimestamp, page, pageKey, pageSize, requestScope, startTimestamp, teamId]);

  const loading = enabled && currentPageState === null;
  const failed = currentPageState?.failed ?? false;
  const rows = failed ? [] : currentPageState?.response?.tags ?? [];
  const totalTags = currentScopeState?.response?.total_tags ?? 0;
  const onPageChange = (nextPage: number) => setPageState({ scope: requestScope, pageSize, page: nextPage });
  const onPageSizeChange = (nextPageSize: number) =>
    setPageState({ scope: requestScope, pageSize: nextPageSize, page: 0 });

  return { page, pageSize, loading, failed, rows, totalTags, onPageChange, onPageSizeChange };
};

const TeamTagSpendCard: React.FC<TeamTagSpendCardProps> = (props) => {
  const { page, pageSize, loading, failed, rows, totalTags, onPageChange, onPageSizeChange } =
    useTeamTagSpendPage(props);
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
          pageSize={pageSize}
          rowCount={totalTags}
          onPageChange={onPageChange}
          onPageSizeChange={onPageSizeChange}
          pageSizeOptions={DEFAULT_PAGE_SIZE_OPTIONS}
          isLoading={loading}
        />
      )}
    </Panel>
  );
};

export default TeamTagSpendCard;
