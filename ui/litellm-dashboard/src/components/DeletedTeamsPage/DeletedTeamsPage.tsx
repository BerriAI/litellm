"use client";
import { Info } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/shared/Alert";
import { DEFAULT_PAGE_SIZE_OPTIONS, useUrlTableState, type UrlTableStateOptions } from "@/components/shared/DataTable";
import { useDeletedTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { DeletedTeamsTable } from "./DeletedTeamsTable/DeletedTeamsTable";
import { DELETED_TEAMS_SORT_FIELDS } from "./DeletedTeamsTable/DeletedTeamsTableColumns";

const TABLE_STATE_OPTIONS: UrlTableStateOptions<never> = {
  sortFields: DELETED_TEAMS_SORT_FIELDS,
  defaultSort: { id: "deleted_at", desc: true },
  defaultPageSize: DEFAULT_PAGE_SIZE_OPTIONS[0],
  filterColumns: [],
  keyPrefix: "deleted_teams_",
};

export default function DeletedTeamsPage() {
  const { premiumUser } = useAuthorized();
  const { sorting, onSortingChange, pagination, onPaginationChange } = useUrlTableState(TABLE_STATE_OPTIONS);
  const { data: teamsData, isLoading, isError } = useDeletedTeams(pagination.pageIndex + 1, pagination.pageSize);

  return (
    <div className="flex flex-col gap-4">
      {!premiumUser && (
        <Alert>
          <Info />
          <AlertTitle>Coming soon to Enterprise</AlertTitle>
          <AlertDescription>
            Deleted team auditing is graduating from beta into our Enterprise audit &amp; compliance suite.
          </AlertDescription>
        </Alert>
      )}
      <DeletedTeamsTable
        teams={teamsData?.teams ?? []}
        isLoading={isLoading}
        isError={isError}
        sorting={sorting}
        onSortingChange={onSortingChange}
        pagination={pagination}
        onPaginationChange={onPaginationChange}
        rowCount={teamsData?.total ?? 0}
      />
    </div>
  );
}
