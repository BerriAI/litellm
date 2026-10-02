"use client";
import { isQueryPending } from "@/app/(dashboard)/hooks/common/queryReadiness";
import { PaginationState } from "@tanstack/react-table";
import { Info } from "lucide-react";
import { useState } from "react";
import { Alert, AlertDescription, AlertTitle } from "@/components/shared/Alert";
import { DEFAULT_PAGE_SIZE_OPTIONS } from "@/components/shared/DataTable";
import { useDeletedTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { DeletedTeamsTable } from "./DeletedTeamsTable/DeletedTeamsTable";

export default function DeletedTeamsPage() {
  const { premiumUser } = useAuthorized();
  const [pagination, setPagination] = useState<PaginationState>({
    pageIndex: 0,
    pageSize: DEFAULT_PAGE_SIZE_OPTIONS[0],
  });
  const isLoadingQuery = useDeletedTeams(pagination.pageIndex + 1, pagination.pageSize);
  const { data: teamsData } = isLoadingQuery;
  const isLoading = isQueryPending(isLoadingQuery);

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
        pagination={pagination}
        onPaginationChange={setPagination}
        rowCount={teamsData?.total ?? 0}
      />
    </div>
  );
}
