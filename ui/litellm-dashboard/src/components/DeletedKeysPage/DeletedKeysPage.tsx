"use client";
import { isQueryPending } from "@/app/(dashboard)/hooks/common/queryReadiness";
import { useState } from "react";
import { PaginationState } from "@tanstack/react-table";
import { Info } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/shared/Alert";
import { useDeletedKeys } from "@/app/(dashboard)/hooks/keys/useKeys";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { DeletedKeysTable } from "./DeletedKeysTable/DeletedKeysTable";

export default function DeletedKeysPage() {
  const { premiumUser } = useAuthorized();
  const [pagination, setPagination] = useState<PaginationState>({ pageIndex: 0, pageSize: 50 });

  const isLoadingQuery = useDeletedKeys(pagination.pageIndex + 1, pagination.pageSize);
  const { data: keysData } = isLoadingQuery;
  const isLoading = isQueryPending(isLoadingQuery);

  return (
    <div className="flex flex-col gap-4">
      {!premiumUser && (
        <Alert>
          <Info />
          <AlertTitle>Coming soon to Enterprise</AlertTitle>
          <AlertDescription>
            Deleted key auditing is graduating from beta into our Enterprise audit &amp; compliance suite.
          </AlertDescription>
        </Alert>
      )}
      <DeletedKeysTable
        keys={keysData?.keys || []}
        totalCount={keysData?.total_count || 0}
        isLoading={isLoading}
        pagination={pagination}
        onPaginationChange={setPagination}
      />
    </div>
  );
}
