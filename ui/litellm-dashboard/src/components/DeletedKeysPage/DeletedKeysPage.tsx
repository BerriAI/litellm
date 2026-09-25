"use client";
import { useCallback } from "react";
import { OnChangeFn, SortingState } from "@tanstack/react-table";
import { Info } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/shared/Alert";
import { useUrlTableState, type UrlTableStateOptions } from "@/components/shared/DataTable";
import { useDeletedKeys } from "@/app/(dashboard)/hooks/keys/useKeys";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { DeletedKeysTable } from "./DeletedKeysTable/DeletedKeysTable";
import { DELETED_KEYS_SORT_FIELDS } from "./DeletedKeysTable/DeletedKeysTableColumns";

const TABLE_STATE_OPTIONS: UrlTableStateOptions<never> = {
  sortFields: DELETED_KEYS_SORT_FIELDS,
  defaultSort: { id: "deleted_at", desc: true },
  defaultPageSize: 50,
  filterColumns: [],
  keyPrefix: "deleted_keys_",
};

export default function DeletedKeysPage() {
  const { premiumUser } = useAuthorized();
  const { sorting, onSortingChange, pagination, onPaginationChange } = useUrlTableState(TABLE_STATE_OPTIONS);
  const sortLoadedPage = useCallback<OnChangeFn<SortingState>>(
    (updater) => {
      onSortingChange(updater);
      onPaginationChange(pagination);
    },
    [onSortingChange, onPaginationChange, pagination],
  );

  const { data: keysData, isLoading, isError } = useDeletedKeys(pagination.pageIndex + 1, pagination.pageSize);

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
        isError={isError}
        sorting={sorting}
        onSortingChange={sortLoadedPage}
        pagination={pagination}
        onPaginationChange={onPaginationChange}
      />
    </div>
  );
}
