"use client";
import { useState } from "react";
import { Info } from "lucide-react";
import { useDeletedKeys } from "@/app/(dashboard)/hooks/keys/useKeys";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { DeletedKeysTable } from "./DeletedKeysTable/DeletedKeysTable";

export default function DeletedKeysPage() {
  const { premiumUser } = useAuthorized();
  const [pageIndex, setPageIndex] = useState(0);
  const [pageSize] = useState(50);

  const {
    data: keysData,
    isPending: isLoading,
    isFetching,
  } = useDeletedKeys(pageIndex + 1, pageSize);

  return (
    <div className="flex flex-col gap-4">
      {!premiumUser && (
        <div className="flex gap-2 items-start p-3 rounded-md bg-blue-50 dark:bg-blue-950/30 border border-blue-200 dark:border-blue-900 text-blue-800 dark:text-blue-200">
          <Info className="h-4 w-4 mt-0.5 shrink-0" />
          <div>
            <div className="font-semibold">Coming soon to Enterprise</div>
            <div className="text-sm">
              Deleted key auditing is graduating from beta into our Enterprise
              audit &amp; compliance suite.
            </div>
          </div>
        </div>
      )}
      <DeletedKeysTable
        keys={keysData?.keys || []}
        totalCount={keysData?.total_count || 0}
        isLoading={isLoading}
        isFetching={isFetching}
        pageIndex={pageIndex}
        pageSize={pageSize}
        onPageChange={setPageIndex}
      />
    </div>
  );
}
