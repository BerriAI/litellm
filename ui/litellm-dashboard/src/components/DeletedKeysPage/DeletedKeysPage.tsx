"use client";
import { useState } from "react";
import { Alert } from "antd";
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
        <Alert
          type="info"
          banner
          showIcon
          message="Coming soon to Enterprise"
          description="Deleted key auditing is graduating from beta into our Enterprise audit & compliance suite."
        />
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
