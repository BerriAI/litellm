"use client";
import { useState } from "react";
import { useDeletedKeys } from "@/app/(dashboard)/hooks/keys/useKeys";
import { DeletedKeysTable } from "./DeletedKeysTable/DeletedKeysTable";

export default function DeletedKeysPage() {
  const [pageIndex, setPageIndex] = useState(0);
  const [pageSize] = useState(50);

  const {
    data: keysData,
    isPending: isLoading,
    isFetching,
  } = useDeletedKeys(pageIndex + 1, pageSize);

  return (
    <DeletedKeysTable
      keys={keysData?.keys || []}
      totalCount={keysData?.total_count || 0}
      isLoading={isLoading}
      isFetching={isFetching}
      pageIndex={pageIndex}
      pageSize={pageSize}
      onPageChange={setPageIndex}
    />
  );
}
