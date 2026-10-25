"use client";

import { useLanguage } from "@/contexts/LanguageContext";
import { OnChangeFn, PaginationState, SortingState } from "@tanstack/react-table";
import { Inbox } from "lucide-react";
import { useMemo, useState } from "react";

import { DataTable } from "@/components/shared/DataTable";
import { DeletedKeyResponse } from "@/app/(dashboard)/hooks/keys/useKeys";

import { getDeletedKeysTableColumns } from "./DeletedKeysTableColumns";

interface DeletedKeysTableProps {
  keys: DeletedKeyResponse[];
  totalCount: number;
  isLoading: boolean;
  pagination: PaginationState;
  onPaginationChange: OnChangeFn<PaginationState>;
}

const DEFAULT_SORTING: SortingState = [{ id: "deleted_at", desc: true }];

function EmptyState() {
  const { t } = useLanguage();
  return (
    <div className="flex flex-col items-center gap-1 py-6">
      <div className="mb-1 flex size-10 items-center justify-center rounded-lg bg-muted">
        <Inbox className="size-5 text-muted-foreground" />
      </div>
      <div className="text-sm font-medium text-foreground">{t("No deleted keys found")}</div>
      <div className="text-sm text-muted-foreground">{t("Keys deleted from this proxy will show up here.")}</div>
    </div>
  );
}

export function DeletedKeysTable({
  keys,
  totalCount,
  isLoading,
  pagination,
  onPaginationChange,
}: DeletedKeysTableProps) {
  const { t } = useLanguage();
  const [sorting, setSorting] = useState<SortingState>(DEFAULT_SORTING);

  const columns = useMemo(() => getDeletedKeysTableColumns(t), [t]);

  return (
    <DataTable
      fillHeight
      data={keys}
      columns={columns}
      getRowId={(key, index) => key.token || String(index)}
      sortingMode="client"
      sorting={sorting}
      onSortingChange={setSorting}
      paginationMode="server"
      pagination={pagination}
      onPaginationChange={onPaginationChange}
      rowCount={totalCount}
      isLoading={isLoading}
      loadingMessage={t("Loading deleted keys…")}
      noDataMessage={<EmptyState />}
      size="compact"
    />
  );
}
