"use client";

import { OnChangeFn, PaginationState } from "@tanstack/react-table";
import { Database } from "lucide-react";
import React, { useMemo } from "react";
import { useTranslation } from "react-i18next";

import { MemoryRow } from "@/components/networking";
import { DataTable, DataTableToolbar } from "@/components/shared/DataTable";

import { getMemoryTableColumns } from "./MemoryTableColumns";

interface MemoryTableProps {
  data: MemoryRow[];
  isLoading: boolean;
  rowCount: number;
  pagination: PaginationState;
  onPaginationChange: OnChangeFn<PaginationState>;
  searchValue: string;
  onSearchChange: (value: string) => void;
  isRefreshing: boolean;
  onRefresh: () => void;
  hasActiveSearch: boolean;
  onViewClick: (row: MemoryRow) => void;
  onEditClick: (row: MemoryRow) => void;
  onDeleteClick: (row: MemoryRow) => void;
}

function MemoryEmptyState({ hasActiveSearch }: { hasActiveSearch: boolean }) {
  const { t } = useTranslation();
  return (
    <div className="flex flex-col items-center gap-1 py-6">
      <div className="mb-1 flex size-10 items-center justify-center rounded-lg bg-muted">
        <Database className="size-5 text-muted-foreground" />
      </div>
      <div className="text-sm font-medium text-foreground">
        {hasActiveSearch ? t("memory.table.empty.titleSearch") : t("memory.table.empty.title")}
      </div>
      <div className="text-sm text-muted-foreground">
        {hasActiveSearch ? t("memory.table.empty.descriptionSearch") : t("memory.table.empty.description")}
      </div>
    </div>
  );
}

export function MemoryTable({
  data,
  isLoading,
  rowCount,
  pagination,
  onPaginationChange,
  searchValue,
  onSearchChange,
  isRefreshing,
  onRefresh,
  hasActiveSearch,
  onViewClick,
  onEditClick,
  onDeleteClick,
}: MemoryTableProps) {
  const { t } = useTranslation();
  const columns = useMemo(() => {
    const columnDeps = { onViewClick, onEditClick, onDeleteClick, t };
    return getMemoryTableColumns(columnDeps);
  }, [onViewClick, onEditClick, onDeleteClick, t]);

  return (
    <DataTable
      data={data}
      columns={columns}
      getRowId={(row) => row.memory_id}
      paginationMode="server"
      pagination={pagination}
      onPaginationChange={onPaginationChange}
      rowCount={rowCount}
      isLoading={isLoading}
      loadingMessage={t("memory.table.loading")}
      noDataMessage={<MemoryEmptyState hasActiveSearch={hasActiveSearch} />}
      size="compact"
      toolbar={(table) => (
        <DataTableToolbar
          table={table}
          searchValue={searchValue}
          onSearchChange={onSearchChange}
          searchPlaceholder={t("memory.table.searchPlaceholder")}
          onRefresh={onRefresh}
          isRefreshing={isRefreshing}
          showViewOptions={false}
        />
      )}
    />
  );
}

export default MemoryTable;
