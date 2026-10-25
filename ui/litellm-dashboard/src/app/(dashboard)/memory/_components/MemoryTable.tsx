"use client";

import { useLanguage } from "@/contexts/LanguageContext";
import { OnChangeFn, PaginationState } from "@tanstack/react-table";
import { Database } from "lucide-react";
import React, { useMemo } from "react";

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
  const { t } = useLanguage();
  return (
    <div className="flex flex-col items-center gap-1 py-6">
      <div className="mb-1 flex size-10 items-center justify-center rounded-lg bg-muted">
        <Database className="size-5 text-muted-foreground" />
      </div>
      <div className="text-sm font-medium text-foreground">
        {hasActiveSearch ? t("No matching memories") : t("No memories stored yet")}
      </div>
      <div className="text-sm text-muted-foreground">
        {hasActiveSearch
          ? t("No memories match your search.")
          : t("Memories your agents store under /v1/memory will appear here.")}
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
  const { t } = useLanguage();
  const columns = useMemo(() => {
    const columnDeps = { onViewClick, onEditClick, onDeleteClick };
    return getMemoryTableColumns({ ...columnDeps, t });
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
      loadingMessage={t("Loading memories…")}
      noDataMessage={<MemoryEmptyState hasActiveSearch={hasActiveSearch} />}
      size="compact"
      toolbar={(table) => (
        <DataTableToolbar
          table={table}
          searchValue={searchValue}
          onSearchChange={onSearchChange}
          searchPlaceholder={t("Search by key prefix or memory ID…")}
          onRefresh={onRefresh}
          isRefreshing={isRefreshing}
          showViewOptions={false}
        />
      )}
    />
  );
}

export default MemoryTable;
