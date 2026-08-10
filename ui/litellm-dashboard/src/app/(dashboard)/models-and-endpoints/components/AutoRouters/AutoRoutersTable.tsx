"use client";

import { SortingState } from "@tanstack/react-table";
import { useMemo, useState } from "react";

import { DataTable } from "@/components/shared/DataTable";
import { AutoRouterIcon } from "@/components/shared/table_cells";

import { getAutoRoutersTableColumns } from "./AutoRoutersTableColumns";
import { AutoRouterRow } from "./autoRouterRows";
import { useTranslation } from "react-i18next";

interface AutoRoutersTableProps {
  routers: AutoRouterRow[];
  isLoading: boolean;
  canModify: boolean;
  onRouterClick: (row: AutoRouterRow) => void;
  onDeleteClick: (row: AutoRouterRow) => void;
}

const PAGE_SIZE_OPTIONS = [10, 25, 50];

function EmptyState({ canModify }: { canModify: boolean }) {
  const { t } = useTranslation("gateway");
  return (
    <div className="flex flex-col items-center gap-1 py-6">
      <div className="mb-1 flex size-10 items-center justify-center rounded-lg bg-muted">
        <AutoRouterIcon size={20} className="text-muted-foreground" />
      </div>
      <div className="text-sm font-medium text-foreground">{t("models.autoRouters.emptyTitle")}</div>
      <div className="text-sm text-muted-foreground">
        {canModify ? t("models.autoRouters.emptyCanCreate") : t("models.autoRouters.emptyReadOnly")}
      </div>
    </div>
  );
}

export function AutoRoutersTable({
  routers,
  isLoading,
  canModify,
  onRouterClick,
  onDeleteClick,
}: AutoRoutersTableProps) {
  const { t } = useTranslation("gateway");
  const [sorting, setSorting] = useState<SortingState>([]);

  const columns = useMemo(
    () => getAutoRoutersTableColumns({ canModify, onRouterClick, onDeleteClick, t }),
    [canModify, onRouterClick, onDeleteClick, t],
  );

  return (
    <DataTable
      data={routers}
      columns={columns}
      getRowId={(router) => router.id}
      sortingMode="client"
      sorting={sorting}
      onSortingChange={setSorting}
      paginationMode="client"
      pageSizeOptions={PAGE_SIZE_OPTIONS}
      isLoading={isLoading}
      loadingMessage={t("models.autoRouters.loading")}
      noDataMessage={<EmptyState canModify={canModify} />}
      size="compact"
    />
  );
}
