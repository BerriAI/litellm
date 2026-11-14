"use client";

import { SortingState } from "@tanstack/react-table";
import { Inbox } from "lucide-react";
import React, { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";

import { DataTable } from "@/components/shared/DataTable";

import { getSearchToolTableColumns, searchToolKey } from "./SearchToolTableColumns";
import { AvailableSearchProvider, SearchTool } from "./types";

interface SearchToolTableProps {
  searchTools: SearchTool[];
  isLoading: boolean;
  availableProviders: AvailableSearchProvider[];
  onView: (searchToolId: string) => void;
  onEdit: (searchToolId: string) => void;
  onDelete: (searchToolId: string) => void;
}

const DEFAULT_SORTING: SortingState = [{ id: "created_at", desc: true }];

function EmptyState() {
  const { t } = useTranslation();
  return (
    <div className="flex flex-col items-center gap-1 py-6">
      <div className="mb-1 flex size-10 items-center justify-center rounded-lg bg-muted">
        <Inbox className="size-5 text-muted-foreground" />
      </div>
      <div className="text-sm font-medium text-foreground">{t("searchTools.table.empty.title")}</div>
      <div className="text-sm text-muted-foreground">{t("searchTools.table.empty.description")}</div>
    </div>
  );
}

const SearchToolTable: React.FC<SearchToolTableProps> = ({
  searchTools,
  isLoading,
  availableProviders,
  onView,
  onEdit,
  onDelete,
}) => {
  const { t } = useTranslation();
  const [sorting, setSorting] = useState<SortingState>(DEFAULT_SORTING);

  const columns = useMemo(() => {
    const deps = { availableProviders, onView, onEdit, onDelete, t };
    return getSearchToolTableColumns(deps);
  }, [availableProviders, onView, onEdit, onDelete, t]);

  return (
    <DataTable
      data={searchTools}
      paginationMode="client"
      columns={columns}
      getRowId={(tool, index) => searchToolKey(tool) || String(index)}
      sortingMode="client"
      sorting={sorting}
      onSortingChange={setSorting}
      isLoading={isLoading}
      loadingMessage={t("searchTools.table.loading")}
      noDataMessage={<EmptyState />}
      size="compact"
    />
  );
};

export default SearchToolTable;
