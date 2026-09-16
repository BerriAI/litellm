"use client";

import { Inbox } from "lucide-react";
import React, { useMemo } from "react";

import { DataTable, useUrlTableState, type UrlTableStateOptions } from "@/components/shared/DataTable";

import { getSearchToolTableColumns, searchToolKey } from "./SearchToolTableColumns";
import { AvailableSearchProvider, SearchTool } from "./types";

interface SearchToolTableProps {
  searchTools: SearchTool[];
  isLoading: boolean;
  isError?: boolean;
  availableProviders: AvailableSearchProvider[];
  onView: (searchToolId: string) => void;
  onEdit: (searchToolId: string) => void;
  onDelete: (searchToolId: string) => void;
}

const TABLE_STATE_OPTIONS: UrlTableStateOptions<never> = {
  sortFields: ["search_tool_id", "search_tool_name", "created_at", "updated_at"],
  defaultSort: { id: "created_at", desc: true },
  defaultPageSize: 25,
  filterColumns: [],
};

function EmptyState() {
  return (
    <div className="flex flex-col items-center gap-1 py-6">
      <div className="mb-1 flex size-10 items-center justify-center rounded-lg bg-muted">
        <Inbox className="size-5 text-muted-foreground" />
      </div>
      <div className="text-sm font-medium text-foreground">No search tools configured</div>
      <div className="text-sm text-muted-foreground">Add a search tool to enable web search for your models.</div>
    </div>
  );
}

const SearchToolTable: React.FC<SearchToolTableProps> = ({
  searchTools,
  isLoading,
  isError,
  availableProviders,
  onView,
  onEdit,
  onDelete,
}) => {
  const { sorting, onSortingChange, pagination, onPaginationChange } = useUrlTableState(TABLE_STATE_OPTIONS);

  const columns = useMemo(() => {
    const deps = { availableProviders, onView, onEdit, onDelete };
    return getSearchToolTableColumns(deps);
  }, [availableProviders, onView, onEdit, onDelete]);

  return (
    <DataTable
      data={searchTools}
      paginationMode="client"
      pagination={pagination}
      onPaginationChange={onPaginationChange}
      columns={columns}
      getRowId={(tool, index) => searchToolKey(tool) || String(index)}
      sortingMode="client"
      sorting={sorting}
      onSortingChange={onSortingChange}
      isLoading={isLoading}
      isError={isError}
      loadingMessage="Loading search tools…"
      noDataMessage={<EmptyState />}
      size="compact"
    />
  );
};

export default SearchToolTable;
