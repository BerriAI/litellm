"use client";

import { SortingState } from "@tanstack/react-table";
import { Inbox } from "lucide-react";
import React, { useMemo, useState } from "react";

import { DataTable } from "@/components/shared/DataTable";
import { Plugin } from "@/components/claude_code_plugins/types";

import { getPluginTableColumns } from "./PluginTableColumns";

interface PluginTableProps {
  pluginsList: Plugin[];
  isLoading: boolean;
  onDeleteClick: (pluginName: string, displayName: string) => void;
  accessToken: string | null;
  isAdmin: boolean;
  onPluginClick: (pluginId: string) => void;
}

const DEFAULT_SORTING: SortingState = [{ id: "created_at", desc: true }];

function EmptyState() {
  return (
    <div className="flex flex-col items-center gap-1 py-6">
      <div className="mb-1 flex size-10 items-center justify-center rounded-lg bg-muted">
        <Inbox className="size-5 text-muted-foreground" />
      </div>
      <div className="text-sm font-medium text-foreground">No skills found</div>
      <div className="text-sm text-muted-foreground">Add one to get started.</div>
    </div>
  );
}

const PluginTable: React.FC<PluginTableProps> = ({ pluginsList, isLoading, onDeleteClick, isAdmin, onPluginClick }) => {
  const [sorting, setSorting] = useState<SortingState>(DEFAULT_SORTING);

  const columns = useMemo(
    () => getPluginTableColumns({ isAdmin, onPluginClick, onDeleteClick }),
    [isAdmin, onPluginClick, onDeleteClick],
  );

  return (
    <DataTable
      data={pluginsList}
      columns={columns}
      getRowId={(plugin, index) => plugin.id || String(index)}
      sortingMode="client"
      sorting={sorting}
      onSortingChange={setSorting}
      onRowClick={(plugin) => onPluginClick(plugin.id)}
      isLoading={isLoading}
      loadingMessage="Loading skills…"
      noDataMessage={<EmptyState />}
      size="compact"
    />
  );
};

export default PluginTable;
