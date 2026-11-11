"use client";

import { type OnChangeFn, type SortingState } from "@tanstack/react-table";
import { Inbox } from "lucide-react";
import React, { useMemo } from "react";
import { useTranslation } from "react-i18next";

import { DataTable } from "@/components/shared/DataTable";
import { SkillHubSkill } from "@/components/claude_code_plugins/types";

import { getPluginTableColumns } from "./PluginTableColumns";

interface PluginTableProps {
  pluginsList: SkillHubSkill[];
  sorting: SortingState;
  onSortingChange: OnChangeFn<SortingState>;
  isLoading: boolean;
  filtered?: boolean;
  onDeleteClick: (pluginName: string, displayName: string) => void;
  isAdmin: boolean;
  onPluginClick: (pluginId: string) => void;
}

function EmptyState({ filtered }: { filtered: boolean }) {
  const { t } = useTranslation();
  return (
    <div className="flex flex-col items-center gap-1 py-6">
      <div className="mb-1 flex size-10 items-center justify-center rounded-lg bg-muted">
        <Inbox className="size-5 text-muted-foreground" />
      </div>
      <div className="text-sm font-medium text-foreground">
        {t(filtered ? "aiHub.skillHubDashboard.emptyState.noMatchTitle" : "skills.table.empty.title")}
      </div>
      <div className="text-sm text-muted-foreground">
        {t(filtered ? "aiHub.skillHubDashboard.emptyState.noMatchBody" : "skills.table.empty.description")}
      </div>
    </div>
  );
}

const PluginTable: React.FC<PluginTableProps> = ({
  pluginsList,
  sorting,
  onSortingChange,
  isLoading,
  filtered = false,
  onDeleteClick,
  isAdmin,
  onPluginClick,
}) => {
  const { t } = useTranslation();

  const columns = useMemo(
    () => getPluginTableColumns({ isAdmin, onPluginClick, onDeleteClick }),
    [isAdmin, onPluginClick, onDeleteClick],
  );

  return (
    <DataTable
      data={pluginsList}
      paginationMode="none"
      columns={columns}
      getRowId={(plugin, index) => plugin.id || String(index)}
      sortingMode="server"
      sorting={sorting}
      onSortingChange={onSortingChange}
      isLoading={isLoading}
      loadingMessage={t("skills.table.loading")}
      noDataMessage={<EmptyState filtered={filtered} />}
      size="compact"
    />
  );
};

export default PluginTable;
