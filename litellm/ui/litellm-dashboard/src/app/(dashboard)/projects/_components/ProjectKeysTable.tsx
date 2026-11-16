"use client";

import { OnChangeFn, PaginationState } from "@tanstack/react-table";
import { KeyRound } from "lucide-react";
import { useRouter } from "next/navigation";
import { useMemo } from "react";
import { useTranslation } from "react-i18next";

import { KeyResponse } from "@/components/key_team_helpers/key_list";
import { DataTable } from "@/components/shared/DataTable";
import { keyDetailHref } from "@/utils/entityLinks";

import { getProjectKeysTableColumns } from "./ProjectKeysTableColumns";
import { PROJECT_KEYS_PAGE_SIZE_OPTIONS } from "./useProjectsUrlState";

interface ProjectKeysTableProps {
  keys: KeyResponse[];
  totalCount: number;
  isLoading: boolean;
  isError?: boolean;
  pagination: PaginationState;
  onPaginationChange: OnChangeFn<PaginationState>;
}

function EmptyState() {
  const { t } = useTranslation();
  return (
    <div className="flex flex-col items-center gap-1 py-6">
      <div className="mb-1 flex size-10 items-center justify-center rounded-lg bg-muted">
        <KeyRound className="size-5 text-muted-foreground" />
      </div>
      <div className="text-sm font-medium text-foreground">{t("projects.keys.emptyState.title")}</div>
      <div className="text-sm text-muted-foreground">{t("projects.keys.emptyState.description")}</div>
    </div>
  );
}

export function ProjectKeysTable({
  keys,
  totalCount,
  isLoading,
  isError = false,
  pagination,
  onPaginationChange,
}: ProjectKeysTableProps) {
  const router = useRouter();
  const { t } = useTranslation();
  const columns = useMemo(() => getProjectKeysTableColumns(t), [t]);

  return (
    <DataTable
      data={keys}
      columns={columns}
      getRowId={(key, index) => key.token || String(index)}
      paginationMode="server"
      pagination={pagination}
      onPaginationChange={onPaginationChange}
      rowCount={totalCount}
      pageSizeOptions={PROJECT_KEYS_PAGE_SIZE_OPTIONS}
      isLoading={isLoading}
      isError={isError}
      loadingMessage={t("projects.keys.loading")}
      noDataMessage={<EmptyState />}
      size="compact"
      onRowClick={(key) => {
        if (key.token) router.push(keyDetailHref(key.token));
      }}
      getRowLabel={(key) => t("projects.keys.openKeyRowLabel", { name: key.key_alias || key.key_name || key.token })}
    />
  );
}
