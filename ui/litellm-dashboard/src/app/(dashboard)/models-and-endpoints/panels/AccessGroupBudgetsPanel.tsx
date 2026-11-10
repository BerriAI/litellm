"use client";

import { SortingState } from "@tanstack/react-table";
import { Inbox } from "lucide-react";
import React, { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";

import DeleteResourceModal from "@/components/common_components/DeleteResourceModal";
import { DataTable } from "@/components/shared/DataTable";
import { toast } from "@/lib/toast";
import { isProxyAdminRole } from "@/utils/roles";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { ModelAccessGroup, useModelAccessGroups } from "@/app/(dashboard)/hooks/modelAccessGroups/useModelAccessGroups";
import { useDeleteModelAccessGroupBudget } from "@/app/(dashboard)/hooks/modelAccessGroups/useDeleteModelAccessGroupBudget";
import {
  SetModelAccessGroupBudgetParams,
  useSetModelAccessGroupBudget,
} from "@/app/(dashboard)/hooks/modelAccessGroups/useSetModelAccessGroupBudget";
import AccessGroupBudgetModal from "@/app/(dashboard)/models-and-endpoints/components/AccessGroupBudgetModal";
import { getAccessGroupBudgetColumns } from "@/app/(dashboard)/models-and-endpoints/components/AccessGroupBudgetColumns";

const DEFAULT_SORTING: SortingState = [{ id: "access_group", desc: false }];

function EmptyState() {
  const { t } = useTranslation();
  return (
    <div className="flex flex-col items-center gap-1 py-6">
      <div className="mb-1 flex size-10 items-center justify-center rounded-lg bg-muted">
        <Inbox className="size-5 text-muted-foreground" />
      </div>
      <div className="text-sm font-medium text-foreground">
        {t("modelsAndEndpoints.accessGroupBudgetsPanel.empty.title")}
      </div>
      <div className="text-sm text-muted-foreground">
        {t("modelsAndEndpoints.accessGroupBudgetsPanel.empty.description")}
      </div>
    </div>
  );
}

export default function AccessGroupBudgetsPanel() {
  const { t } = useTranslation();
  const { userRole } = useAuthorized();
  const { data: accessGroups, isLoading } = useModelAccessGroups();
  const setBudget = useSetModelAccessGroupBudget();
  const clearBudget = useDeleteModelAccessGroupBudget();

  const [sorting, setSorting] = useState<SortingState>(DEFAULT_SORTING);
  const [editing, setEditing] = useState<ModelAccessGroup | null>(null);
  const [clearing, setClearing] = useState<ModelAccessGroup | null>(null);

  const canWrite = isProxyAdminRole(userRole ?? "");
  const columns = useMemo(
    () => getAccessGroupBudgetColumns({ canWrite, onSetBudget: setEditing, onClearBudget: setClearing }),
    [canWrite],
  );

  const handleSubmit = (params: SetModelAccessGroupBudgetParams) => {
    if (!editing) return;
    const accessGroup = editing.access_group;
    setBudget.mutate(
      { accessGroup, params },
      {
        onSuccess: () => {
          toast.success(t("modelsAndEndpoints.accessGroupBudgetsPanel.toasts.saved", { accessGroup }));
          setEditing(null);
        },
      },
    );
  };

  const handleConfirmClear = () => {
    if (!clearing) return;
    const accessGroup = clearing.access_group;
    clearBudget.mutate(accessGroup, {
      onSuccess: () => {
        toast.success(t("modelsAndEndpoints.accessGroupBudgetsPanel.toasts.cleared", { accessGroup }));
        setClearing(null);
      },
    });
  };

  return (
    <div className="flex flex-col gap-4">
      <p className="text-sm text-muted-foreground">{t("modelsAndEndpoints.accessGroupBudgetsPanel.description")}</p>

      <DataTable
        data={accessGroups ?? []}
        paginationMode="client"
        columns={columns}
        getRowId={(group) => group.access_group}
        sortingMode="client"
        sorting={sorting}
        onSortingChange={setSorting}
        isLoading={isLoading}
        loadingMessage={t("modelsAndEndpoints.accessGroupBudgetsPanel.loading")}
        noDataMessage={<EmptyState />}
        size="compact"
      />

      <AccessGroupBudgetModal
        accessGroup={editing}
        isSaving={setBudget.isPending}
        onCancel={() => setEditing(null)}
        onSubmit={handleSubmit}
      />

      <DeleteResourceModal
        isOpen={clearing !== null}
        title={t("modelsAndEndpoints.accessGroupBudgetsPanel.clearModal.title")}
        message={t("modelsAndEndpoints.accessGroupBudgetsPanel.clearModal.message")}
        resourceInformationTitle={t("modelsAndEndpoints.accessGroupBudgetsPanel.clearModal.informationTitle")}
        resourceInformation={[
          {
            label: t("modelsAndEndpoints.accessGroupBudgetsPanel.clearModal.accessGroup"),
            value: clearing?.access_group ?? null,
            code: true,
          },
          {
            label: t("modelsAndEndpoints.accessGroupBudgetsPanel.clearModal.maxBudget"),
            value: clearing?.budget?.max_budget?.toString() ?? null,
          },
        ]}
        onCancel={() => setClearing(null)}
        onOk={handleConfirmClear}
        confirmLoading={clearBudget.isPending}
      />
    </div>
  );
}
