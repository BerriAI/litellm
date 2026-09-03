"use client";

import { SortingState } from "@tanstack/react-table";
import { Inbox, Plus } from "lucide-react";
import React, { useMemo, useState } from "react";

import DeleteResourceModal from "@/components/common_components/DeleteResourceModal";
import { DataTable } from "@/components/shared/DataTable";
import { Button } from "@/components/ui/button";
import { toast } from "@/lib/toast";
import { isProxyAdminRole } from "@/utils/roles";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useDatabaseModelGroups } from "@/app/(dashboard)/hooks/models/useModels";
import { ModelAccessGroup, useModelAccessGroups } from "@/app/(dashboard)/hooks/modelAccessGroups/useModelAccessGroups";
import {
  CreateModelAccessGroupParams,
  useCreateModelAccessGroup,
} from "@/app/(dashboard)/hooks/modelAccessGroups/useCreateModelAccessGroup";
import { useDeleteModelAccessGroupBudget } from "@/app/(dashboard)/hooks/modelAccessGroups/useDeleteModelAccessGroupBudget";
import {
  SetModelAccessGroupBudgetParams,
  useSetModelAccessGroupBudget,
} from "@/app/(dashboard)/hooks/modelAccessGroups/useSetModelAccessGroupBudget";
import AccessGroupBudgetModal from "@/app/(dashboard)/models-and-endpoints/components/AccessGroupBudgetModal";
import CreateAccessGroupModal from "@/app/(dashboard)/models-and-endpoints/components/CreateAccessGroupModal";
import { getAccessGroupBudgetColumns } from "@/app/(dashboard)/models-and-endpoints/components/AccessGroupBudgetColumns";

const DEFAULT_SORTING: SortingState = [{ id: "access_group", desc: false }];

function EmptyState() {
  return (
    <div className="flex flex-col items-center gap-1 py-6">
      <div className="mb-1 flex size-10 items-center justify-center rounded-lg bg-muted">
        <Inbox className="size-5 text-muted-foreground" />
      </div>
      <div className="text-sm font-medium text-foreground">No model access groups yet</div>
      <div className="text-sm text-muted-foreground">
        Create one here or put a deployment in a group from its model settings, then give the group a shared budget.
      </div>
    </div>
  );
}

export default function AccessGroupBudgetsPanel() {
  const { userRole } = useAuthorized();
  const { data: accessGroups, isLoading } = useModelAccessGroups();
  const setBudget = useSetModelAccessGroupBudget();
  const clearBudget = useDeleteModelAccessGroupBudget();
  const createGroup = useCreateModelAccessGroup();
  const { data: databaseModelGroups, isLoading: isLoadingModels } = useDatabaseModelGroups();

  const [sorting, setSorting] = useState<SortingState>(DEFAULT_SORTING);
  const [editing, setEditing] = useState<ModelAccessGroup | null>(null);
  const [clearing, setClearing] = useState<ModelAccessGroup | null>(null);
  const [creating, setCreating] = useState(false);

  const canWrite = isProxyAdminRole(userRole ?? "");
  const columns = useMemo(
    () => getAccessGroupBudgetColumns({ canWrite, onSetBudget: setEditing, onClearBudget: setClearing }),
    [canWrite],
  );

  const existingGroups = useMemo(() => (accessGroups ?? []).map((group) => group.access_group), [accessGroups]);

  const handleCreate = (params: CreateModelAccessGroupParams) => {
    createGroup.mutate(params, {
      onSuccess: () => {
        toast.success(`Access group "${params.access_group}" created`);
        setCreating(false);
      },
    });
  };

  const handleSubmit = (params: SetModelAccessGroupBudgetParams) => {
    if (!editing) return;
    const accessGroup = editing.access_group;
    setBudget.mutate(
      { accessGroup, params },
      {
        onSuccess: () => {
          toast.success(`Budget saved for "${accessGroup}"`);
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
        toast.success(`Budget cleared for "${accessGroup}"`);
        setClearing(null);
      },
    });
  };

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-start justify-between gap-4">
        <p className="text-sm text-muted-foreground">
          A model access group can carry one budget that every key granted the group by name draws from together. Keys
          that reach the group&apos;s models through a wildcard or all-proxy-models are not charged against it.
        </p>
        {canWrite && (
          <Button size="sm" className="shrink-0" onClick={() => setCreating(true)} data-testid="create-access-group">
            <Plus />
            Create Access Group
          </Button>
        )}
      </div>

      <DataTable
        data={accessGroups ?? []}
        columns={columns}
        getRowId={(group) => group.access_group}
        sortingMode="client"
        sorting={sorting}
        onSortingChange={setSorting}
        isLoading={isLoading}
        loadingMessage="Loading model access groups…"
        noDataMessage={<EmptyState />}
        size="compact"
      />

      {creating && (
        <CreateAccessGroupModal
          existingGroups={existingGroups}
          modelOptions={databaseModelGroups}
          isLoadingModels={isLoadingModels}
          isSaving={createGroup.isPending}
          onCancel={() => setCreating(false)}
          onSubmit={handleCreate}
        />
      )}

      <AccessGroupBudgetModal
        accessGroup={editing}
        isSaving={setBudget.isPending}
        onCancel={() => setEditing(null)}
        onSubmit={handleSubmit}
      />

      <DeleteResourceModal
        isOpen={clearing !== null}
        title="Clear Budget"
        message="Are you sure you want to clear this access group's budget? The recorded shared spend is cleared with it, and the group's models stay available."
        resourceInformationTitle="Access Group"
        resourceInformation={[
          { label: "Access Group", value: clearing?.access_group ?? null, code: true },
          { label: "Max Budget", value: clearing?.budget?.max_budget?.toString() ?? null },
        ]}
        onCancel={() => setClearing(null)}
        onOk={handleConfirmClear}
        confirmLoading={clearBudget.isPending}
      />
    </div>
  );
}
