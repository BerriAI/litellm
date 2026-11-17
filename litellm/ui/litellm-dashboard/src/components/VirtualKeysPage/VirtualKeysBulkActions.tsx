"use client";

import { useIsMutating, useMutation, useQueryClient } from "@tanstack/react-query";
import type { Table } from "@tanstack/react-table";
import type { TFunction } from "i18next";
import { ChevronDown, DollarSign, ListChecks, Trash2, UserRound, UsersRound } from "lucide-react";
import { useState } from "react";
import { useTranslation } from "react-i18next";

import { invalidateKeyQueries } from "@/app/(dashboard)/hooks/keys/useKeys";
import DeleteResourceModal from "@/components/common_components/DeleteResourceModal";
import type { KeyResponse } from "@/components/key_team_helpers/key_list";
import { keyBulkUpdateCall, keyDeleteCall, type BulkKeyUpdateResponse } from "@/components/networking";
import { parseErrorMessage } from "@/components/shared/errorUtils";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { toast } from "@/lib/toast";
import { VirtualKeysBulkEditDialog, type BulkKeyEditAction, type BulkKeyEditValues } from "./VirtualKeysBulkEditDialog";

const BULK_MUTATION_KEY = ["keys", "bulkAction"] as const;
type KeyTarget = Readonly<Pick<KeyResponse, "token" | "key_alias">>;
type BulkDialog = {
  action: BulkKeyEditAction | "delete";
  targets: readonly KeyTarget[];
  failures?: BulkKeyUpdateResponse["failed_updates"];
};

interface VirtualKeysBulkActionsProps {
  table: Table<KeyResponse>;
  accessToken: string;
  disabled: boolean;
  canEdit: boolean;
  selectionMode: boolean;
  onSelectionModeChange: (enabled: boolean) => void;
}

function targetsAreEligible(table: Table<KeyResponse>, dialog: BulkDialog | null): boolean {
  if (!dialog || dialog.targets.length === 0) return false;
  return dialog.targets.every((target) => {
    const row = table.getRowModel().rows.find((row) => row.original.token === target.token);
    if (!row) return dialog.action !== "delete";
    if (!row.getCanSelect()) return false;
    return dialog.action !== "team" || !row.original.project_id;
  });
}

function bulkActionError(error: unknown, selectionChanged: boolean, isDelete: boolean, t: TFunction): string | null {
  if (error) return parseErrorMessage(error);
  if (!selectionChanged) return null;
  return t(
    isDelete
      ? "apiKeys.virtualKeysTable.bulkDelete.selectionChanged"
      : "apiKeys.virtualKeysTable.bulkEdit.selectionChanged",
  );
}

export function VirtualKeysBulkActions({
  table,
  accessToken,
  disabled,
  canEdit,
  selectionMode,
  onSelectionModeChange,
}: VirtualKeysBulkActionsProps) {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const [dialog, setDialog] = useState<BulkDialog | null>(null);
  const targets = dialog?.targets ?? [];
  const isDelete = dialog?.action === "delete";
  const isPending = useIsMutating({ mutationKey: BULK_MUTATION_KEY }) > 0;
  const deleteOptions = {
    mutationKey: [...BULK_MUTATION_KEY, "delete"],
    mutationFn: (keys: readonly KeyTarget[]) =>
      keyDeleteCall(
        accessToken,
        keys.map((key) => key.token),
      ),
    retry: false,
    onSuccess: (_: unknown, keys: readonly KeyTarget[]) =>
      toast.success(t("apiKeys.virtualKeysTable.bulkDelete.success", { count: keys.length })),
    onSettled: () => invalidateKeyQueries(queryClient),
  };
  const deletion = useMutation(deleteOptions);
  const updateOptions = {
    mutationKey: [...BULK_MUTATION_KEY, "update"],
    mutationFn: ({ keys, values }: { keys: readonly KeyTarget[]; values: BulkKeyEditValues }) =>
      keyBulkUpdateCall(
        accessToken,
        keys.map((key) => ({ key: key.token, ...values })),
      ),
    retry: false,
    onSuccess: (result: BulkKeyUpdateResponse) => {
      if (result.successful_updates.length > 0) {
        toast.success(t("apiKeys.virtualKeysTable.bulkEdit.success", { count: result.successful_updates.length }));
      }
    },
    onSettled: () => invalidateKeyQueries(queryClient),
  };
  const update = useMutation(updateOptions);

  const selectedRows = table.getSelectedRowModel().rows;
  const canOpen = !disabled && selectedRows.length > 0 && selectedRows.every((row) => row.getCanSelect());
  const canMoveToTeam = selectedRows.every(({ original }) => !original.project_id);
  const actionAllowed = isDelete || canEdit;
  const canConfirm = !disabled && actionAllowed && targetsAreEligible(table, dialog);

  const openDialog = (action: BulkDialog["action"]) => {
    if (!canOpen || isPending) return;
    if (action !== "delete" && !canEdit) return;
    if (action === "team" && !canMoveToTeam) return;
    deletion.reset();
    update.reset();
    setDialog({
      action,
      targets: selectedRows.map(({ original }) => ({ token: original.token, key_alias: original.key_alias })),
    });
  };

  const closeDialog = () => {
    if (queryClient.isMutating({ mutationKey: BULK_MUTATION_KEY }) > 0) return;
    setDialog(null);
    deletion.reset();
    update.reset();
  };

  const confirmDelete = () => {
    if (!canConfirm || queryClient.isMutating({ mutationKey: BULK_MUTATION_KEY }) > 0) return;
    deletion.mutate(targets, {
      onSuccess: () => {
        table.resetRowSelection();
        setDialog(null);
      },
    });
  };
  const confirmUpdate = (values: BulkKeyEditValues) => {
    if (!dialog || !canConfirm) return;
    if (!canEdit || queryClient.isMutating({ mutationKey: BULK_MUTATION_KEY }) > 0) return;
    update.mutate(
      { keys: dialog.targets, values },
      {
        onSuccess: (result) => {
          const succeeded = new Set(result.successful_updates.map((key) => key.key));
          table.setRowSelection((current) =>
            Object.fromEntries(Object.entries(current).filter(([token]) => !succeeded.has(token))),
          );
          const remaining = dialog.targets.filter((target) => !succeeded.has(target.token));
          setDialog(remaining.length > 0 ? { ...dialog, targets: remaining, failures: result.failed_updates } : null);
        },
      },
    );
  };

  const mutationError = isDelete ? deletion.error : update.error;
  const selectionChanged = Boolean(dialog) && !disabled && !canConfirm;
  const displayError = bulkActionError(mutationError, selectionChanged, isDelete, t);

  if (!selectionMode) {
    return (
      <Button variant="outline" size="sm" onClick={() => onSelectionModeChange(true)} disabled={disabled || isPending}>
        <ListChecks />
        {t("apiKeys.virtualKeysTable.bulkActions.select")}
      </Button>
    );
  }

  return (
    <>
      <span className="text-sm text-muted-foreground" role="status">
        {t("apiKeys.virtualKeysTable.bulkDelete.selected", { count: selectedRows.length })}
      </span>
      <Button variant="ghost" size="sm" onClick={() => onSelectionModeChange(false)} disabled={isPending}>
        {t("apiKeys.virtualKeysTable.bulkActions.cancel")}
      </Button>
      {selectedRows.length > 0 && (
        <DropdownMenu>
          <DropdownMenuTrigger render={<Button variant="outline" size="sm" />} disabled={!canOpen || isPending}>
            {t("apiKeys.virtualKeysTable.bulkActions.menu")}
            <ChevronDown />
          </DropdownMenuTrigger>
          <DropdownMenuContent className="min-w-44">
            {canEdit && (
              <>
                <DropdownMenuItem
                  onClick={() => openDialog("team")}
                  disabled={!canMoveToTeam}
                  title={!canMoveToTeam ? t("apiKeys.virtualKeysTable.bulkEdit.projectTeamLocked") : undefined}
                >
                  <UsersRound />
                  {t("apiKeys.virtualKeysTable.bulkEdit.team")}
                </DropdownMenuItem>
                <DropdownMenuItem onClick={() => openDialog("user")}>
                  <UserRound />
                  {t("apiKeys.virtualKeysTable.bulkEdit.user")}
                </DropdownMenuItem>
                <DropdownMenuItem onClick={() => openDialog("budget")}>
                  <DollarSign />
                  {t("apiKeys.virtualKeysTable.bulkEdit.budget")}
                </DropdownMenuItem>
                <DropdownMenuSeparator />
              </>
            )}
            <DropdownMenuItem variant="destructive" onClick={() => openDialog("delete")}>
              <Trash2 />
              {t("apiKeys.virtualKeysTable.bulkDelete.action", { count: selectedRows.length })}
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      )}
      <DeleteResourceModal
        isOpen={isDelete}
        title={t("apiKeys.virtualKeysTable.bulkDelete.title", { count: targets.length })}
        alertMessage={t("apiKeys.virtualKeysTable.bulkDelete.warning")}
        message={t("apiKeys.virtualKeysTable.bulkDelete.message")}
        resourceInformationTitle={t("apiKeys.virtualKeysTable.bulkDelete.selected", { count: targets.length })}
        resourceInformation={targets.flatMap((target, index) => [
          {
            label: t("apiKeys.virtualKeysTable.bulkDelete.keyAlias", { number: index + 1 }),
            value: target.key_alias || t("apiKeys.virtualKeysTable.bulkDelete.noAlias"),
          },
          {
            label: t("apiKeys.virtualKeysTable.bulkDelete.keyId", { number: index + 1 }),
            value: target.token,
            code: true,
          },
        ])}
        onCancel={closeDialog}
        onOk={confirmDelete}
        confirmLoading={isPending}
        confirmDisabled={!canConfirm}
        requiredConfirmation={t("apiKeys.virtualKeysTable.bulkDelete.requiredConfirmation")}
        errorMessage={displayError}
      />
      {dialog && dialog.action !== "delete" && (
        <VirtualKeysBulkEditDialog
          action={dialog.action}
          targets={dialog.targets}
          pending={isPending}
          disabled={!canConfirm}
          errorMessage={displayError}
          failures={dialog.failures}
          onCancel={closeDialog}
          onConfirm={confirmUpdate}
        />
      )}
    </>
  );
}
