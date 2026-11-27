"use client";

import { useId, type FormEvent } from "react";
import { useController } from "react-hook-form";
import { useTranslation } from "react-i18next";

import BudgetDurationDropdown from "@/components/common_components/budget_duration_dropdown";
import TeamDropdown from "@/components/common_components/team_dropdown";
import UserDropdown from "@/components/common_components/UserDropdown";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Field, FieldError, FieldLabel, FieldSet } from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import { useZodForm } from "@/lib/forms/useZodForm";
import { mapBulkKeyEdit } from "./bulk-edit-form/mapper";
import { bulkKeyBudgetSchema, bulkKeyEditSchema } from "./bulk-edit-form/schema";

export type BulkKeyEditAction = "team" | "user" | "budget";
export type BulkKeyEditValues =
  | { team_id: string }
  | { user_id: string }
  | { max_budget: number; budget_duration: string };

interface VirtualKeysBulkEditDialogProps {
  action: BulkKeyEditAction;
  targets: readonly { token: string; key_alias?: string | null }[];
  pending: boolean;
  disabled: boolean;
  errorMessage?: string | null;
  failures?: readonly { key: string; failed_reason: string }[];
  onCancel: () => void;
  onConfirm: (values: BulkKeyEditValues) => void;
}

export function VirtualKeysBulkEditDialog({
  action,
  targets,
  pending,
  disabled,
  errorMessage,
  failures = [],
  onCancel,
  onConfirm,
}: VirtualKeysBulkEditDialogProps) {
  const { t } = useTranslation();
  const fieldId = useId();
  const { control } = useZodForm(bulkKeyEditSchema, {
    defaultValues: { teamId: null, userId: null, budgetInput: "", duration: "30d" },
  });
  const { field: team } = useController({ control, name: "teamId" });
  const { field: user } = useController({ control, name: "userId" });
  const { field: budgetField } = useController({ control, name: "budgetInput" });
  const { field: duration } = useController({ control, name: "duration" });
  const locked = pending || disabled;
  const parsedBudget = bulkKeyBudgetSchema.safeParse(budgetField.value);
  const budget = parsedBudget.success ? parsedBudget.data : null;
  const invalidBudget = budgetField.value !== "" && budget === null;
  const values = mapBulkKeyEdit(action, { teamId: team.value, userId: user.value, duration: duration.value }, budget);
  const canApply = !locked && values !== null && targets.length > 0;

  const cancel = () => {
    if (!pending) onCancel();
  };

  const submit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!canApply || !values) return;
    onConfirm(values);
  };

  return (
    <Dialog open onOpenChange={(open) => !open && cancel()}>
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto" showCloseButton={!pending}>
        <DialogHeader>
          <DialogTitle>{t(`apiKeys.virtualKeysTable.bulkEdit.${action}`)}</DialogTitle>
          <DialogDescription>{t(`apiKeys.virtualKeysTable.bulkEdit.descriptions.${action}`)}</DialogDescription>
        </DialogHeader>
        <div className="space-y-2 rounded-lg border border-border bg-muted/30 p-3">
          <p className="font-medium">{t("apiKeys.virtualKeysTable.bulkEdit.selected", { count: targets.length })}</p>
          <ul className="max-h-28 space-y-1 overflow-y-auto text-sm text-muted-foreground">
            {targets.map((target) => (
              <li key={target.token} className="break-all">
                {target.key_alias || target.token}
              </li>
            ))}
          </ul>
        </div>
        <form onSubmit={submit} className="space-y-6" aria-busy={pending}>
          <FieldSet disabled={locked} inert={locked}>
            {action === "team" && (
              <Field>
                <FieldLabel htmlFor={`${fieldId}-team`}>
                  {t("apiKeys.virtualKeysTable.bulkEdit.fields.team")}
                </FieldLabel>
                <TeamDropdown
                  id={`${fieldId}-team`}
                  value={team.value}
                  onChange={(value) => !locked && team.onChange(value)}
                  disabled={locked}
                />
              </Field>
            )}
            {action === "user" && (
              <Field>
                <FieldLabel htmlFor={`${fieldId}-user`}>
                  {t("apiKeys.virtualKeysTable.bulkEdit.fields.user")}
                </FieldLabel>
                <UserDropdown
                  id={`${fieldId}-user`}
                  value={user.value}
                  onChange={(value) => !locked && user.onChange(value)}
                  disabled={locked}
                />
              </Field>
            )}
            {action === "budget" && (
              <>
                <Field data-invalid={invalidBudget}>
                  <FieldLabel htmlFor={`${fieldId}-budget`}>
                    {t("apiKeys.virtualKeysTable.bulkEdit.fields.budget")}
                  </FieldLabel>
                  <Input
                    id={`${fieldId}-budget`}
                    type="number"
                    min={0}
                    step="any"
                    required
                    value={budgetField.value}
                    onChange={(event) => !locked && budgetField.onChange(event.target.value)}
                    disabled={locked}
                    aria-invalid={invalidBudget}
                    aria-describedby={invalidBudget ? `${fieldId}-budget-error` : undefined}
                  />
                  {invalidBudget && (
                    <FieldError id={`${fieldId}-budget-error`}>
                      {t("apiKeys.virtualKeysTable.bulkEdit.invalidBudget")}
                    </FieldError>
                  )}
                </Field>
                <Field data-invalid={duration.value === null}>
                  <FieldLabel htmlFor={`${fieldId}-duration`}>
                    {t("apiKeys.virtualKeysTable.bulkEdit.fields.resetPeriod")}
                  </FieldLabel>
                  <BudgetDurationDropdown
                    key={locked ? "locked" : "editable"}
                    id={`${fieldId}-duration`}
                    value={duration.value}
                    onChange={(value) => !locked && duration.onChange(value)}
                    placeholder={t("apiKeys.virtualKeysTable.bulkEdit.resetPeriodPlaceholder")}
                  />
                  {duration.value === null && (
                    <FieldError>{t("apiKeys.virtualKeysTable.bulkEdit.resetPeriodRequired")}</FieldError>
                  )}
                </Field>
              </>
            )}
          </FieldSet>
          {errorMessage && <FieldError>{errorMessage}</FieldError>}
          {failures.length > 0 && (
            <div role="alert" className="space-y-2 text-sm text-destructive">
              <p>{t("apiKeys.virtualKeysTable.bulkEdit.partialFailure", { count: failures.length })}</p>
              <ul className="max-h-40 space-y-3 overflow-y-auto">
                {failures.map((failure) => (
                  <li key={failure.key} className="space-y-1 break-words">
                    <p className="font-medium">
                      {targets.find((target) => target.token === failure.key)?.key_alias || failure.key}
                    </p>
                    <p className="whitespace-pre-wrap">{failure.failed_reason}</p>
                  </li>
                ))}
              </ul>
            </div>
          )}
          <DialogFooter>
            <Button type="button" variant="outline" onClick={cancel} disabled={pending}>
              {t("apiKeys.virtualKeysTable.bulkEdit.cancel")}
            </Button>
            <Button type="submit" disabled={!canApply}>
              {pending
                ? t("apiKeys.virtualKeysTable.bulkEdit.pending")
                : t("apiKeys.virtualKeysTable.bulkEdit.apply", { count: targets.length })}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
