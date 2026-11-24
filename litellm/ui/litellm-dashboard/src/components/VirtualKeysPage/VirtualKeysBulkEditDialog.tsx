"use client";

import { useId, useState, type FormEvent } from "react";
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

const parseBudget = (input: string): number | null => {
  if (input.trim() === "") return null;
  const amount = Number(input);
  return Number.isFinite(amount) && amount >= 0 ? amount : null;
};

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
  const [teamId, setTeamId] = useState<string | null>(null);
  const [userId, setUserId] = useState<string | null>(null);
  const [budgetInput, setBudgetInput] = useState("");
  const [duration, setDuration] = useState<string | null>("30d");
  const locked = pending || disabled;
  const budget = parseBudget(budgetInput);
  const invalidBudget = budgetInput !== "" && budget === null;
  const valuesByAction: Record<BulkKeyEditAction, BulkKeyEditValues | null> = {
    team: teamId ? { team_id: teamId } : null,
    user: userId ? { user_id: userId } : null,
    budget: budget !== null && duration ? { max_budget: budget, budget_duration: duration } : null,
  };
  const values = valuesByAction[action];
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
                  value={teamId}
                  onChange={(value) => !locked && setTeamId(value)}
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
                  value={userId}
                  onChange={(value) => !locked && setUserId(value)}
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
                    value={budgetInput}
                    onChange={(event) => !locked && setBudgetInput(event.target.value)}
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
                <Field data-invalid={duration === null}>
                  <FieldLabel htmlFor={`${fieldId}-duration`}>
                    {t("apiKeys.virtualKeysTable.bulkEdit.fields.resetPeriod")}
                  </FieldLabel>
                  <BudgetDurationDropdown
                    key={locked ? "locked" : "editable"}
                    id={`${fieldId}-duration`}
                    value={duration}
                    onChange={(value) => !locked && setDuration(value)}
                    placeholder={t("apiKeys.virtualKeysTable.bulkEdit.resetPeriodPlaceholder")}
                  />
                  {duration === null && (
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
