"use client";

import { useLanguage } from "@/contexts/LanguageContext";
import { CircleHelp } from "lucide-react";
import React from "react";
import { z } from "zod";
import BudgetDurationDropdown from "@/components/common_components/budget_duration_dropdown";
import { FieldGroup } from "@/components/ui/field";
import { FormField } from "@/components/shared/form/FormField";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import NumericalInput from "@/components/shared/numerical_input";
import { Button } from "@/components/ui/button";
import { useZodForm } from "@/lib/forms/useZodForm";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { ModelAccessGroup } from "@/app/(dashboard)/hooks/modelAccessGroups/useModelAccessGroups";
import { SetModelAccessGroupBudgetParams } from "@/app/(dashboard)/hooks/modelAccessGroups/useSetModelAccessGroupBudget";
import { accessGroupBudgetFormValues, buildAccessGroupBudgetBody, hasAnyBudgetValue } from "./accessGroupBudgetPayload";

const labelWithHint = (label: React.ReactNode, hint: string): React.ReactNode => (
  <>
    {label}
    <Tooltip>
      <TooltipTrigger render={<CircleHelp className="size-3.5 shrink-0 cursor-help text-muted-foreground" />} />
      <TooltipContent>{hint}</TooltipContent>
    </Tooltip>
  </>
);

// 本地预算错误在展示时翻译，不因语言切换重建校验或重置预算输入。
const budgetSchema = z
  .object({
    max_budget: z.string().optional(),
    soft_budget: z.string().optional(),
    budget_duration: z.string().optional(),
  })
  .refine(hasAnyBudgetValue, {
    message: "Set at least one of max budget, soft budget or reset window",
    path: ["max_budget"],
  });

interface AccessGroupBudgetModalProps {
  accessGroup: ModelAccessGroup | null;
  isSaving: boolean;
  onCancel: () => void;
  onSubmit: (params: SetModelAccessGroupBudgetParams) => void;
}

const AccessGroupBudgetModal: React.FC<AccessGroupBudgetModalProps> = ({
  accessGroup,
  isSaving,
  onCancel,
  onSubmit,
}) => {
  const { t } = useLanguage();
  const budget = accessGroup?.budget ?? null;
  const form = useZodForm(budgetSchema, { values: accessGroupBudgetFormValues(budget) });

  return (
    <Dialog open={accessGroup !== null} onOpenChange={(open) => !open && onCancel()}>
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[560px]">
        <DialogHeader>
          <DialogTitle>
            {budget
              ? t("Edit budget for {group}", { group: accessGroup?.access_group ?? "" })
              : t("Set budget for {group}", { group: accessGroup?.access_group ?? "" })}
          </DialogTitle>
        </DialogHeader>
        <p className="text-sm text-muted-foreground">
          {t(
            "Every key granted this access group by name draws from this one budget. A key that reaches the group's models through a wildcard or",
          )}{" "}
          <code>all-proxy-models</code> {t("is not charged against it.")}
        </p>
        <form onSubmit={form.handleSubmit((values) => onSubmit(buildAccessGroupBudgetBody(values)))} noValidate>
          <TooltipProvider>
            <FieldGroup className="mt-4">
              <FormField
                control={form.control}
                name="max_budget"
                renderError={(error) => error.message && t(error.message)}
                label={labelWithHint(
                  t("Max Budget (USD)"),
                  t(
                    "Total the whole group may spend. Once its shared spend reaches this, every key that draws from the group is refused",
                  ),
                )}
              >
                {({ ref, value, ...field }) => <NumericalInput {...field} value={value ?? ""} step={0.01} />}
              </FormField>

              <FormField
                control={form.control}
                name="soft_budget"
                label={labelWithHint(
                  t("Soft Budget (USD)"),
                  t("Fires an alert when the group's spend reaches this. Requests keep succeeding"),
                )}
              >
                {({ ref, value, ...field }) => <NumericalInput {...field} value={value ?? ""} step={0.01} />}
              </FormField>

              <FormField
                control={form.control}
                name="budget_duration"
                label={labelWithHint(
                  t("Reset Budget"),
                  t("How often the group's spend resets. Leave empty for a budget that never resets"),
                )}
              >
                {({ id, value, onChange }) => (
                  <BudgetDurationDropdown
                    id={id}
                    value={value || null}
                    onChange={(next) => onChange(next ?? undefined)}
                    placeholder={t("Keep current")}
                  />
                )}
              </FormField>
            </FieldGroup>

            <p className="mt-3 text-xs text-muted-foreground">
              {t(
                "A field left blank keeps whatever the budget already has. Use Clear budget to remove the budget itself.",
              )}
            </p>

            <div className="mt-6 flex justify-end gap-2">
              <Button type="button" variant="outline" onClick={onCancel}>
                {t("Cancel")}
              </Button>
              <Button type="submit" disabled={isSaving}>
                {isSaving ? t("Saving...") : t("Save Budget")}
              </Button>
            </div>
          </TooltipProvider>
        </form>
      </DialogContent>
    </Dialog>
  );
};

export default AccessGroupBudgetModal;
