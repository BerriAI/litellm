"use client";

import { CircleHelp } from "lucide-react";
import React from "react";
import { z } from "zod/v4";
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
  const budget = accessGroup?.budget ?? null;
  const form = useZodForm(budgetSchema, { values: accessGroupBudgetFormValues(budget) });

  return (
    <Dialog open={accessGroup !== null} onOpenChange={(open) => !open && onCancel()}>
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[560px]">
        <DialogHeader>
          <DialogTitle>
            {budget ? "Edit" : "Set"} budget for &quot;{accessGroup?.access_group}&quot;
          </DialogTitle>
        </DialogHeader>
        <p className="text-sm text-muted-foreground">
          Every key granted this access group by name draws from this one budget. A key that reaches the group&apos;s
          models through a wildcard or <code>all-proxy-models</code> is not charged against it.
        </p>
        <form onSubmit={form.handleSubmit((values) => onSubmit(buildAccessGroupBudgetBody(values)))} noValidate>
          <TooltipProvider>
            <FieldGroup className="mt-4">
              <FormField
                control={form.control}
                name="max_budget"
                label={labelWithHint(
                  "Max Budget (USD)",
                  "Total the whole group may spend. Once its shared spend reaches this, every key that draws from the group is refused",
                )}
              >
                {({ ref, value, ...field }) => <NumericalInput {...field} value={value ?? ""} step={0.01} />}
              </FormField>

              <FormField
                control={form.control}
                name="soft_budget"
                label={labelWithHint(
                  "Soft Budget (USD)",
                  "Fires an alert when the group's spend reaches this. Requests keep succeeding",
                )}
              >
                {({ ref, value, ...field }) => <NumericalInput {...field} value={value ?? ""} step={0.01} />}
              </FormField>

              <FormField
                control={form.control}
                name="budget_duration"
                label={labelWithHint(
                  "Reset Budget",
                  "How often the group's spend resets. Leave empty for a budget that never resets",
                )}
              >
                {({ id, value, onChange }) => (
                  <BudgetDurationDropdown id={id} value={value || null} onChange={onChange} />
                )}
              </FormField>
            </FieldGroup>

            <p className="mt-3 text-xs text-muted-foreground">
              A field left blank keeps whatever the budget already has. Use Clear budget to remove the budget itself.
            </p>

            <div className="mt-6 flex justify-end gap-2">
              <Button type="button" variant="outline" onClick={onCancel}>
                Cancel
              </Button>
              <Button type="submit" disabled={isSaving}>
                {isSaving ? "Saving..." : "Save Budget"}
              </Button>
            </div>
          </TooltipProvider>
        </form>
      </DialogContent>
    </Dialog>
  );
};

export default AccessGroupBudgetModal;
