import type { FormState } from "react-hook-form";
import type { components } from "@/lib/http/schema";
import { pickDirty } from "@/lib/forms/pickDirty";
import type { BudgetFormValues } from "./BudgetFormFields";
import { applyBudgetPrecision } from "./budgetPrecision";

export const COMBINED_BUDGET_DURATION = "monthly-and-annually";

type DirtyFields = FormState<BudgetFormValues>["dirtyFields"];
type BudgetWriteRequest = components["schemas"]["BudgetWriteRequest"];
type BudgetLimit = NonNullable<BudgetWriteRequest["budget_limits"]>[number];

function calendarLimit(
  existing: BudgetLimit | undefined,
  values: BudgetFormValues,
  dirty: DirtyFields | undefined,
  field: "max_budget" | "annual_budget",
): BudgetLimit {
  const changed = !dirty || dirty.budget_duration || dirty[field];
  if (!changed && existing) return existing;
  const amount = values[field];
  if (amount == null) throw new Error("Monthly and annual limits are required for a combined budget");
  return {
    ...existing,
    budget_duration: field === "max_budget" ? "1mo" : "12mo",
    max_budget: amount,
  };
}

export function budgetFormPayload(
  values: BudgetFormValues,
  dirty?: DirtyFields,
  existingLimits?: BudgetWriteRequest["budget_limits"],
): BudgetWriteRequest {
  const selected = dirty ? pickDirty(values, dirty) : values;
  const { annual_budget: _annual, ...scalar } = selected;
  const monthly = existingLimits?.find((limit) => limit.budget_duration === "1mo");
  const annual = existingLimits?.find((limit) => limit.budget_duration === "12mo");
  const otherLimits = existingLimits?.filter((limit) => !["1mo", "12mo"].includes(limit.budget_duration)) ?? [];
  if (values.budget_duration !== COMBINED_BUDGET_DURATION) {
    const remaining = otherLimits.length ? otherLimits : null;
    const single = {
      max_budget: values.max_budget,
      budget_duration: values.budget_duration,
      budget_limits: remaining,
    };
    return applyBudgetPrecision({
      ...scalar,
      budget_id: values.budget_id,
      ...(monthly && annual && dirty?.budget_duration ? single : {}),
    });
  }
  const { max_budget: _max, budget_duration: _duration, ...other } = scalar;
  const capsChanged = !dirty || Boolean(dirty.max_budget || dirty.annual_budget || dirty.budget_duration);
  if (!capsChanged) {
    return applyBudgetPrecision({ ...other, budget_id: values.budget_id });
  }
  const payload = {
    ...other,
    budget_id: values.budget_id,
    ...(dirty?.budget_duration ? { max_budget: null, budget_duration: null } : {}),
    budget_limits: [
      ...otherLimits,
      calendarLimit(monthly, values, dirty, "max_budget"),
      calendarLimit(annual, values, dirty, "annual_budget"),
    ],
  };
  return applyBudgetPrecision(payload);
}
