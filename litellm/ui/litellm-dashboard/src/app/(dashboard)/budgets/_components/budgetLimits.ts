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

function singlePeriodPayload(
  scalar: BudgetWriteRequest,
  values: BudgetFormValues,
  dirty: DirtyFields | undefined,
  existingLimits: BudgetWriteRequest["budget_limits"],
): BudgetWriteRequest {
  if (!existingLimits?.length) return scalar;
  if (dirty?.budget_duration) {
    return { ...scalar, max_budget: values.max_budget, budget_duration: values.budget_duration, budget_limits: null };
  }
  if (!dirty?.max_budget) return scalar;
  const { max_budget: _max, ...other } = scalar;
  const duration = values.budget_duration === "30d" ? "1mo" : values.budget_duration;
  const amount = values.max_budget;
  const limits = existingLimits.flatMap((limit) => {
    if (limit.budget_duration !== duration) return [limit];
    return amount == null ? [] : [{ ...limit, max_budget: amount }];
  });
  return { ...other, budget_limits: limits.length ? limits : null };
}

export function budgetFormPayload(
  values: BudgetFormValues,
  dirty?: DirtyFields,
  existingLimits?: BudgetWriteRequest["budget_limits"],
): BudgetWriteRequest {
  const selected = dirty ? pickDirty(values, dirty) : values;
  const { annual_budget: _annual, ...fields } = selected;
  const scalar = { ...fields, budget_id: values.budget_id };
  if (values.budget_duration !== COMBINED_BUDGET_DURATION) {
    return applyBudgetPrecision(singlePeriodPayload(scalar, values, dirty, existingLimits));
  }
  const monthly = existingLimits?.find((limit) => limit.budget_duration === "1mo");
  const annual = existingLimits?.find((limit) => limit.budget_duration === "12mo");
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
      calendarLimit(monthly, values, dirty, "max_budget"),
      calendarLimit(annual, values, dirty, "annual_budget"),
    ],
  };
  return applyBudgetPrecision(payload);
}
