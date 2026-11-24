import type { FormState } from "react-hook-form";
import type { components } from "@/lib/http/schema";
import { pickDirty } from "@/lib/forms/pickDirty";
import type { BudgetFormValues } from "./BudgetFormFields";
import { applyBudgetPrecision } from "./budgetPrecision";
import { budgetFormWindows, CALENDAR_BUDGET_DURATIONS, mergeBudgetLimits } from "@/lib/forms/budgetWindows";

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
    const shownDurations = budgetFormWindows(existingLimits).map((window) => window.budget_duration);
    const remaining = mergeBudgetLimits(existingLimits, [], shownDurations);
    if (values.additional_max_budget !== undefined) {
      const { max_budget: _max, budget_duration: _duration, ...other } = scalar;
      const duration = values.budget_duration === "30d" ? "1mo" : values.budget_duration;
      const existing = existingLimits.find((limit) => limit.budget_duration === duration);
      const limits = mergeBudgetLimits(
        existingLimits,
        values.max_budget == null || duration == null
          ? []
          : [{ ...existing, budget_duration: duration, max_budget: values.max_budget }],
        shownDurations,
      );
      return {
        ...other,
        budget_limits: limits.length ? limits : null,
      };
    }
    return {
      ...scalar,
      max_budget: values.max_budget,
      budget_duration: values.budget_duration,
      budget_limits: remaining.length ? remaining : null,
    };
  }
  if (!dirty?.max_budget) return scalar;
  const { max_budget: _max, ...other } = scalar;
  const duration =
    values.budget_duration === "30d" && existingLimits.some((limit) => limit.budget_duration === "1mo")
      ? "1mo"
      : values.budget_duration;
  const amount = values.max_budget;
  const limits = existingLimits.flatMap((limit) => {
    if (limit.budget_duration !== duration) return [limit];
    return amount == null ? [] : [{ ...limit, max_budget: amount }];
  });
  const merged = mergeBudgetLimits(existingLimits, limits, [duration ?? "", ...CALENDAR_BUDGET_DURATIONS]);
  return { ...other, budget_limits: merged.length ? merged : null };
}

function combinedPeriodPayload(
  scalar: BudgetWriteRequest,
  values: BudgetFormValues,
  dirty: DirtyFields | undefined,
  existingLimits: BudgetWriteRequest["budget_limits"],
): BudgetWriteRequest {
  const monthly = existingLimits?.find((limit) => limit.budget_duration === "1mo");
  const annual = existingLimits?.find((limit) => limit.budget_duration === "12mo");
  const { max_budget: _max, budget_duration: _duration, ...other } = scalar;
  const capsChanged = !dirty || Boolean(dirty.max_budget || dirty.annual_budget || dirty.budget_duration);
  if (!capsChanged) {
    return other;
  }
  return {
    ...other,
    ...(dirty?.budget_duration && values.additional_max_budget === undefined
      ? { max_budget: null, budget_duration: null }
      : {}),
    budget_limits: mergeBudgetLimits(
      existingLimits,
      [calendarLimit(monthly, values, dirty, "max_budget"), calendarLimit(annual, values, dirty, "annual_budget")],
      budgetFormWindows(existingLimits).map((window) => window.budget_duration),
    ),
  };
}

export function budgetFormPayload(
  values: BudgetFormValues,
  dirty?: DirtyFields,
  existingLimits?: BudgetWriteRequest["budget_limits"],
): BudgetWriteRequest {
  const selected = dirty ? pickDirty(values, dirty) : values;
  const { annual_budget: _annual, additional_max_budget, additional_budget_duration, ...fields } = selected;
  const scalar = { ...fields, budget_id: values.budget_id };
  const payload =
    values.budget_duration === COMBINED_BUDGET_DURATION
      ? combinedPeriodPayload(scalar, values, dirty, existingLimits)
      : singlePeriodPayload(scalar, values, dirty, existingLimits);
  return applyBudgetPrecision({
    ...payload,
    ...(additional_max_budget !== undefined ? { max_budget: additional_max_budget } : {}),
    ...(additional_budget_duration !== undefined ? { budget_duration: additional_budget_duration } : {}),
  });
}
