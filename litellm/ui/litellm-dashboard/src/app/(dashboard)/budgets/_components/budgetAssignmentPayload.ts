import type {
  BudgetAssignment,
  BudgetAssignmentPatch,
  BudgetDuration,
  CapPolicy,
} from "../../hooks/budgets/budgetProfileTypes";
import { profileLimits } from "./budgetProfilePayload";

export interface AssignmentValues {
  profile: string;
  monthly: number | null;
  annual: number | null;
  policy: CapPolicy | "inherit";
  increase: number | null;
  increaseExpiry: string;
  increaseDuration: BudgetDuration;
  overrideExpiry: string;
  clearOverride: boolean;
}

export const localDateTime = (value: string | null | undefined): string => {
  if (!value) return "";
  const date = new Date(value);
  return new Date(date.getTime() - date.getTimezoneOffset() * 60_000).toISOString().slice(0, 16);
};

export const assignmentValues = (assignment: BudgetAssignment): AssignmentValues => ({
  profile: assignment.budget_id ?? "",
  monthly:
    assignment.budget_override?.budget_limits?.find((limit) => limit.budget_duration === "1mo")?.max_budget ?? null,
  annual:
    assignment.budget_override?.budget_limits?.find((limit) => limit.budget_duration === "12mo")?.max_budget ?? null,
  policy: assignment.budget_override?.cap_policy ?? "inherit",
  increase: assignment.budget_override?.temp_budget_increase ?? null,
  increaseExpiry: localDateTime(assignment.budget_override?.temp_budget_expiry),
  increaseDuration: assignment.budget_override?.temp_budget_duration ?? "1mo",
  overrideExpiry: localDateTime(assignment.budget_override_expires_at),
  clearOverride: false,
});

type PayloadResult =
  | { ok: true; payload: BudgetAssignmentPatch }
  | { ok: false; field: "increaseExpiry" | "overrideExpiry"; error: "futureExpiryRequired" };

const INCREASE_FIELDS = ["increase", "increaseExpiry", "increaseDuration"] as const;
const increaseChanged = (values: AssignmentValues, previous: AssignmentValues) =>
  INCREASE_FIELDS.some((key) => values[key] !== previous[key]);

function validateExpiries(
  values: AssignmentValues,
  previous: AssignmentValues,
  now: number,
): Extract<PayloadResult, { ok: false }> | null {
  const isFuture = (value: string) => Date.parse(value) > now;
  if (increaseChanged(values, previous) && values.increase !== null && !isFuture(values.increaseExpiry)) {
    return { ok: false, field: "increaseExpiry", error: "futureExpiryRequired" };
  }
  if (values.overrideExpiry !== previous.overrideExpiry && values.overrideExpiry && !isFuture(values.overrideExpiry)) {
    return { ok: false, field: "overrideExpiry", error: "futureExpiryRequired" };
  }
  return null;
}

export function assignmentPayload(
  values: AssignmentValues,
  baseline: BudgetAssignment,
  now = Date.now(),
): PayloadResult {
  const previous = assignmentValues(baseline);
  const changed = (key: keyof AssignmentValues) => values[key] !== previous[key];
  const selectedProfile = changed("profile") ? { budget_id: values.profile || null } : {};
  if (values.clearOverride) {
    return {
      ok: true,
      payload: {
        revision: baseline.revision,
        ...selectedProfile,
        budget_override: null,
        budget_override_expires_at: null,
      },
    };
  }
  const invalid = validateExpiries(values, previous, now);
  if (invalid) return invalid;
  const override = {
    ...(changed("monthly") || changed("annual") ? { budget_limits: profileLimits(values.monthly, values.annual) } : {}),
    ...(changed("policy") ? { cap_policy: values.policy === "inherit" ? null : values.policy } : {}),
    ...(increaseChanged(values, previous)
      ? {
          temp_budget_increase: values.increase,
          temp_budget_expiry: values.increase === null ? null : new Date(values.increaseExpiry).toISOString(),
          temp_budget_duration: values.increase === null ? null : values.increaseDuration,
        }
      : {}),
  };
  return {
    ok: true,
    payload: {
      revision: baseline.revision,
      ...selectedProfile,
      ...(Object.keys(override).length ? { budget_override: override } : {}),
      ...(changed("overrideExpiry")
        ? { budget_override_expires_at: values.overrideExpiry ? new Date(values.overrideExpiry).toISOString() : null }
        : {}),
    },
  };
}
