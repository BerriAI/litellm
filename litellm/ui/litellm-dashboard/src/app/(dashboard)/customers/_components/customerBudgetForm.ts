import type { components } from "@/lib/http/schema";
import { isValidBudgetDuration, normalizeBudgetDuration } from "@/utils/budgetUtils";

export type Budget = components["schemas"]["Budget"];
const budgetKeys = [
  "max_budget",
  "soft_budget",
  "budget_duration",
  "budget_limits",
  "tpm_limit",
  "rpm_limit",
  "allowed_models",
  "model_tpm_limit",
  "model_rpm_limit",
  "cap_policy",
] as const satisfies readonly (keyof Budget)[];

export const budgetSettings = (budget?: Budget | null): Budget =>
  Object.fromEntries(budgetKeys.filter((key) => budget?.[key] !== undefined).map((key) => [key, budget![key]]));
export interface BudgetFields {
  maxBudget: string;
  softBudget: string;
  budgetDuration: string;
  tpm: string;
  rpm: string;
  allowedModels: readonly string[];
}

type BudgetWindow = NonNullable<Budget["budget_limits"]>[number];

export interface BudgetEditorFields extends BudgetFields {
  budgetEnabled: boolean;
  budgetWindows: { budget: BudgetWindow; maxBudget: string }[];
}

export const BUDGET_EDITOR_FIELDS = [
  "budgetEnabled",
  "maxBudget",
  "softBudget",
  "budgetDuration",
  "tpm",
  "rpm",
  "allowedModels",
  "budgetWindows",
] as const satisfies readonly (keyof BudgetEditorFields)[];

const asText = (value: number | null | undefined): string => (value == null ? "" : String(value));

export const budgetFields = (budget: Budget | null | undefined): BudgetFields => ({
  maxBudget: asText(budget?.max_budget),
  softBudget: asText(budget?.soft_budget),
  budgetDuration: budget?.budget_duration ?? "",
  tpm: asText(budget?.tpm_limit),
  rpm: asText(budget?.rpm_limit),
  allowedModels: budget?.allowed_models ?? [],
});

const numericLimit = (raw: string, integer = false): number | null | undefined => {
  const value = raw.trim();
  if (!value) return null;
  if (!(integer ? /^\d+$/ : /^(?:\d+(?:\.\d*)?|\.\d+)$/).test(value)) return undefined;
  const number = Number(value);
  if (!Number.isFinite(number) || (integer && !Number.isSafeInteger(number))) return undefined;
  return number;
};

const validPrimaryPeriod = (fields: BudgetFields, original?: Budget | null): boolean => {
  const period = fields.budgetDuration.trim();
  return !period || period === original?.budget_duration?.trim() || isValidBudgetDuration(period);
};

export const parseBudget = (fields: BudgetFields, original?: Budget | null): Budget | null => {
  if (!validPrimaryPeriod(fields, original)) return null;
  const max = numericLimit(fields.maxBudget);
  const soft = numericLimit(fields.softBudget);
  const tpm = numericLimit(fields.tpm, true);
  const rpm = numericLimit(fields.rpm, true);
  if ([max, soft, tpm, rpm].includes(undefined)) return null;
  const models = fields.allowedModels.map((model) => model.trim()).filter(Boolean);
  return {
    max_budget: max,
    soft_budget: soft,
    tpm_limit: tpm,
    rpm_limit: rpm,
    budget_duration: fields.budgetDuration.trim() || null,
    allowed_models: models.length ? [...new Set(models)] : null,
  };
};

export const budgetEditorFields = (budget?: Budget | null): BudgetEditorFields => ({
  budgetEnabled: budget != null,
  ...budgetFields(budget),
  budgetWindows: (budget?.budget_limits ?? []).map((window) => ({
    budget: window,
    maxBudget: String(window.max_budget),
  })),
});

export const budgetEditorValue = (fields: BudgetEditorFields, original?: Budget | null) => {
  if (!fields.budgetEnabled) return { ok: true, value: null } as const;
  if (
    !validPrimaryPeriod(fields, original) ||
    fields.budgetWindows.some(({ budget }) => !isValidBudgetDuration(budget.budget_duration))
  ) {
    return { ok: false, error: "customerPortal.budgetEditor.invalidPeriod" } as const;
  }
  const parsed = parseBudget(fields, original);
  if (!parsed) return { ok: false, error: "customerPortal.hierarchy.invalidNumber" } as const;
  const windows: BudgetWindow[] = [];
  for (const window of fields.budgetWindows) {
    const maxBudget = numericLimit(window.maxBudget);
    if (maxBudget == null) return { ok: false, error: "customerPortal.hierarchy.invalidNumber" } as const;
    windows.push({ ...window.budget, max_budget: maxBudget });
  }
  const periods = [
    ...(parsed.max_budget == null ? [] : [fields.budgetDuration]),
    ...windows.map((window) => window.budget_duration),
  ].map(normalizeBudgetDuration);
  const originalPeriods = [
    ...(original?.max_budget == null ? [] : [original.budget_duration ?? ""]),
    ...(original?.budget_limits ?? []).map((window) => window.budget_duration),
  ].map(normalizeBudgetDuration);
  if (
    periods.some(
      (period) =>
        periods.filter((value) => value === period).length >
        Math.max(1, originalPeriods.filter((value) => value === period).length),
    )
  )
    return { ok: false, error: "customerPortal.budgetEditor.duplicatePeriod" } as const;
  return {
    ok: true,
    value: { ...budgetSettings(original), ...parsed, budget_limits: windows.length ? windows : null },
  } as const;
};

export const compactBudgetEditorFields = (budget?: Budget | null): BudgetEditorFields => ({
  ...budgetEditorFields(budget),
  budgetDuration: budget == null ? "1mo" : budget.budget_duration ?? "",
});

export const compactBudgetEditorValue = (fields: BudgetEditorFields, original?: Budget | null) =>
  budgetEditorValue(
    { ...fields, budgetWindows: fields.budgetWindows.filter((window) => window.maxBudget.trim() !== "") },
    original,
  );

export const permanentBudgetEditorValue = (fields: BudgetEditorFields, original?: Budget | null) => {
  const parsed = compactBudgetEditorValue(fields, original);
  if (!parsed.ok || parsed.value === null) return parsed;
  const initial = compactBudgetEditorValue({ ...compactBudgetEditorFields(original), budgetEnabled: true }, original);
  if (!initial.ok || !initial.value) return parsed;
  return { ok: true, value: budgetChanges(parsed.value, initial.value) } as const;
};

export function budgetChanges(value: Budget, initial?: Budget | null): Budget {
  const changed = Object.fromEntries(
    Object.entries(value).filter(([key, next]) => {
      const previous = initial?.[key as keyof Budget];
      return !(next == null && previous == null) && JSON.stringify(next) !== JSON.stringify(previous);
    }),
  ) as Budget;
  return {
    ...changed,
    ...(changed.max_budget != null ? { budget_duration: value.budget_duration } : {}),
  };
}

export const budgetPolicyChanges = (policy: Budget | null | undefined, original: Budget | null | undefined): Budget =>
  budgetChanges(
    {
      model_tpm_limit: policy?.model_tpm_limit ?? null,
      model_rpm_limit: policy?.model_rpm_limit ?? null,
      cap_policy: policy?.cap_policy ?? null,
    },
    original,
  );

interface BudgetSelectionOwner {
  budget_id?: string | null;
  budget?: Budget | null;
}

type BudgetPreset = Budget & { budget_id: string };

export const budgetSelectionFields = (
  entity: BudgetSelectionOwner | undefined,
  presets: readonly BudgetPreset[],
  defaultBudgetId?: string | null,
) => {
  const selectedId = entity ? entity.budget_id : defaultBudgetId;
  const customSource = entity?.budget ? "custom" : "inherit";
  return {
    budget: budgetEditorFields(entity?.budget),
    policy: entity?.budget,
    budgetSource: presets.some((preset) => preset.budget_id === selectedId) ? `profile:${selectedId}` : customSource,
  };
};

type BudgetSelectionFields = ReturnType<typeof budgetSelectionFields>;

export const budgetSelectionValue = (fields: BudgetSelectionFields, original?: Budget | null) => {
  if (fields.budgetSource !== "custom") {
    return {
      ok: true,
      value: { budget_id: fields.budgetSource.startsWith("profile:") ? fields.budgetSource.slice(8) : null },
    } as const;
  }
  const parsed = permanentBudgetEditorValue(fields.budget, original);
  if (!parsed.ok) return parsed;
  return { ok: true, value: { budget: { ...parsed.value, ...budgetPolicyChanges(fields.policy, original) } } } as const;
};

export const changeBudgetSelection = (
  fields: BudgetSelectionFields,
  source: string,
  presets: readonly BudgetPreset[],
): BudgetSelectionFields => {
  const selected =
    source === "custom" ? presets.find((preset) => `profile:${preset.budget_id}` === fields.budgetSource) : null;
  return {
    budgetSource: source,
    budget: { ...(selected ? budgetEditorFields(selected) : fields.budget), budgetEnabled: source === "custom" },
    policy: selected ?? fields.policy,
  };
};

interface BudgetAssignment {
  budget?: Budget | null;
  budget_override?: Budget | null;
  budget_override_expires_at?: string | null;
}

export const hasLegacyCustomerBudget = (row?: BudgetAssignment | null) =>
  row?.budget_override != null && row.budget_override_expires_at == null;

export function customerBaseBudget(row: BudgetAssignment | null, inherited?: Budget | null): Budget | null {
  const base = row?.budget ?? inherited ?? null;
  if (!hasLegacyCustomerBudget(row)) return base;
  const override = row!.budget_override!;
  const windows = new Map(
    (override.max_budget != null ? [] : base?.budget_limits ?? []).map((window) => [window.budget_duration, window]),
  );
  const merged = new Map([
    ...windows,
    ...(override.budget_limits ?? []).map((window) => [window.budget_duration, window] as const),
  ]);
  return {
    ...base,
    ...Object.fromEntries(Object.entries(override).filter(([, value]) => value != null)),
    allowed_models: override.allowed_models?.length ? override.allowed_models : base?.allowed_models,
    budget_limits: merged.size ? [...merged.values()] : null,
  };
}

export function customerBudgetPatch(fields: BudgetEditorFields, original: Budget | null, row: BudgetAssignment | null) {
  const parsed = hasLegacyCustomerBudget(row)
    ? compactBudgetEditorValue(fields, original)
    : permanentBudgetEditorValue(fields, original);
  return parsed.ok
    ? ({
        ok: true,
        value: { budget: parsed.value, ...(hasLegacyCustomerBudget(row) ? { budget_override: null } : {}) },
      } as const)
    : parsed;
}
