import type { components } from "@/lib/http/schema";
import { budgetFields, parseBudget, type BudgetFields } from "../../customers/_components/customerHierarchyForms";

export type App = components["schemas"]["App"];
export type Budget = components["schemas"]["Budget"];
export type AppOrganization = components["schemas"]["AppOrganization"];
export type AppMember = components["schemas"]["AppMember"];
export type AppResourceType = components["schemas"]["AppResourceType"];
export type AppResource = components["schemas"]["AppResource"];
type BudgetWindow = NonNullable<Budget["budget_limits"]>[number];

export interface AppBudgetFields extends BudgetFields {
  budgetEnabled: boolean;
  budgetWindows: { budget: BudgetWindow; maxBudget: string }[];
}

export const APP_RESOURCE_LABELS = {
  team: "apps.resourceTypes.team",
  project: "apps.resourceTypes.project",
  virtual_key: "apps.resourceTypes.virtual_key",
} as const satisfies Record<AppResourceType, string>;

export const appBudgetFields = (budget?: Budget | null): AppBudgetFields => ({
  budgetEnabled: budget != null,
  ...budgetFields(budget),
  budgetWindows: (budget?.budget_limits ?? []).map((window) => ({
    budget: window,
    maxBudget: String(window.max_budget),
  })),
});

export const appBudgetValue = (fields: AppBudgetFields, original?: Budget | null) => {
  if (!fields.budgetEnabled) return { ok: true, value: null } as const;
  const parsed = parseBudget(fields);
  if (!parsed) return { ok: false, error: "customerPortal.hierarchy.invalidNumber" } as const;
  const windows: BudgetWindow[] = [];
  for (const window of fields.budgetWindows) {
    const parsedWindow = parseBudget({ ...fields, maxBudget: window.maxBudget });
    if (parsedWindow?.max_budget == null)
      return { ok: false, error: "customerPortal.hierarchy.invalidNumber" } as const;
    windows.push({ ...window.budget, max_budget: parsedWindow.max_budget });
  }
  return { ok: true, value: { ...original, ...parsed, budget_limits: windows.length ? windows : null } } as const;
};

const MAX_SURCHARGE_PERCENT = 1000;

export const appSurchargeField = (margin?: number | null): string =>
  margin === null || margin === undefined ? "" : String(Number((margin * 100).toFixed(6)));

export const appSurchargeValue = (text: string): { ok: true; value: number | null } | { ok: false } => {
  const trimmed = text.trim();
  if (trimmed === "") return { ok: true, value: null };
  const percent = Number(trimmed);
  if (!Number.isFinite(percent) || percent < 0 || percent > MAX_SURCHARGE_PERCENT) return { ok: false };
  return { ok: true, value: Number((percent / 100).toFixed(8)) };
};
