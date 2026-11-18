import type { components } from "@/lib/http/schema";
import {
  budgetFields,
  parseBudget,
  type CustomerHierarchyFields,
} from "../../customers/_components/customerHierarchyForms";

export type App = components["schemas"]["App"];
export type Budget = components["schemas"]["Budget"];
export type AppOrganization = components["schemas"]["AppOrganization"];
export type AppMember = components["schemas"]["AppMember"];

export const appBudgetFields = (budget?: Budget | null): CustomerHierarchyFields => ({
  alias: "",
  templateId: "",
  overrideExpiry: "",
  budgetEnabled: budget != null,
  ...budgetFields(budget),
});

export const appBudgetValue = (fields: CustomerHierarchyFields, original?: Budget | null) => {
  if (!fields.budgetEnabled) return { ok: true, value: null } as const;
  const parsed = parseBudget(fields);
  return parsed
    ? ({ ok: true, value: { ...original, ...parsed } } as const)
    : ({ ok: false, error: "customerPortal.hierarchy.invalidNumber" } as const);
};

export const appTierBudgetValue = (fields: CustomerHierarchyFields, original?: Budget | null) => {
  const budget = appBudgetValue(fields, original);
  if (!budget.ok || budget.value !== null) return budget;
  return {
    ok: true,
    value: {
      max_budget: null,
      soft_budget: null,
      tpm_limit: null,
      rpm_limit: null,
      budget_duration: null,
      budget_limits: null,
      allowed_models: null,
      model_tpm_limit: null,
      model_rpm_limit: null,
      cap_policy: null,
    },
  } as const;
};
