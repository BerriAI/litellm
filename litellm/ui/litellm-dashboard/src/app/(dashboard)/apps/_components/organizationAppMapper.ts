import {
  budgetPolicyChanges,
  compactBudgetEditorFields,
  customerBudgetPatch,
  hasLegacyCustomerBudget,
} from "../../customers/_components/customerBudgetForm";
import type { AppOrganization, Budget } from "./appForms";
import type { OrganizationAppDraft } from "./organizationAppSchema";

export function organizationBudgetSource(row: AppOrganization | null): "inherit" | "profile" | "custom" {
  if (row?.budget || hasLegacyCustomerBudget(row)) return "custom";
  if (row?.budget_id || row?.tier_budget_id) return "profile";
  return "inherit";
}

export const organizationAppDraft = (row: AppOrganization | null, original: Budget | null): OrganizationAppDraft => {
  const source = organizationBudgetSource(row);
  return {
    source: source === "profile" ? `profile:${row?.budget_id ?? row?.tier_budget_id}` : source,
    budget: compactBudgetEditorFields(original),
    policy: original,
  };
};

export const organizationAppBody = (
  { source, budget, policy }: OrganizationAppDraft,
  original: Budget | null,
  row: AppOrganization | null,
) => {
  const parsed = customerBudgetPatch({ ...budget, budgetEnabled: source === "custom" }, original, row);
  if (!parsed.ok) return parsed;
  const policyChanges = budgetPolicyChanges(policy, original);
  return {
    ok: true,
    value: {
      ...(source === "custom"
        ? { ...parsed.value, budget: { ...parsed.value.budget, ...policyChanges } }
        : { budget_id: source.startsWith("profile:") ? source.slice(8) : null }),
      ...(hasLegacyCustomerBudget(row) ? { budget_override: null } : {}),
    },
  } as const;
};
