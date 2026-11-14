import type { components } from "@/lib/http/schema";
import type { Organization } from "./customerHierarchyForms";
import {
  defaultCustomerCapPolicy,
  limit,
  validateCustomerBudgetGroups,
  type CustomerBudgetGroupFields,
} from "./customerOrganizationForm";

export const customerBudgetGroupRows = (organization: Organization): CustomerBudgetGroupFields[] =>
  organization.default_budgets.map((group) => ({
    id: group.budget_id,
    name: group.name,
    capPolicy: group.cap_policy ?? defaultCustomerCapPolicy,
    monthlyBudget: String(group.budget_limits?.find((entry) => entry.budget_duration === "1mo")?.max_budget ?? ""),
    yearlyBudget: String(group.budget_limits?.find((entry) => entry.budget_duration === "12mo")?.max_budget ?? ""),
  }));

export function planCustomerBudgetGroups(
  organization: Organization,
  groups: CustomerBudgetGroupFields[],
  defaultGroupId: string,
) {
  const originals = customerBudgetGroupRows(organization);
  const validation = validateCustomerBudgetGroups(
    { groups, defaultGroupId },
    originals.map((group) => group.id),
  );
  if (!validation.ok) return validation;
  const default_budgets = groups.map((group) => {
    const original = originals.find((row) => row.id === group.id);
    const limits = [...limit(group.monthlyBudget, "1mo"), ...limit(group.yearlyBudget, "12mo")];
    const previous = original
      ? [...limit(original.monthlyBudget, "1mo"), ...limit(original.yearlyBudget, "12mo")]
      : null;
    return {
      ...(original ? { budget_id: original.id } : {}),
      name: group.name.trim(),
      ...(group.capPolicy !== original?.capPolicy ? { cap_policy: group.capPolicy } : {}),
      ...(JSON.stringify(limits) !== JSON.stringify(previous) ? { budget_limits: limits } : {}),
    };
  });
  const body: components["schemas"]["BudgetGroupsUpdate"] = {
    expected_revision: organization.budget_groups_revision,
    default_budgets,
    default_budget_template: groups.find((group) => group.id === defaultGroupId)?.name.trim() ?? null,
  };
  const changed =
    defaultGroupId !== (organization.default_budget_template_id ?? "") ||
    groups.length !== originals.length ||
    default_budgets.some(
      (group) =>
        group.name !== originals.find((original) => original.id === group.budget_id)?.name ||
        group.cap_policy !== undefined ||
        group.budget_limits !== undefined,
    );
  return { ok: true, body, changed } as const;
}
