import type { components } from "@/lib/http/schema";
export type BudgetGroups = components["schemas"]["BudgetGroups"];
import {
  customerBudgetGroupValue,
  defaultCustomerCapPolicy,
  validateCustomerBudgetGroups,
  type CustomerBudgetGroupFields,
} from "./customerOrganizationForm";
import { budgetChanges, budgetEditorFields } from "./customerBudgetForm";

export const customerBudgetGroupRows = (organization: BudgetGroups): CustomerBudgetGroupFields[] =>
  organization.default_budgets.map((group) => ({
    ...budgetEditorFields(group),
    id: group.budget_id,
    name: group.name,
    capPolicy: group.cap_policy ?? defaultCustomerCapPolicy,
  }));

export function planCustomerBudgetGroups(
  organization: BudgetGroups,
  groups: CustomerBudgetGroupFields[],
  defaultGroupId: string,
  memberGroups = true,
) {
  const originals = customerBudgetGroupRows(organization);
  const validation = validateCustomerBudgetGroups(
    { groups, defaultGroupId },
    organization.default_budgets,
    memberGroups,
  );
  if (!validation.ok) return validation;
  const default_budgets = groups.map((group) => {
    const original = originals.find((row) => row.id === group.id);
    const originalBudget = organization.default_budgets.find((budget) => budget.budget_id === group.id);
    return {
      ...(original ? { budget_id: original.id } : {}),
      name: group.name.trim(),
      ...(memberGroups && group.capPolicy !== original?.capPolicy ? { cap_policy: group.capPolicy } : {}),
      ...budgetChanges(
        customerBudgetGroupValue(group, originalBudget)!,
        original ? customerBudgetGroupValue(original, originalBudget) : null,
      ),
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
        Object.keys(group).some((key) => key !== "budget_id" && key !== "name"),
    );
  return { ok: true, body, changed } as const;
}
