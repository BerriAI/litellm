import type { components } from "@/lib/http/schema";
import type { Organization } from "./customerHierarchyForms";
import { limit, validateCustomerBudgetGroups, type CustomerBudgetGroupFields } from "./customerOrganizationForm";

type Template = components["schemas"]["BudgetTemplate"];
type Patch = components["schemas"]["TemplatePatch"];
export const customerBudgetGroupRows = (organization: Organization): CustomerBudgetGroupFields[] =>
  organization.default_budgets.map((group) => ({
    id: group.budget_id,
    name: group.name,
    monthlyBudget: String(group.budget_limits?.find((entry) => entry.budget_duration === "1mo")?.max_budget ?? ""),
    yearlyBudget: String(group.budget_limits?.find((entry) => entry.budget_duration === "12mo")?.max_budget ?? ""),
  }));

export function planCustomerBudgetGroups(
  organization: Organization,
  groups: CustomerBudgetGroupFields[],
  defaultGroupId: string,
) {
  const validation = validateCustomerBudgetGroups({ groups, defaultGroupId });
  if (!validation.ok) return validation;
  const originals = organization.default_budgets;
  const collision = groups.some((group) =>
    originals.some((original) => original.name === group.name.trim() && original.budget_id !== group.id),
  );
  if (collision) return { ok: false, error: "renameCollision" } as const;
  const creates = groups
    .filter((group) => !originals.some((original) => original.budget_id === group.id))
    .map((group) => ({
      rowId: group.id,
      body: {
        name: group.name.trim(),
        budget_limits: [...limit(group.monthlyBudget, "1mo"), ...limit(group.yearlyBudget, "12mo")],
      } satisfies Template,
    }));
  const updates = originals.flatMap((original) => {
    const group = groups.find((row) => row.id === original.budget_id);
    if (!group) return [];
    const editableLimits = [...limit(group.monthlyBudget, "1mo"), ...limit(group.yearlyBudget, "12mo")];
    const originalLimits = (original.budget_limits ?? [])
      .filter((entry) => entry.budget_duration === "1mo" || entry.budget_duration === "12mo")
      .map((entry) => ({ budget_duration: entry.budget_duration, max_budget: entry.max_budget }))
      .sort((a, b) => a.budget_duration.localeCompare(b.budget_duration));
    const limitsChanged =
      JSON.stringify(originalLimits) !==
      JSON.stringify([...editableLimits].sort((a, b) => a.budget_duration.localeCompare(b.budget_duration)));
    const body: Patch = {
      ...(group.name.trim() !== original.name ? { name: group.name.trim() } : {}),
      ...(limitsChanged
        ? {
            budget_limits: [
              ...(original.budget_limits ?? []).filter(
                (entry) => entry.budget_duration !== "1mo" && entry.budget_duration !== "12mo",
              ),
              ...editableLimits,
            ],
          }
        : {}),
    };
    return Object.keys(body).length ? [{ budgetId: original.budget_id, body }] : [];
  });
  const removes = originals
    .filter((original) => !groups.some((group) => group.id === original.budget_id))
    .map((original) => original.budget_id);
  const defaultChange =
    defaultGroupId === organization.default_budget_template_id
      ? null
      : {
          rowId: defaultGroupId,
          name: groups.find((group) => group.id === defaultGroupId)!.name.trim(),
        };
  return {
    ok: true,
    creates,
    updates,
    removes,
    defaultChange,
    changed: Boolean(creates.length || updates.length || removes.length || defaultChange),
  } as const;
}
