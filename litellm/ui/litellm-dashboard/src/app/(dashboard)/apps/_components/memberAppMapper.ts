import {
  compactBudgetEditorFields,
  customerBudgetPatch,
  hasLegacyCustomerBudget,
  type BudgetEditorFields,
} from "../../customers/_components/customerBudgetForm";
import type { AppMember, Budget } from "./appForms";

export const memberAppDraft = (row: AppMember, original: Budget | null): BudgetEditorFields => ({
  ...compactBudgetEditorFields(original),
  budgetEnabled: Boolean(row.budget) || hasLegacyCustomerBudget(row),
});

export const memberAppBody = (draft: BudgetEditorFields, original: Budget | null, row: AppMember) =>
  customerBudgetPatch(draft, original, row);
