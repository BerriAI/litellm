import type { BudgetEditorFields } from "./customerBudgetForm";
import { CustomerBudgetFields } from "./CustomerBudgetFields";
import CustomerBudgetWindows from "./CustomerBudgetWindows";
import type { CustomerModelScope } from "./customerModelOptions";

export default function CustomerBudgetCompactFields({
  fields,
  onChange,
  disabled,
  id,
  modelScope,
}: {
  fields: BudgetEditorFields;
  onChange: (fields: BudgetEditorFields) => void;
  disabled: boolean;
  id: string;
  modelScope?: CustomerModelScope;
}) {
  return (
    <CustomerBudgetFields
      fields={fields}
      onChange={(key, value) => onChange({ ...fields, [key]: value })}
      idPrefix={id}
      modelScope={modelScope}
      disabled={disabled}
      unavailablePeriods={fields.budgetWindows
        .filter(({ maxBudget }) => maxBudget.trim() !== "")
        .map(({ budget }) => budget.budget_duration)}
    >
      <CustomerBudgetWindows
        id={id}
        windows={fields.budgetWindows}
        onChange={(budgetWindows) => onChange({ ...fields, budgetWindows })}
        disabled={disabled}
        primaryPeriod={fields.maxBudget.trim() ? fields.budgetDuration : undefined}
      />
    </CustomerBudgetFields>
  );
}
