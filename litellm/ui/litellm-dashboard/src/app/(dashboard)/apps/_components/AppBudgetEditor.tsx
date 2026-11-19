"use client";

import { type BudgetEditorFields } from "../../customers/_components/customerBudgetForm";

import { Checkbox } from "@/components/ui/checkbox";
import CustomerBudgetWindows from "../../customers/_components/CustomerBudgetWindows";
import { useTranslation } from "react-i18next";
import { CustomerBudgetFields } from "../../customers/_components/CustomerBudgetFields";
import type { CustomerModelScope } from "../../customers/_components/customerModelOptions";

export default function AppBudgetEditor({
  title,
  fields,
  onChange,
  id,
  modelScope,
}: {
  title: string;
  fields: BudgetEditorFields;
  onChange: (fields: BudgetEditorFields) => void;
  id: string;
  modelScope: CustomerModelScope;
}) {
  const { t } = useTranslation();
  const update = <K extends keyof BudgetEditorFields>(key: K, value: BudgetEditorFields[K]) =>
    onChange({ ...fields, [key]: value });
  return (
    <fieldset className="min-w-0 space-y-4 border-t pt-4">
      <legend className="px-1 text-sm font-semibold">{title}</legend>
      <label className="flex items-center gap-2 text-sm">
        <Checkbox checked={fields.budgetEnabled} onCheckedChange={(checked) => update("budgetEnabled", checked)} />
        {t("customerPortal.hierarchy.budgetEnabled")}
      </label>
      {fields.budgetEnabled && (
        <div className="space-y-4">
          <CustomerBudgetWindows
            id={id}
            windows={fields.budgetWindows}
            onChange={(windows) => update("budgetWindows", windows)}
            removable
          />
          <CustomerBudgetFields fields={fields} onChange={update} idPrefix={id} modelScope={modelScope} />
        </div>
      )}
    </fieldset>
  );
}
