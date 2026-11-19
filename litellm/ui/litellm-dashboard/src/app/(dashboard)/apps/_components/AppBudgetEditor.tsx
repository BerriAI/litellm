"use client";

import { type BudgetEditorFields } from "../../customers/_components/customerBudgetForm";
import type { ComponentProps } from "react";
import CustomerBudgetWindows from "../../customers/_components/CustomerBudgetWindows";
import { useTranslation } from "react-i18next";
import { CustomerBudgetFields } from "../../customers/_components/CustomerBudgetFields";
import { CustomerBudgetEditor, CustomerBudgetSourceSelect } from "../../customers/_components/CustomerLimitsFields";
import type { CustomerModelScope } from "../../customers/_components/customerModelOptions";

export default function AppBudgetEditor({
  title,
  description,
  fields,
  onChange,
  id,
  modelScope,
  source,
  disabled = false,
}: {
  title: string;
  description: string;
  fields: BudgetEditorFields;
  onChange: (fields: BudgetEditorFields) => void;
  id: string;
  modelScope: CustomerModelScope;
  source?: Pick<ComponentProps<typeof CustomerBudgetSourceSelect>, "profiles" | "value" | "onChange">;
  disabled?: boolean;
}) {
  const { t } = useTranslation();
  const update = <K extends keyof BudgetEditorFields>(key: K, value: BudgetEditorFields[K]) =>
    onChange({ ...fields, [key]: value });
  return (
    <CustomerBudgetEditor
      id={id}
      title={title}
      description={description}
      source={{
        profiles: [],
        value: fields.budgetEnabled ? "custom" : "inherit",
        title,
        disabled,
        inheritedOption: {
          label: t("customerPortal.createDialog.noLimit"),
          summary: t("apps.noAdditionalLimits"),
        },
        onChange: (source) => update("budgetEnabled", source === "custom"),
        ...source,
      }}
    >
      <CustomerBudgetFields
        layout="scalar"
        fields={fields}
        onChange={update}
        idPrefix={id}
        modelScope={modelScope}
        disabled={disabled}
      >
        <CustomerBudgetWindows
          id={id}
          windows={fields.budgetWindows}
          onChange={(windows) => update("budgetWindows", windows)}
          removable
          disabled={disabled}
        />
      </CustomerBudgetFields>
    </CustomerBudgetEditor>
  );
}
