"use client";

import { useTranslation } from "react-i18next";
import {
  type CustomerHierarchyFields,
  type CustomerHierarchyEntity,
  type Organization,
} from "./customerHierarchyForms";
import { BUDGET_EDITOR_FIELDS, type BudgetEditorFields } from "./customerBudgetForm";
import CustomerOrganizationBudgetFields from "./CustomerOrganizationBudgetFields";
import { CustomerBudgetEditor } from "./CustomerLimitsFields";

interface Props {
  kind: CustomerHierarchyEntity["kind"];
  organization: Organization;
  fields: CustomerHierarchyFields;
  onChange: <K extends keyof CustomerHierarchyFields>(key: K, value: CustomerHierarchyFields[K]) => void;
  idPrefix: string;
  compact?: boolean;
}

export default function CustomerHierarchyBudgetFields({
  kind,
  organization,
  fields,
  onChange: update,
  idPrefix,
  compact = false,
}: Props) {
  const { t } = useTranslation();
  if (kind === "organization") {
    return (
      <div className="border-t pt-6">
        <CustomerOrganizationBudgetFields
          organization={organization}
          fields={fields}
          onChange={update}
          idPrefix={idPrefix}
        />
      </div>
    );
  }
  const custom = fields.budgetEnabled || !fields.templateId;
  const source = custom ? "custom" : `profile:${fields.templateId}`;
  const selectSource = (value: string | null) => {
    if (value === "custom") return update("budgetEnabled", true);
    if (!value?.startsWith("profile:")) return;
    update("templateId", value.slice("profile:".length));
    update("budgetEnabled", false);
  };
  const updateCompact = (next: BudgetEditorFields) => {
    for (const key of BUDGET_EDITOR_FIELDS) update(key, next[key]);
    update("budgetEnabled", true);
  };
  return (
    <CustomerBudgetEditor
      id={idPrefix}
      className={compact ? "space-y-4" : "space-y-4 border-t pt-6"}
      title={t("customerPortal.hierarchy.budgetTitle")}
      description={t(
        kind === "member" ? "customerPortal.hierarchy.memberBudgetHint" : "customerPortal.hierarchy.budgetHint",
      )}
      budget={{ fields, onChange: updateCompact, disabled: false }}
      source={
        kind !== "member"
          ? undefined
          : {
              profiles: organization.default_budgets.map((profile) => ({
                ...profile,
                name:
                  profile.budget_id === organization.default_budget_template_id
                    ? t("customerPortal.hierarchy.defaultTemplate", { name: profile.name })
                    : profile.name,
              })),
              value: source,
              title: t(compact ? "customerPortal.hierarchy.memberLimits" : "customerPortal.hierarchy.template"),
              disabled: false,
              onChange: selectSource,
            }
      }
    />
  );
}
