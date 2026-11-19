"use client";

import { useTranslation } from "react-i18next";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import {
  BUDGET_FIELDS,
  type CustomerHierarchyFields,
  type CustomerHierarchyEntity,
  type Organization,
} from "./customerHierarchyForms";
import type { Budget, BudgetEditorFields } from "./customerBudgetForm";
import CustomerBudgetCompactFields from "./CustomerBudgetCompactFields";
import CustomerOrganizationBudgetFields from "./CustomerOrganizationBudgetFields";
import { CustomerBudgetFields } from "./CustomerBudgetFields";
import { CustomerBudgetEditor } from "./CustomerLimitsFields";

interface Props {
  kind: CustomerHierarchyEntity["kind"];
  organization: Organization;
  fields: CustomerHierarchyFields;
  onChange: <K extends keyof CustomerHierarchyFields>(key: K, value: CustomerHierarchyFields[K]) => void;
  idPrefix: string;
  compact?: boolean;
  original?: Budget | null;
}

export default function CustomerHierarchyBudgetFields({
  kind,
  organization,
  fields,
  onChange: update,
  idPrefix,
  compact = false,
  original,
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
    for (const key of BUDGET_FIELDS) update(key, next[key]);
    update("budgetEnabled", true);
    update("allowedModels", next.allowedModels);
    update("budgetWindows", next.budgetWindows);
  };
  return (
    <CustomerBudgetEditor
      id={idPrefix}
      className={compact ? "space-y-4" : "space-y-4 border-t pt-6"}
      title={t("customerPortal.hierarchy.budgetTitle")}
      description={t(
        kind === "member" ? "customerPortal.hierarchy.memberBudgetHint" : "customerPortal.hierarchy.budgetHint",
      )}
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
    >
      {kind === "team" && (
        <div className="space-y-1.5">
          <div className="flex items-center gap-2">
            <Checkbox
              id={`${idPrefix}-budget-enabled`}
              checked={fields.budgetEnabled}
              onCheckedChange={(checked) => update("budgetEnabled", checked)}
            />
            <Label htmlFor={`${idPrefix}-budget-enabled`}>{t("customerPortal.hierarchy.budgetEnabled")}</Label>
          </div>
        </div>
      )}
      {kind === "member" && (
        <CustomerBudgetCompactFields
          fields={fields}
          original={original}
          onChange={updateCompact}
          disabled={false}
          id={idPrefix}
        />
      )}
      {kind === "team" && fields.budgetEnabled && (
        <CustomerBudgetFields fields={fields} onChange={update} idPrefix={idPrefix} />
      )}
    </CustomerBudgetEditor>
  );
}
