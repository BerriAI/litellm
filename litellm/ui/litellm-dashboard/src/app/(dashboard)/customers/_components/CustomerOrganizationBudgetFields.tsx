import { useTranslation } from "react-i18next";
import { $api } from "@/lib/http/api";
import { ErrorState } from "@/components/shared/ErrorState";
import { CustomerBudgetEditor } from "./CustomerLimitsFields";
import { BUDGET_EDITOR_FIELDS, type BudgetEditorFields } from "./customerBudgetForm";
import { customerHierarchyFields, type CustomerHierarchyFields, type Organization } from "./customerHierarchyForms";

export default function CustomerOrganizationBudgetFields({
  organization,
  fields,
  onChange,
  idPrefix,
}: {
  organization: Organization;
  fields: CustomerHierarchyFields;
  onChange: <K extends keyof CustomerHierarchyFields>(key: K, value: CustomerHierarchyFields[K]) => void;
  idPrefix: string;
}) {
  const { t } = useTranslation();
  const groups = $api.useQuery("get", "/management/v1/customer_organization_budget_groups", {});
  const profiles = groups.data?.default_budgets ?? [];
  const originalGroup =
    organization.budget_is_shared ?? profiles.some((group) => group.budget_id === organization.budget_id);
  const usingGroup =
    fields.templateId === organization.budget_id
      ? originalGroup
      : profiles.some((group) => group.budget_id === fields.templateId);
  const budgetReady = !organization.budget_id || organization.budget_is_shared === false || Boolean(groups.data);
  const updateBudgetFields = (next: BudgetEditorFields) => {
    for (const key of BUDGET_EDITOR_FIELDS) onChange(key, next[key]);
    onChange("budgetEnabled", true);
  };
  const selectSource = (value: string | null) => {
    if (value === "custom") {
      onChange("templateId", originalGroup ? "" : organization.budget_id ?? "");
      if (usingGroup) {
        onChange("budgetEnabled", true);
      }
      return;
    }
    const profile = profiles.find((group) => `profile:${group.budget_id}` === value);
    if (!profile) return;
    const initial = customerHierarchyFields({ kind: "organization", record: organization });
    for (const key of BUDGET_EDITOR_FIELDS) onChange(key, initial[key]);
    onChange("templateId", profile.budget_id);
  };
  return (
    <CustomerBudgetEditor
      id={idPrefix}
      title={t("customerPortal.createDialog.organizationLimits")}
      description={t("customerPortal.createDialog.organizationLimitsHelp")}
      source={{
        profiles,
        value: usingGroup ? `profile:${fields.templateId}` : "custom",
        onChange: selectSource,
        title: t("customerPortal.createDialog.organizationLimits"),
        disabled: !budgetReady,
        loading: groups.isPending,
      }}
      error={groups.isError && <ErrorState onRetry={() => void groups.refetch()} />}
      budget={
        budgetReady
          ? {
              fields: {
                ...fields,
                maxBudget: fields.budgetEnabled ? fields.maxBudget : "",
                softBudget: fields.budgetEnabled ? fields.softBudget : "",
                tpm: fields.budgetEnabled ? fields.tpm : "",
                rpm: fields.budgetEnabled ? fields.rpm : "",
                allowedModels: fields.budgetEnabled ? fields.allowedModels : [],
              },
              disabled: false,
              onChange: updateBudgetFields,
            }
          : null
      }
    />
  );
}
