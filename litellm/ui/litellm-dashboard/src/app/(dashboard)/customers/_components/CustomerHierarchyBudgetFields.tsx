"use client";

import { useTranslation } from "react-i18next";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import type { CustomerHierarchyFields, CustomerHierarchyEntity, Organization } from "./customerHierarchyForms";
import CustomerAllowedModelsSelect from "./CustomerAllowedModelsSelect";

interface Props {
  kind: CustomerHierarchyEntity["kind"];
  organization: Organization;
  fields: CustomerHierarchyFields;
  onChange: <K extends keyof CustomerHierarchyFields>(key: K, value: CustomerHierarchyFields[K]) => void;
  idPrefix: string;
  operation: "create" | "edit";
}

export default function CustomerHierarchyBudgetFields({
  kind,
  organization,
  fields,
  onChange: update,
  idPrefix,
  operation,
}: Props) {
  const { t } = useTranslation();
  const templateLabel = (templateId: string): string => {
    if (!templateId) return t("customerPortal.hierarchy.noTemplate");
    const template = organization.default_budgets.find((budget) => budget.budget_id === templateId);
    if (!template) return t("customerPortal.hierarchy.unknownTemplate", { id: templateId });
    if (templateId === organization.default_budget_template_id)
      return t("customerPortal.hierarchy.defaultTemplate", { name: template.name });
    return template.name;
  };
  const numericFields = [
    ["maxBudget", "customerPortal.hierarchy.maxBudget", "any"],
    ["softBudget", "customerPortal.hierarchy.softBudget", "any"],
    ["tpm", "customerPortal.hierarchy.tpm", "1"],
    ["rpm", "customerPortal.hierarchy.rpm", "1"],
  ] as const;
  return (
    <section className="space-y-4 border-t pt-6" aria-labelledby={`${idPrefix}-budget-title`}>
      <div>
        <h2 id={`${idPrefix}-budget-title`} className="text-base font-medium">
          {t("customerPortal.hierarchy.budgetTitle")}
        </h2>
        <p className="mt-1 text-sm text-muted-foreground">
          {t(kind === "member" ? "customerPortal.hierarchy.memberBudgetHint" : "customerPortal.hierarchy.budgetHint")}
        </p>
      </div>
      {kind === "member" && (
        <div className="max-w-sm space-y-1.5">
          <Label htmlFor={`${idPrefix}-template`}>{t("customerPortal.hierarchy.template")}</Label>
          <Select
            value={fields.templateId || "__none__"}
            onValueChange={(value) => update("templateId", value === "__none__" ? "" : value ?? "")}
          >
            <SelectTrigger id={`${idPrefix}-template`} className="w-full">
              <SelectValue>{templateLabel(fields.templateId)}</SelectValue>
            </SelectTrigger>
            <SelectContent>
              {(operation === "edit" || !organization.default_budget_template_id) && (
                <SelectItem value="__none__">{t("customerPortal.hierarchy.noTemplate")}</SelectItem>
              )}
              {organization.default_budgets.map((template) => (
                <SelectItem key={template.budget_id} value={template.budget_id}>
                  {templateLabel(template.budget_id)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      )}
      <div className="space-y-1.5">
        <div className="flex items-center gap-2">
          <Checkbox
            id={`${idPrefix}-budget-enabled`}
            checked={fields.budgetEnabled}
            onCheckedChange={(checked) => update("budgetEnabled", checked)}
          />
          <Label htmlFor={`${idPrefix}-budget-enabled`}>
            {t(
              kind === "member" ? "customerPortal.hierarchy.overrideEnabled" : "customerPortal.hierarchy.budgetEnabled",
            )}
          </Label>
        </div>
        {kind === "member" && (
          <p className="text-xs text-muted-foreground">{t("customerPortal.hierarchy.overrideHint")}</p>
        )}
      </div>
      {fields.budgetEnabled && (
        <div className="space-y-4">
          <div className="grid gap-4 sm:grid-cols-2">
            {numericFields.map(([name, label, step]) => (
              <div className="space-y-1.5" key={name}>
                <Label htmlFor={`${idPrefix}-${name}`}>{t(label)}</Label>
                <Input
                  id={`${idPrefix}-${name}`}
                  type="number"
                  min="0"
                  step={step}
                  value={fields[name]}
                  onChange={(event) => update(name, event.target.value)}
                />
              </div>
            ))}
            <div className="space-y-1.5">
              <Label htmlFor={`${idPrefix}-duration`}>{t("customerPortal.hierarchy.budgetDuration")}</Label>
              <Input
                id={`${idPrefix}-duration`}
                value={fields.budgetDuration}
                onChange={(event) => update("budgetDuration", event.target.value)}
                aria-describedby={`${idPrefix}-duration-hint`}
              />
              <p id={`${idPrefix}-duration-hint`} className="text-xs text-muted-foreground">
                {t("customerPortal.hierarchy.budgetDurationHint")}
              </p>
            </div>
            {kind === "member" && (
              <div className="space-y-1.5">
                <Label htmlFor={`${idPrefix}-expiry`}>{t("customerPortal.hierarchy.overrideExpiry")}</Label>
                <Input
                  id={`${idPrefix}-expiry`}
                  type="datetime-local"
                  value={fields.overrideExpiry}
                  onChange={(event) => update("overrideExpiry", event.target.value)}
                  aria-describedby={`${idPrefix}-expiry-hint`}
                />
                <p id={`${idPrefix}-expiry-hint`} className="text-xs text-muted-foreground">
                  {t("customerPortal.hierarchy.overrideExpiryHint")}
                </p>
              </div>
            )}
          </div>
          <div className="space-y-1.5">
            <Label htmlFor={`${idPrefix}-models`}>{t("customerPortal.hierarchy.allowedModels")}</Label>
            <CustomerAllowedModelsSelect
              id={`${idPrefix}-models`}
              value={fields.allowedModels}
              onChange={(models) => update("allowedModels", models)}
            />
            <p id={`${idPrefix}-models-hint`} className="text-xs text-muted-foreground">
              {t("customerPortal.hierarchy.allowedModelsHint")}
            </p>
          </div>
        </div>
      )}
    </section>
  );
}
