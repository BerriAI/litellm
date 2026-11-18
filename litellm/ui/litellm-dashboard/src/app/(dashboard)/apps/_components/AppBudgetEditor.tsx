"use client";

import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { useTranslation } from "react-i18next";
import CustomerAllowedModelsSelect from "../../customers/_components/CustomerAllowedModelsSelect";
import type { AppBudgetFields } from "./appForms";

export default function AppBudgetEditor({
  title,
  fields,
  onChange,
  id,
  optional = true,
}: {
  title: string;
  fields: AppBudgetFields;
  onChange: (fields: AppBudgetFields) => void;
  id: string;
  optional?: boolean;
}) {
  const { t } = useTranslation();
  const update = <K extends keyof AppBudgetFields>(key: K, value: AppBudgetFields[K]) =>
    onChange({ ...fields, [key]: value });
  const numericFields = [
    ["maxBudget", "customerPortal.hierarchy.maxBudget", 0.01],
    ["softBudget", "customerPortal.hierarchy.softBudget", 0.01],
    ["tpm", "customerPortal.hierarchy.tpm", 1],
    ["rpm", "customerPortal.hierarchy.rpm", 1],
  ] as const;
  return (
    <fieldset className="min-w-0 space-y-4 border-t pt-4">
      <legend className="px-1 text-sm font-semibold">{title}</legend>
      {optional && (
        <label className="flex items-center gap-2 text-sm">
          <Checkbox checked={fields.budgetEnabled} onCheckedChange={(checked) => update("budgetEnabled", checked)} />
          {t("customerPortal.hierarchy.budgetEnabled")}
        </label>
      )}
      {fields.budgetEnabled && (
        <div className="space-y-4">
          {fields.budgetWindows.map((window, index) => (
            <div key={`${window.budget.budget_duration}-${index}`} className="space-y-1.5">
              <label htmlFor={`${id}-window-${index}`} className="block text-sm">
                {t("apps.windowBudget", { period: window.budget.budget_duration })}
              </label>
              <div className="flex items-center gap-2">
                <Input
                  id={`${id}-window-${index}`}
                  type="number"
                  min="0"
                  step="any"
                  value={window.maxBudget}
                  onChange={(event) =>
                    update(
                      "budgetWindows",
                      fields.budgetWindows.map((entry, entryIndex) =>
                        entryIndex === index ? { ...entry, maxBudget: event.target.value } : entry,
                      ),
                    )
                  }
                />
                <Button
                  type="button"
                  variant="outline"
                  onClick={() =>
                    update(
                      "budgetWindows",
                      fields.budgetWindows.filter((_, entryIndex) => entryIndex !== index),
                    )
                  }
                >
                  {t("apps.removeWindow")}
                </Button>
              </div>
            </div>
          ))}
          <div className="grid gap-4 sm:grid-cols-2">
            {numericFields.map(([name, label, step]) => (
              <div key={name} className="space-y-1.5">
                <label htmlFor={`${id}-${name}`} className="block text-sm">
                  {t(label)}
                </label>
                <Input
                  id={`${id}-${name}`}
                  type="number"
                  min="0"
                  step={step}
                  className="w-full"
                  value={fields[name]}
                  onChange={(event) => update(name, event.target.value)}
                />
              </div>
            ))}
            <div className="space-y-1.5">
              <label htmlFor={`${id}-duration`} className="block text-sm">
                {t("customerPortal.hierarchy.budgetDuration")}
              </label>
              <Input
                id={`${id}-duration`}
                value={fields.budgetDuration}
                onChange={(event) => update("budgetDuration", event.target.value)}
                aria-describedby={`${id}-duration-hint`}
              />
              <p id={`${id}-duration-hint`} className="text-xs text-muted-foreground">
                {t("customerPortal.hierarchy.budgetDurationHint")}
              </p>
            </div>
          </div>
          <div className="space-y-1.5">
            <label htmlFor={`${id}-models`} className="block text-sm">
              {t("customerPortal.hierarchy.allowedModels")}
            </label>
            <CustomerAllowedModelsSelect
              id={`${id}-models`}
              value={fields.allowedModels}
              onChange={(value) => update("allowedModels", value)}
            />
          </div>
        </div>
      )}
    </fieldset>
  );
}
