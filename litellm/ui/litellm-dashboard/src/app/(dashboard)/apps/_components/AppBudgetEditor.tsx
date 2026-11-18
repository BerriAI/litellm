"use client";

import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { useTranslation } from "react-i18next";
import { CustomerBudgetFields } from "../../customers/_components/CustomerHierarchyBudgetFields";
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
          <CustomerBudgetFields fields={fields} onChange={update} idPrefix={id} />
        </div>
      )}
    </fieldset>
  );
}
