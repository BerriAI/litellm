import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import type { BudgetEditorFields } from "./customerBudgetForm";

type Windows = BudgetEditorFields["budgetWindows"];

export default function CustomerBudgetWindows({
  id,
  windows,
  onChange,
  removable = false,
}: {
  id: string;
  windows: Windows;
  onChange: (windows: Windows) => void;
  removable?: boolean;
}) {
  const { t } = useTranslation();
  const WindowLabel = removable ? "label" : Label;
  return windows.map(({ budget, maxBudget }, index) => {
    const period = budget.budget_duration;
    if (!removable && (period === "1mo" || period === "12mo")) return null;
    const inputId = `${id}-window-${index}`;
    return (
      <div key={`${period}-${index}`} className={removable ? "space-y-1.5" : "space-y-2"}>
        <WindowLabel htmlFor={inputId} className={removable ? "block text-sm" : "font-normal text-muted-foreground"}>
          {t("apps.windowBudget", { period })}
        </WindowLabel>
        <div className={removable ? "flex items-center gap-2" : undefined}>
          <Input
            id={inputId}
            type="number"
            min={0}
            step="any"
            value={maxBudget}
            placeholder={removable ? undefined : t("customerPortal.createDialog.noLimit")}
            onChange={(event) =>
              onChange(windows.map((entry, i) => (i === index ? { ...entry, maxBudget: event.target.value } : entry)))
            }
          />
          {removable && (
            <Button type="button" variant="outline" onClick={() => onChange(windows.filter((_, i) => i !== index))}>
              {t("apps.removeWindow")}
            </Button>
          )}
        </div>
      </div>
    );
  });
}
