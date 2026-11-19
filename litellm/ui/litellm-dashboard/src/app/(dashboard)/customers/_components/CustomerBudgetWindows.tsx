import { useState } from "react";
import { useTranslation } from "react-i18next";
import { Plus, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Popover, PopoverContent, PopoverTitle, PopoverTrigger } from "@/components/ui/popover";
import { isValidBudgetDuration, normalizeBudgetDuration } from "@/utils/budgetUtils";
import CustomerBudgetAmountInput from "./CustomerBudgetAmountInput";
import CustomerBudgetPeriodSelect from "./CustomerBudgetPeriodSelect";
import { budgetFields, parseBudget, type BudgetEditorFields } from "./customerBudgetForm";

type Windows = BudgetEditorFields["budgetWindows"];

export default function CustomerBudgetWindows({
  id,
  windows,
  onChange,
  disabled = false,
  compact = false,
  primaryPeriod,
  onAdd,
}: {
  id: string;
  windows: Windows;
  onChange: (windows: Windows) => void;
  disabled?: boolean;
  compact?: boolean;
  primaryPeriod?: string;
  onAdd?: (window: Windows[number]) => void;
}) {
  const { t } = useTranslation();
  const primaryPeriods = primaryPeriod ? [primaryPeriod] : [];
  const rows = windows.map(({ budget, maxBudget }, index) => {
    const period = budget.budget_duration;
    const inputId = `${id}-window-${index}`;
    const changeAmount = (value: string) =>
      onChange(windows.map((entry, i) => (i === index ? { ...entry, maxBudget: value } : entry)));
    const unavailablePeriods = [
      ...primaryPeriods,
      ...windows.filter((_, i) => i !== index).map((entry) => entry.budget.budget_duration),
    ].map(normalizeBudgetDuration);
    const remove = (
      <Button
        type="button"
        variant={compact ? "ghost" : "outline"}
        size={compact ? "icon-sm" : "default"}
        className={compact ? "mr-0.5 size-6 text-muted-foreground" : undefined}
        disabled={disabled}
        aria-label={t("apps.removeWindow")}
        onClick={() => onChange(windows.filter((_, i) => i !== index))}
      >
        {compact ? <X className="size-3" /> : t("apps.removeWindow")}
      </Button>
    );
    return (
      <div key={index} className={compact ? "shrink-0" : "space-y-2"}>
        <Label htmlFor={inputId} className={compact ? "sr-only" : "font-normal text-muted-foreground"}>
          {t("apps.additionalSpendLimit")}
        </Label>
        <div className="flex items-center gap-2">
          <CustomerBudgetAmountInput
            id={inputId}
            label={t("apps.additionalSpendLimit")}
            aria-describedby={`${id}-windows-hint`}
            value={maxBudget}
            onChange={changeAmount}
            disabled={disabled}
            compact={compact}
          >
            <CustomerBudgetPeriodSelect
              value={period}
              allowNoReset={false}
              compact={compact}
              disabled={disabled}
              unavailablePeriods={unavailablePeriods}
              onChange={(value) => {
                if (value === period || unavailablePeriods.includes(normalizeBudgetDuration(value))) return;
                onChange(
                  windows.map((entry, i) =>
                    i === index
                      ? { ...entry, budget: { ...entry.budget, budget_duration: value, reset_at: null } }
                      : entry,
                  ),
                );
              }}
            />
            {compact && remove}
          </CustomerBudgetAmountInput>
          {!compact && remove}
        </div>
      </div>
    );
  });
  return (
    <>
      {windows.length > 0 && (
        <p id={`${id}-windows-hint`} className={compact ? "sr-only" : "text-xs text-muted-foreground"}>
          {t("apps.windowBudgetHint")}
        </p>
      )}
      {rows}
      {compact && (
        <AddBudgetWindow
          id={id}
          disabled={disabled}
          unavailablePeriods={[...primaryPeriods, ...windows.map(({ budget }) => budget.budget_duration)]}
          onAdd={onAdd ?? ((window) => onChange([...windows, window]))}
        />
      )}
    </>
  );
}

function AddBudgetWindow({
  id,
  disabled,
  unavailablePeriods,
  onAdd,
}: {
  id: string;
  disabled: boolean;
  unavailablePeriods: readonly string[];
  onAdd: (window: Windows[number]) => void;
}) {
  const { t } = useTranslation();
  const [draft, setDraft] = useState<{ amount: string; period: string } | null>(null);
  const period = draft?.period.trim() ?? "";
  const unavailable = unavailablePeriods.map(normalizeBudgetDuration);
  const amount = parseBudget({ ...budgetFields(null), maxBudget: draft?.amount ?? "" })?.max_budget;
  const validPeriod =
    period !== "" && isValidBudgetDuration(period) && !unavailable.includes(normalizeBudgetDuration(period));
  const valid = amount != null && validPeriod;
  const nextPeriod = ["1mo", "12mo"].find((value) => !unavailable.includes(value)) ?? "";
  const add = () => {
    if (!valid || !draft) return;
    onAdd({ budget: { max_budget: amount, budget_duration: period }, maxBudget: draft.amount });
    setDraft(null);
  };
  return (
    <Popover open={draft !== null} onOpenChange={(open) => setDraft(open ? { amount: "", period: nextPeriod } : null)}>
      <PopoverTrigger
        render={<Button type="button" variant="outline" size="icon-sm" />}
        className="shrink-0 border-dashed text-muted-foreground shadow-none"
        disabled={disabled}
        aria-label={t("keyTeamHelpers.budgetWindowsEditor.addWindow")}
      >
        <Plus className="size-3" />
      </PopoverTrigger>
      {draft && (
        <PopoverContent align="start">
          <PopoverTitle>{t("apps.additionalSpendLimit")}</PopoverTitle>
          <CustomerBudgetAmountInput
            id={`${id}-add-window`}
            label={t("apps.additionalSpendLimit")}
            value={draft.amount}
            onChange={(amount) => setDraft({ ...draft, amount })}
          >
            <CustomerBudgetPeriodSelect
              value={draft.period}
              allowNoReset={false}
              unavailablePeriods={unavailablePeriods}
              onChange={(period) => setDraft({ ...draft, period })}
            />
          </CustomerBudgetAmountInput>
          <Button type="button" disabled={!valid} onClick={add}>
            {t("apps.addLimit")}
          </Button>
        </PopoverContent>
      )}
    </Popover>
  );
}
