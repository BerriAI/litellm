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

const nextPeriod = (periods: readonly string[]) =>
  ["1mo", "12mo"].find((period) => !periods.map(normalizeBudgetDuration).includes(period)) ?? "";

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
        size={compact ? "icon-sm" : "icon"}
        className={compact ? "mr-0.5 size-6 text-muted-foreground" : "text-muted-foreground"}
        disabled={disabled}
        aria-label={t("apps.removeWindow")}
        onClick={() => onChange(windows.filter((_, i) => i !== index))}
      >
        <X className={compact ? "size-3" : "size-4"} />
      </Button>
    );
    return (
      <div key={index} className={compact ? "shrink-0" : "space-y-2"}>
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
                if (value === period) return;
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
  const unavailablePeriods = [...primaryPeriods, ...windows.map(({ budget }) => budget.budget_duration)];
  const add = onAdd ?? ((window: Windows[number]) => onChange([...windows, window]));
  const hint = (
    <p id={`${id}-windows-hint`} className={compact ? "sr-only" : "text-xs text-muted-foreground"}>
      {t("apps.windowBudgetHint")}
    </p>
  );
  if (!compact) {
    return (
      <div className="space-y-2">
        {windows.length > 0 && (
          <Label htmlFor={`${id}-window-0`} className="font-normal text-muted-foreground">
            {t("apps.additionalSpendLimit")}
          </Label>
        )}
        {rows}
        <Button
          type="button"
          variant="link"
          size="sm"
          className="h-auto gap-1 px-0 font-normal"
          disabled={disabled}
          onClick={() =>
            add({ budget: { max_budget: 0, budget_duration: nextPeriod(unavailablePeriods) }, maxBudget: "" })
          }
        >
          <Plus className="size-3" />
          {t("apps.addAnotherSpendLimit")}
        </Button>
        {hint}
      </div>
    );
  }
  return (
    <>
      {windows.length > 0 && hint}
      {rows}
      <AddBudgetWindow id={id} disabled={disabled} unavailablePeriods={unavailablePeriods} onAdd={add} />
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
  const add = () => {
    if (!valid || !draft) return;
    onAdd({ budget: { max_budget: amount, budget_duration: period }, maxBudget: draft.amount });
    setDraft(null);
  };
  return (
    <Popover
      open={draft !== null}
      onOpenChange={(open) => setDraft(open ? { amount: "", period: nextPeriod(unavailablePeriods) } : null)}
    >
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
