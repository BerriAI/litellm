"use client";

import { Meter, MeterIndicator, MeterTrack } from "@/components/shared/Meter";
import { getSpendBudgetMeterTone } from "@/components/shared/table_cells/spend_budget_cell";
import { formatBudgetReset } from "@/utils/budgetUtils";
import { formatNumberWithCommas, getSpendString } from "@/utils/dataUtils";

interface CallerMembership {
  spend: number;
  max_budget: number | null;
  budget_reset_at: string | null;
}

interface TeamMemberSpendBudgetCellProps {
  teamSpend: number | null | undefined;
  teamMaxBudget: number | null | undefined;
  callerMembership?: CallerMembership | null;
}

interface BudgetLineProps {
  label: "Team" | "Member";
  spend: number | null | undefined;
  maxBudget: number | null | undefined;
  resetAt?: string | null;
}

function BudgetLine({ label, spend, maxBudget, resetAt }: BudgetLineProps) {
  const spendValue = typeof spend === "number" && Number.isFinite(spend) ? spend : 0;
  const budget = typeof maxBudget === "number" && Number.isFinite(maxBudget) ? maxBudget : null;
  const spendText = spendValue === 0 ? "$0.00" : getSpendString(spendValue, 2);
  const budgetText = budget === null ? "Unlimited" : `$${formatNumberWithCommas(budget, 2)}`;
  const resetText = label === "Member" ? formatBudgetReset(resetAt) : null;

  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-baseline justify-between gap-2 whitespace-nowrap text-xs">
        <span className="text-[10px] font-semibold text-muted-foreground">{label}</span>
        <span>
          <span className="font-medium tabular-nums text-foreground">{spendText}</span>{" "}
          <span className="text-muted-foreground">/ {budgetText}</span>
        </span>
      </div>
      {budget !== null && budget > 0 && (
        <Meter value={spendValue} max={budget} aria-valuetext={`${label} ${spendText} of ${budgetText}`}>
          <MeterTrack>
            <MeterIndicator tone={getSpendBudgetMeterTone((spendValue / budget) * 100)} />
          </MeterTrack>
        </Meter>
      )}
      {resetText && <div className="text-[10px] text-muted-foreground">Resets {resetText}</div>}
    </div>
  );
}

export function TeamMemberSpendBudgetCell({
  teamSpend,
  teamMaxBudget,
  callerMembership,
}: TeamMemberSpendBudgetCellProps) {
  return (
    <div className="flex min-w-[130px] flex-col gap-2">
      <BudgetLine label="Team" spend={teamSpend} maxBudget={teamMaxBudget} />
      {callerMembership && (
        <BudgetLine
          label="Member"
          spend={callerMembership.spend}
          maxBudget={callerMembership.max_budget}
          resetAt={callerMembership.budget_reset_at}
        />
      )}
    </div>
  );
}
