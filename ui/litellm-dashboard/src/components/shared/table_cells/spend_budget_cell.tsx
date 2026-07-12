"use client";

import { Meter, MeterIndicator, MeterTrack } from "@/components/ui/meter";
import { formatNumberWithCommas } from "@/utils/dataUtils";

import { MoneyCell } from "./money_cell";

interface SpendBudgetCellProps {
  spend: number | null | undefined;
  maxBudget: number | null | undefined;
  teamMaxBudget?: number | null;
}

const meterTone = (pct: number): "default" | "warning" | "over" => {
  if (pct > 100) return "over";
  if (pct >= 80) return "warning";
  return "default";
};

export function SpendBudgetCell({ spend, maxBudget, teamMaxBudget }: SpendBudgetCellProps) {
  const spendValue = typeof spend === "number" && !Number.isNaN(spend) ? spend : 0;
  const budget = maxBudget ?? teamMaxBudget ?? null;
  const isTeamBudget = maxBudget == null && teamMaxBudget != null;
  const hasBudget = typeof budget === "number" && budget > 0;
  const pct = hasBudget ? (spendValue / budget) * 100 : 0;

  const budgetLabel =
    budget === null ? "· Unlimited" : `of $${formatNumberWithCommas(budget)}${isTeamBudget ? " (Team)" : ""}`;

  return (
    <div className="flex min-w-[130px] flex-col gap-1">
      <div className="whitespace-nowrap text-xs">
        <MoneyCell value={spend} decimals={4} />
        <span className="text-muted-foreground"> {budgetLabel}</span>
      </div>
      {hasBudget && (
        <Meter
          value={spendValue}
          max={budget}
          aria-valuetext={`$${formatNumberWithCommas(spendValue)} of $${formatNumberWithCommas(budget)}`}
        >
          <MeterTrack>
            <MeterIndicator tone={meterTone(pct)} />
          </MeterTrack>
        </Meter>
      )}
    </div>
  );
}
