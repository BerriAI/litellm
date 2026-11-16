"use client";

import { useTranslation } from "react-i18next";

import { InheritedBudgetHint, type InheritedBudgetGate } from "@/components/shared/InheritedBudgetHint";
import { Meter, MeterIndicator, MeterTrack } from "@/components/shared/Meter";
import i18n from "@/i18n/i18n";
import { formatNumberWithCommas, getSpendString } from "@/utils/dataUtils";

interface SpendBudgetCellProps {
  spend: number | null | undefined;
  maxBudget: number | null | undefined;
  teamMemberGate?: InheritedBudgetGate | null;
  inheritedGates?: readonly InheritedBudgetGate[];
  spendDecimals?: number;
  budgetDecimals?: number;
}

export const getSpendBudgetMeterTone = (pct: number): "default" | "warning" | "over" => {
  if (pct > 100) return "over";
  if (pct >= 80) return "warning";
  return "default";
};

const getBudgetLabel = (
  budget: number | null,
  usesTeamMemberBudget: boolean,
  memberSpendText: string,
  budgetDecimals: number,
): string => {
  if (budget === null) return i18n.t("shared.tableCells.spendBudgetCell.unlimited");
  const formattedBudget = formatNumberWithCommas(budget, budgetDecimals);
  if (usesTeamMemberBudget)
    return i18n.t("shared.tableCells.spendBudgetCell.memberBudget", {
      spend: memberSpendText,
      budget: formattedBudget,
    });
  return i18n.t("shared.tableCells.spendBudgetCell.ofBudget", { budget: formattedBudget });
};

const getMeterValueText = (
  hasBudget: boolean,
  budget: number | null,
  meterSpendText: string,
  budgetDecimals: number,
): string => {
  if (!hasBudget || budget === null) return "";
  const formattedBudget = formatNumberWithCommas(budget, budgetDecimals);
  return i18n.t("shared.tableCells.spendBudgetCell.meterValue", { spend: meterSpendText, budget: formattedBudget });
};

const resolveBudgetDecimals = (
  budget: number | null,
  budgetDecimals: number | undefined,
): number => {
  if (budgetDecimals !== undefined) return budgetDecimals;
  if (budget === null) return 0;
  if (budget > 0 && budget < 0.01) return 5;
  if (!Number.isInteger(budget)) return 2;
  return 0;
};

export function SpendBudgetCell({
  spend,
  maxBudget,
  teamMemberGate = null,
  inheritedGates = [],
  spendDecimals = 4,
  budgetDecimals,
}: SpendBudgetCellProps) {
  const { t } = useTranslation();
  const spendValue = typeof spend === "number" && !Number.isNaN(spend) ? spend : 0;
  const ownBudget = maxBudget ?? null;
  const budget = ownBudget ?? teamMemberGate?.maxBudget ?? null;
  const effectiveBudgetDecimals = resolveBudgetDecimals(budget, budgetDecimals);
  const usesTeamMemberBudget = ownBudget === null && teamMemberGate !== null;
  const memberSpendValue =
    typeof teamMemberGate?.spend === "number" && !Number.isNaN(teamMemberGate.spend) ? teamMemberGate.spend : 0;
  const meterSpendValue = usesTeamMemberBudget ? memberSpendValue : spendValue;
  const hasBudget = typeof budget === "number" && budget > 0;
  const pct = hasBudget ? (meterSpendValue / budget) * 100 : 0;

  const spendText = spendValue > 0 ? getSpendString(spendValue, spendDecimals) : "$0.00";
  const memberSpendText = memberSpendValue > 0 ? getSpendString(memberSpendValue, spendDecimals) : "$0.00";
  const budgetLabel = getBudgetLabel(budget, usesTeamMemberBudget, memberSpendText, effectiveBudgetDecimals);
  const meterSpendText = usesTeamMemberBudget
    ? t("shared.tableCells.spendBudgetCell.memberSpend", { spend: memberSpendText })
    : spendText;
  const meterValueText = getMeterValueText(hasBudget, budget, meterSpendText, effectiveBudgetDecimals);

  return (
    <div className="flex min-w-[130px] flex-col gap-1">
      <div className="whitespace-nowrap text-xs">
        <span className="font-medium tabular-nums text-foreground">{spendText}</span>{" "}
        <span className="text-muted-foreground">{budgetLabel}</span>
        {ownBudget === null && <InheritedBudgetHint gates={inheritedGates} />}
      </div>
      {hasBudget && (
        <Meter value={meterSpendValue} max={budget} aria-valuetext={meterValueText}>
          <MeterTrack>
            <MeterIndicator tone={getSpendBudgetMeterTone(pct)} />
          </MeterTrack>
        </Meter>
      )}
    </div>
  );
}
