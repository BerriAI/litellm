"use client";

import { InheritedBudgetHint, type InheritedBudgetGate } from "@/components/shared/InheritedBudgetHint";
import { Meter, MeterIndicator, MeterTrack } from "@/components/shared/Meter";
import { formatNumberWithCommas, getSpendString } from "@/utils/dataUtils";
import { useLanguage } from "@/contexts/LanguageContext";
import type { Translate } from "@/i18n/language";

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
  options: { usesTeamMemberBudget: boolean; memberSpendText: string; budgetDecimals: number },
  t: Translate,
): string => {
  const { usesTeamMemberBudget, memberSpendText, budgetDecimals } = options;
  if (budget === null) return `· ${t("Unlimited")}`;
  const formattedBudget = formatNumberWithCommas(budget, budgetDecimals);
  if (usesTeamMemberBudget)
    return t("· {spend} of {budget} (team member)", { spend: memberSpendText, budget: `$${formattedBudget}` });
  return t("of {budget}", { budget: `$${formattedBudget}` });
};

const getMeterValueText = (
  options: {
    hasBudget: boolean;
    budget: number | null;
    meterSpendText: string;
    budgetDecimals: number;
    usesTeamMemberBudget: boolean;
  },
  t: Translate,
): string => {
  const { hasBudget, budget, meterSpendText, budgetDecimals, usesTeamMemberBudget } = options;
  if (!hasBudget || budget === null) return "";
  const formattedBudget = formatNumberWithCommas(budget, budgetDecimals);
  return t(usesTeamMemberBudget ? "team member spend {spend} of {budget}" : "{spend} of {budget}", {
    spend: meterSpendText,
    budget: `$${formattedBudget}`,
  });
};

export function SpendBudgetCell({
  spend,
  maxBudget,
  teamMemberGate = null,
  inheritedGates = [],
  spendDecimals = 4,
  budgetDecimals = 0,
}: SpendBudgetCellProps) {
  const { t } = useLanguage();
  const spendValue = typeof spend === "number" && !Number.isNaN(spend) ? spend : 0;
  const ownBudget = maxBudget ?? null;
  const budget = ownBudget ?? teamMemberGate?.maxBudget ?? null;
  const usesTeamMemberBudget = ownBudget === null && teamMemberGate !== null;
  const memberSpendValue =
    typeof teamMemberGate?.spend === "number" && !Number.isNaN(teamMemberGate.spend) ? teamMemberGate.spend : 0;
  const meterSpendValue = usesTeamMemberBudget ? memberSpendValue : spendValue;
  const hasBudget = typeof budget === "number" && budget > 0;
  const pct = hasBudget ? (meterSpendValue / budget) * 100 : 0;

  const spendText = spendValue > 0 ? getSpendString(spendValue, spendDecimals) : "$0.00";
  const memberSpendText = memberSpendValue > 0 ? getSpendString(memberSpendValue, spendDecimals) : "$0.00";
  // 显示文案与无障碍文案采用完整模板，金额计算和计量值保持原样。
  const budgetLabel = getBudgetLabel(budget, { usesTeamMemberBudget, memberSpendText, budgetDecimals }, t);
  const meterSpendText = usesTeamMemberBudget ? memberSpendText : spendText;
  const meterOptions = { hasBudget, budget, meterSpendText, budgetDecimals, usesTeamMemberBudget };
  const meterValueText = getMeterValueText(meterOptions, t);

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
