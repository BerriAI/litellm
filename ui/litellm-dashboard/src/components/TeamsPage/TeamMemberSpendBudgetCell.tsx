"use client";

import { Meter, MeterIndicator, MeterTrack } from "@/components/shared/Meter";
import { getSpendBudgetMeterTone } from "@/components/shared/table_cells/spend_budget_cell";
import { useLanguage } from "@/contexts/LanguageContext";
import type { components } from "@/lib/http/schema";
import { formatBudgetResetUtc } from "@/utils/budgetUtils";
import { formatNumberWithCommas, getSpendString } from "@/utils/dataUtils";

type CallerMembership = components["schemas"]["TeamListCallerMembership"];

interface TeamMemberSpendBudgetCellProps {
  teamSpend: number | null | undefined;
  teamMaxBudget: number | null | undefined;
  callerMembership?: CallerMembership | null;
}

interface BudgetLineProps {
  label: "Team" | "My Spend";
  spend: number | null | undefined;
  maxBudget: number | null | undefined;
  resetAt?: string | null;
}

function BudgetLine({ label, spend, maxBudget, resetAt }: BudgetLineProps) {
  const { t } = useLanguage();
  const spendValue = typeof spend === "number" && Number.isFinite(spend) ? spend : 0;
  const budget = typeof maxBudget === "number" && Number.isFinite(maxBudget) ? maxBudget : null;
  const spendText = spendValue === 0 ? "$0.00" : getSpendString(spendValue, 2);
  const budgetText = budget === null ? t("Unlimited") : `$${formatNumberWithCommas(budget, 2)}`;
  const resetText = formatBudgetResetUtc(resetAt);

  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-baseline justify-between gap-2 whitespace-nowrap text-xs">
        <span className="text-[10px] font-semibold text-muted-foreground">{t(label)}</span>
        <span>
          <span className="font-medium tabular-nums text-foreground">{spendText}</span>{" "}
          <span className="text-muted-foreground">/ {budgetText}</span>
        </span>
      </div>
      {budget !== null && budget > 0 && (
        <Meter
          value={spendValue}
          max={budget}
          aria-valuetext={t("{label} {spend} of {budget}", { label: t(label), spend: spendText, budget: budgetText })}
        >
          <MeterTrack>
            <MeterIndicator tone={getSpendBudgetMeterTone((spendValue / budget) * 100)} />
          </MeterTrack>
        </Meter>
      )}
      {/* 保持上游 UTC 日期语义，切换语言不重新解释重置时间。 */}
      {resetText && (
        <div className="text-[10px] text-muted-foreground">{t("Resets {date} UTC", { date: resetText })}</div>
      )}
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
          label="My Spend"
          spend={callerMembership.spend}
          maxBudget={callerMembership.max_budget}
          resetAt={callerMembership.budget_reset_at}
        />
      )}
    </div>
  );
}
