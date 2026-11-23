import { BudgetPeriodTooltip } from "@/components/shared/BudgetPeriodTooltip";
import type { UserInfoV2Response } from "@/components/networking";
import { Card } from "@/components/ui/card";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import { useTranslation } from "react-i18next";

export function UserSpendCard({ user }: { user: UserInfoV2Response }) {
  const { t } = useTranslation();
  const effectiveBudget = user.effective_max_budget ?? user.max_budget;
  const budgetLimit =
    effectiveBudget === null ? t("users.userInfo.unlimited") : `$${formatNumberWithCommas(effectiveBudget, 2)}`;

  return (
    <Card className="block p-6">
      <p>
        {t("users.userInfo.overview.spend")}
        <BudgetPeriodTooltip
          spend={user.spend}
          maxBudget={effectiveBudget}
          budgetDuration={user.budget_duration}
          budgetResetAt={user.budget_reset_at}
        />
      </p>
      <div className="mt-2">
        <h3 className="text-lg font-medium">${formatNumberWithCommas(user.spend || 0, 2)}</h3>
        <p>{t("users.userInfo.overview.budgetOf", { budget: budgetLimit })}</p>
        {user.rollover_credit != null && (
          <p>{t("users.userInfo.overview.rolledOver", { amount: formatNumberWithCommas(user.rollover_credit, 2) })}</p>
        )}
      </div>
    </Card>
  );
}
