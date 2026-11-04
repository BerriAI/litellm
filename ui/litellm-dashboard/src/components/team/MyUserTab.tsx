import { BudgetResetSchedule, describeBudgetResetSchedule } from "@/utils/budgetUtils";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import { SimpleTooltip } from "@/components/ui/tooltip";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import type { TFunction } from "i18next";
import { CircleHelp } from "lucide-react";
import React from "react";
import { useTranslation } from "react-i18next";
import { useMyTeamMember } from "./useMyTeamMember";

interface MyUserTabProps {
  teamId: string;
}

const labelWithTooltip = (t: TFunction, label: string, tooltip: string) => (
  <span className="flex items-center gap-1 text-muted-foreground">
    {label}
    <SimpleTooltip content={tooltip}>
      <CircleHelp className="size-4" aria-label={t("teams.myUserTab.infoAriaLabel", { label })} />
    </SimpleTooltip>
  </span>
);

const formatNumber = (value: number | null | undefined, digits = 4): string => {
  if (value === null || value === undefined) return "0";
  return formatNumberWithCommas(value, digits);
};

const formatRateLimit = (t: TFunction, value: number | null | undefined): string => {
  if (value === null || value === undefined) return t("teams.myUserTab.unlimited");
  return formatNumberWithCommas(value, 0);
};

function describePeriodLine(t: TFunction, schedule: BudgetResetSchedule): string {
  if (schedule.period) return t("teams.myUserTab.periodWithReset", { period: schedule.period, reset: schedule.label });
  if (schedule.resetAt) return t("teams.myUserTab.resetsOn", { reset: schedule.label });
  return schedule.label;
}

export default function MyUserTab({ teamId }: MyUserTabProps) {
  const { t } = useTranslation();
  const { data, isLoading, error } = useMyTeamMember(teamId);

  if (isLoading) {
    return (
      <Card>
        <CardContent className="text-muted-foreground">{t("teams.myUserTab.loading")}</CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardContent className="text-destructive">
          {error instanceof Error ? error.message : t("teams.myUserTab.loadFailed")}
        </CardContent>
      </Card>
    );
  }

  if (!data) {
    return (
      <Card>
        <CardContent className="text-muted-foreground">{t("teams.myUserTab.empty")}</CardContent>
      </Card>
    );
  }

  const budgetTable = data.litellm_budget_table ?? null;
  const maxBudget = budgetTable?.max_budget ?? null;
  const spend = data.spend ?? 0;
  const totalSpend = data.total_spend ?? 0;
  const tpmLimit = budgetTable?.tpm_limit ?? null;
  const rpmLimit = budgetTable?.rpm_limit ?? null;
  const budgetReset = describeBudgetResetSchedule(budgetTable?.budget_duration, budgetTable?.budget_reset_at);
  const allowedModels = budgetTable?.allowed_models ?? null;

  return (
    <div className="flex w-full flex-col gap-4">
      <Card>
        <CardContent>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 md:grid-cols-3">
            <div>
              <span className="text-muted-foreground">{t("teams.myUserTab.user")}</span>
              <div className="mt-1 font-semibold">{data.user_email || data.user_id}</div>
              <span className="font-mono text-xs text-muted-foreground">{data.user_id}</span>
            </div>
            <div>
              <span className="text-muted-foreground">{t("teams.myUserTab.teamRole")}</span>
              <div className="mt-1">
                <Badge variant={data.role === "admin" ? "default" : "secondary"}>{data.role || "user"}</Badge>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <Card>
          <CardContent>
            {labelWithTooltip(t, t("teams.myUserTab.currentPeriodSpend.title"), budgetReset.tooltip)}
            <div className="mt-2">
              <h3 className="text-2xl font-semibold">${formatNumber(spend, 4)}</h3>
              <span className="text-muted-foreground">
                {t("teams.myUserTab.currentPeriodSpend.ofBudget", {
                  budget: maxBudget === null ? t("teams.myUserTab.unlimited") : `$${formatNumber(maxBudget, 4)}`,
                })}
              </span>
            </div>
            <div className="mt-1 text-muted-foreground">{describePeriodLine(t, budgetReset)}</div>
          </CardContent>
        </Card>

        <Card>
          <CardContent>
            {labelWithTooltip(t, t("teams.myUserTab.rateLimits.title"), t("teams.myUserTab.rateLimits.tooltip"))}
            <div className="mt-2">
              <span>{t("teams.myUserTab.rateLimits.tpm", { value: formatRateLimit(t, tpmLimit) })}</span>
              <br />
              <span>{t("teams.myUserTab.rateLimits.rpm", { value: formatRateLimit(t, rpmLimit) })}</span>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent>
            {labelWithTooltip(t, t("teams.myUserTab.totalSpend.title"), t("teams.myUserTab.totalSpend.tooltip"))}
            <h4 className="mt-2 text-xl font-semibold">${formatNumber(totalSpend, 4)}</h4>
          </CardContent>
        </Card>

        <Card>
          <CardContent>
            {labelWithTooltip(t, t("teams.myUserTab.modelScope.title"), t("teams.myUserTab.modelScope.tooltip"))}
            <div className="mt-2">
              {allowedModels && allowedModels.length > 0 ? (
                <div className="flex flex-wrap gap-1">
                  {allowedModels.map((m) => (
                    <Badge key={m} variant="secondary">
                      {m}
                    </Badge>
                  ))}
                </div>
              ) : (
                <span>{t("teams.myUserTab.modelScope.allTeamModels")}</span>
              )}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
