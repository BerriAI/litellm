import { useLanguage } from "@/contexts/LanguageContext";
import type { Translate } from "@/i18n/language";
import { describeBudgetResetSchedule } from "@/utils/budgetUtils";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import { SimpleTooltip } from "@/components/ui/tooltip";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { CircleHelp } from "lucide-react";
import React from "react";
import { useMyTeamMember } from "./useMyTeamMember";

interface MyUserTabProps {
  teamId: string;
}

const labelWithTooltip = (label: string, tooltip: string) => (
  <span className="flex items-center gap-1 text-muted-foreground">
    {label}
    <SimpleTooltip content={tooltip}>
      <CircleHelp className="size-4" aria-label={label} />
    </SimpleTooltip>
  </span>
);

const formatNumber = (value: number | null | undefined, digits = 4): string => {
  if (value === null || value === undefined) return "0";
  return formatNumberWithCommas(value, digits);
};

const formatRateLimit = (value: number | null | undefined, t: Translate): string => {
  if (value === null || value === undefined) return t("Unlimited");
  return formatNumberWithCommas(value, 0);
};

export default function MyUserTab({ teamId }: MyUserTabProps) {
  const { t } = useLanguage();
  const { data, isLoading, error } = useMyTeamMember(teamId);

  if (isLoading) {
    return (
      <Card>
        <CardContent className="text-muted-foreground">{t("Loading your membership info…")}</CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardContent className="text-destructive">
          {error instanceof Error ? error.message : t("Failed to load your membership info for this team.")}
        </CardContent>
      </Card>
    );
  }

  if (!data) {
    return (
      <Card>
        <CardContent className="text-muted-foreground">
          {t("No membership info available for the current user in this team.")}
        </CardContent>
      </Card>
    );
  }

  const budgetTable = data.litellm_budget_table ?? null;
  const maxBudget = budgetTable?.max_budget ?? null;
  const spend = data.spend ?? 0;
  const totalSpend = data.total_spend ?? 0;
  const tpmLimit = budgetTable?.tpm_limit ?? null;
  const rpmLimit = budgetTable?.rpm_limit ?? null;
  const budgetReset = describeBudgetResetSchedule(budgetTable?.budget_duration, budgetTable?.budget_reset_at, t);
  const allowedModels = budgetTable?.allowed_models ?? null;

  return (
    <div className="flex w-full flex-col gap-4">
      <Card>
        <CardContent>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 md:grid-cols-3">
            <div>
              <span className="text-muted-foreground">{t("User")}</span>
              <div className="mt-1 font-semibold">{data.user_email || data.user_id}</div>
              <span className="font-mono text-xs text-muted-foreground">{data.user_id}</span>
            </div>
            <div>
              <span className="text-muted-foreground">{t("Team Role")}</span>
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
            {labelWithTooltip(t("Current Cycle Spend (USD)"), budgetReset.tooltip)}
            <div className="mt-2">
              <h3 className="text-2xl font-semibold">${formatNumber(spend, 4)}</h3>
              <span className="text-muted-foreground">
                {t("of")} {maxBudget === null ? t("Unlimited") : `$${formatNumber(maxBudget, 4)}`}
              </span>
            </div>
            <div className="mt-1 text-muted-foreground">
              {budgetReset.resetAt ? `${t("Resets")} ${budgetReset.label}` : budgetReset.label}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent>
            {labelWithTooltip(t("Rate Limits"), t("Your per-member rate limits within this team."))}
            <div className="mt-2">
              <span>
                {t("TPM:")} {formatRateLimit(tpmLimit, t)}
              </span>
              <br />
              <span>
                {t("RPM:")} {formatRateLimit(rpmLimit, t)}
              </span>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent>
            {labelWithTooltip(
              t("Total Spend (USD)"),
              t(
                "Cumulative spend across budget cycles within this team. This is a lower bound: lifetime tracking began 2026-04-21, so cycles completed before then are not included. It is never less than Current Cycle Spend.",
              ),
            )}
            <h4 className="mt-2 text-xl font-semibold">${formatNumber(totalSpend, 4)}</h4>
          </CardContent>
        </Card>

        <Card>
          <CardContent>
            {labelWithTooltip(t("Model Scope"), t("Models you can access within this team."))}
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
                <span>{t("All Team Models")}</span>
              )}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
