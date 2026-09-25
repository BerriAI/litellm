import { formatBudgetReset } from "@/utils/budgetUtils";
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
      <CircleHelp className="size-4" aria-label={`${label} information`} />
    </SimpleTooltip>
  </span>
);

const formatNumber = (value: number | null | undefined, digits = 4): string => {
  if (value === null || value === undefined) return "0";
  return formatNumberWithCommas(value, digits);
};

const formatRateLimit = (value: number | null | undefined): string => {
  if (value === null || value === undefined) return "Unlimited";
  return formatNumberWithCommas(value, 0);
};

export default function MyUserTab({ teamId }: MyUserTabProps) {
  const { data, isLoading, error } = useMyTeamMember(teamId);

  if (isLoading) {
    return (
      <Card>
        <CardContent className="text-muted-foreground">Loading your membership info…</CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardContent className="text-destructive">
          {error instanceof Error ? error.message : "Failed to load your membership info for this team."}
        </CardContent>
      </Card>
    );
  }

  if (!data) {
    return (
      <Card>
        <CardContent className="text-muted-foreground">
          No membership info available for the current user in this team.
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
  const budgetReset = formatBudgetReset(budgetTable?.budget_reset_at);
  const allowedModels = budgetTable?.allowed_models ?? null;

  return (
    <div className="flex w-full flex-col gap-4">
      <Card>
        <CardContent>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 md:grid-cols-3">
            <div>
              <span className="text-muted-foreground">User</span>
              <div className="mt-1 font-semibold">{data.user_email || data.user_id}</div>
              <span className="font-mono text-xs text-muted-foreground">{data.user_id}</span>
            </div>
            <div>
              <span className="text-muted-foreground">Team Role</span>
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
            {labelWithTooltip(
              "Current Cycle Spend (USD)",
              "Spend for the current budget cycle. Resets to $0 when the budget window rolls over.",
            )}
            <div className="mt-2">
              <h3 className="text-2xl font-semibold">${formatNumber(spend, 4)}</h3>
              <span className="text-muted-foreground">
                of {maxBudget === null ? "Unlimited" : `$${formatNumber(maxBudget, 4)}`}
              </span>
            </div>
            {budgetReset && <div className="mt-1 text-muted-foreground">Resets {budgetReset}</div>}
          </CardContent>
        </Card>

        <Card>
          <CardContent>
            {labelWithTooltip("Rate Limits", "Your per-member rate limits within this team.")}
            <div className="mt-2">
              <span>TPM: {formatRateLimit(tpmLimit)}</span>
              <br />
              <span>RPM: {formatRateLimit(rpmLimit)}</span>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent>
            {labelWithTooltip("Total Spend (USD)", "Cumulative spend across all budget cycles within this team.")}
            <h4 className="mt-2 text-xl font-semibold">${formatNumber(totalSpend, 4)}</h4>
          </CardContent>
        </Card>

        <Card>
          <CardContent>
            {labelWithTooltip("Model Scope", "Models you can access within this team.")}
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
                <span>All Team Models</span>
              )}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
