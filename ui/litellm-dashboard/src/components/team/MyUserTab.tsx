import { formatBudgetReset } from "@/utils/budgetUtils";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import { SimpleTooltip } from "@/components/ui/tooltip";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { CircleHelp } from "lucide-react";
import React, { useState } from "react";
import { type TeamMemberInfo, useMyTeamMember, useUpdateMySelfBudget } from "./useMyTeamMember";

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

const BUDGET_SOURCE_LABELS: Record<NonNullable<TeamMemberInfo["budget_source"]>, string> = {
  team_default: "Team default",
  custom: "Custom",
  self: "Set by you",
  none: "None",
};

function MyLimitEditor({
  teamId,
  selfMaxBudget,
  spend,
}: {
  teamId: string;
  selfMaxBudget: number | null;
  spend: number;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");
  const mutation = useUpdateMySelfBudget(teamId);

  const parsed = draft.trim() === "" ? NaN : Number(draft);
  const draftValid = Number.isFinite(parsed) && parsed >= 0;
  const belowSpend = editing && draftValid && parsed < spend;

  const save = () => {
    if (!draftValid) return;
    mutation.mutate(parsed, {
      onSuccess: () => setEditing(false),
    });
  };

  const clear = () => {
    mutation.mutate(null, {
      onSuccess: () => setEditing(false),
    });
  };

  const startEditing = () => {
    setDraft(selfMaxBudget === null ? "" : String(selfMaxBudget));
    mutation.reset();
    setEditing(true);
  };

  if (!editing) {
    return (
      <div className="mt-2 flex items-center gap-2">
        <span className="text-xl font-semibold" data-testid="my-limit-value">
          {selfMaxBudget === null ? "Not set" : `$${formatNumber(selfMaxBudget, 4)}`}
        </span>
        <Button variant="outline" size="xs" data-testid="edit-my-limit" onClick={startEditing}>
          Edit
        </Button>
        {selfMaxBudget !== null && (
          <Button variant="link" size="xs" data-testid="clear-my-limit" disabled={mutation.isPending} onClick={clear}>
            Clear
          </Button>
        )}
      </div>
    );
  }

  return (
    <div className="mt-2">
      <div className="flex items-center gap-2">
        <Input
          type="number"
          min={0}
          step="any"
          value={draft}
          data-testid="my-limit-input"
          aria-label="My limit"
          className="w-40"
          onChange={(e) => setDraft(e.target.value)}
        />
        <Button size="xs" data-testid="save-my-limit" disabled={!draftValid || mutation.isPending} onClick={save}>
          Save
        </Button>
        <Button
          variant="outline"
          size="xs"
          data-testid="cancel-my-limit"
          onClick={() => {
            mutation.reset();
            setEditing(false);
          }}
        >
          Cancel
        </Button>
      </div>
      {belowSpend && (
        <div className="mt-1 text-amber-600" data-testid="below-spend-warning">
          This is below your current spend of ${formatNumber(spend, 4)}. New requests will be blocked until you raise or
          clear your limit.
        </div>
      )}
      {mutation.isError && (
        <div className="mt-1 text-destructive" data-testid="my-limit-error">
          {mutation.error instanceof Error ? mutation.error.message : "Failed to update your limit."}
        </div>
      )}
    </div>
  );
}

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
  const maxBudget = data.effective_budget ?? null;
  const budgetSource = data.budget_source ?? "none";
  const selfMaxBudget = data.self_max_budget ?? null;
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
              <span className="inline-flex items-center gap-2 text-muted-foreground">
                of {maxBudget === null ? "Unlimited" : `$${formatNumber(maxBudget, 4)}`}
                {budgetSource !== "none" && (
                  <Badge variant={budgetSource === "self" ? "outline" : "secondary"} data-testid="budget-source-badge">
                    {BUDGET_SOURCE_LABELS[budgetSource]}
                  </Badge>
                )}
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
            {labelWithTooltip(
              "My limit",
              "A personal limit you set for yourself. It can only lower your team allocation, never raise it.",
            )}
            <MyLimitEditor teamId={teamId} selfMaxBudget={selfMaxBudget} spend={spend} />
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
