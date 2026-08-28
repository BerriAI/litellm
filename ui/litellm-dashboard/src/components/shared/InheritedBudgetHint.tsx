"use client";

import { SimpleTooltip } from "@/components/ui/tooltip";
import type { Team } from "@/components/key_team_helpers/key_list";
import type { Organization } from "@/components/networking";
import { formatNumberWithCommas } from "@/utils/dataUtils";

export interface InheritedBudgetGate {
  scope: "Team" | "Organization";
  alias: string;
  maxBudget: number;
  budgetDuration: string | null;
}

type TeamBudgetSource = Pick<Team, "team_id" | "team_alias" | "max_budget" | "budget_duration">;
type OrganizationBudgetSource = Pick<Organization, "organization_id" | "organization_alias" | "litellm_budget_table">;

const teamGate = (team: TeamBudgetSource | null | undefined): InheritedBudgetGate | null =>
  team && team.max_budget != null
    ? {
        scope: "Team",
        alias: team.team_alias || team.team_id,
        maxBudget: team.max_budget,
        budgetDuration: team.budget_duration ?? null,
      }
    : null;

const organizationGate = (organization: OrganizationBudgetSource | null | undefined): InheritedBudgetGate | null => {
  const budgetTable: { max_budget?: number | null; budget_duration?: string | null } | null | undefined =
    organization?.litellm_budget_table;
  return organization && budgetTable?.max_budget != null
    ? {
        scope: "Organization",
        alias: organization.organization_alias || organization.organization_id,
        maxBudget: budgetTable.max_budget,
        budgetDuration: budgetTable.budget_duration ?? null,
      }
    : null;
};

export const inheritedBudgetGates = (
  team: TeamBudgetSource | null | undefined,
  organization: OrganizationBudgetSource | null | undefined,
): readonly InheritedBudgetGate[] => [teamGate(team), organizationGate(organization)].filter((gate) => gate !== null);

const formatGate = (gate: InheritedBudgetGate): string =>
  `${gate.scope} ${gate.alias}: $${formatNumberWithCommas(gate.maxBudget, 2)}${gate.budgetDuration ? ` / ${gate.budgetDuration}` : ""}`;

interface InheritedBudgetHintProps {
  gates: readonly InheritedBudgetGate[];
}

export function InheritedBudgetHint({ gates }: InheritedBudgetHintProps) {
  if (gates.length === 0) return null;
  return (
    <SimpleTooltip
      content={
        <div data-testid="inherited-budget-hint" className="flex flex-col gap-1">
          <span>This key has no budget of its own, but its spend still counts toward:</span>
          {gates.map((gate) => (
            <span key={gate.scope}>{formatGate(gate)}</span>
          ))}
        </div>
      }
    />
  );
}
