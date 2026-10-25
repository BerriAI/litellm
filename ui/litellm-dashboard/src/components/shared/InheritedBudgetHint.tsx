"use client";

import { SimpleTooltip } from "@/components/ui/tooltip";
import type { Team } from "@/components/key_team_helpers/key_list";
import type { Organization } from "@/components/networking";
import { formatNumberWithCommas } from "@/utils/dataUtils";

export interface InheritedBudgetGate {
  scope: "Team" | "Team member" | "Organization" | "User";
  alias: string;
  maxBudget: number;
  budgetDuration: string | null;
  spend?: number | null;
}

export type MemberBudgetRow =
  | {
      max_budget?: number | null;
      budget_duration?: string | null;
      temp_budget_increase?: number | null;
      temp_budget_expiry?: string | null;
    }
  | null
  | undefined;

export interface TeamMemberBudgetSource {
  team_info: {
    team_id: string;
    team_alias?: string | null;
    team_member_budget_table?: MemberBudgetRow;
  };
  team_memberships?:
    | readonly {
        user_id: string;
        spend?: number | null;
        litellm_budget_table?: MemberBudgetRow;
      }[]
    | null;
}

type TeamBudgetSource = Pick<Team, "team_id" | "team_alias" | "max_budget" | "budget_duration">;
type OrganizationBudgetSource = Pick<Organization, "organization_id" | "organization_alias" | "litellm_budget_table">;

export interface UserBudgetSource {
  user_id: string;
  user_alias?: string | null;
  user_email?: string | null;
  max_budget?: number | null;
  budget_duration?: string | null;
}

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

const userGate = (user: UserBudgetSource | null | undefined): InheritedBudgetGate | null =>
  user && user.max_budget != null
    ? {
        scope: "User",
        alias: user.user_alias || user.user_email || user.user_id,
        maxBudget: user.max_budget,
        budgetDuration: user.budget_duration ?? null,
      }
    : null;

export const tempBudgetExpiryMs = (row: MemberBudgetRow): number | null => {
  if (row?.temp_budget_increase == null || row.temp_budget_expiry == null) return null;
  const expiryValue = /(?:Z|[+-]\d{2}:?\d{2})$/i.test(row.temp_budget_expiry)
    ? row.temp_budget_expiry
    : `${row.temp_budget_expiry}Z`;
  return new Date(expiryValue).getTime();
};

const activeIncrease = (row: MemberBudgetRow, now: Date): number => {
  const expiry = tempBudgetExpiryMs(row);
  if (expiry === null || expiry <= now.getTime()) return 0;
  return row?.temp_budget_increase ?? 0;
};

export const teamMemberBudgetGate = (
  teamInfo: TeamMemberBudgetSource | null | undefined,
  userId: string | null | undefined,
  memberLabel?: string | null,
  now: Date = new Date(),
): InheritedBudgetGate | null => {
  if (!teamInfo || !userId) return null;

  const membership = teamInfo.team_memberships?.find((row) => row.user_id === userId);
  const membershipBudget = membership?.litellm_budget_table;
  const teamBudget = teamInfo.team_info.team_member_budget_table;
  const hasMembershipBudget = membershipBudget?.max_budget != null;
  const budgetRow = hasMembershipBudget ? membershipBudget : teamBudget;
  if (!budgetRow || budgetRow.max_budget == null || (!hasMembershipBudget && budgetRow.max_budget <= 0)) {
    return null;
  }

  return {
    scope: "Team member",
    alias: `${memberLabel || userId} in ${teamInfo.team_info.team_alias || teamInfo.team_info.team_id}`,
    maxBudget: budgetRow.max_budget + activeIncrease(membershipBudget, now),
    budgetDuration: budgetRow.budget_duration ?? null,
    spend: membership?.spend ?? 0,
  };
};

export const inheritedBudgetGates = (
  team: TeamBudgetSource | null | undefined,
  organization: OrganizationBudgetSource | null | undefined,
  user?: UserBudgetSource | null,
  teamMember?: InheritedBudgetGate | null,
): readonly InheritedBudgetGate[] =>
  [teamGate(team), teamMember ?? null, organizationGate(organization), userGate(user)].filter((gate) => gate !== null);

export const keyOwnerBudgetSource = (
  key: { team_id?: string | null; user?: UserBudgetSource | null },
  applyUserBudgetToTeamKeys: boolean,
): UserBudgetSource | null => (!key.team_id || applyUserBudgetToTeamKeys ? key.user ?? null : null);

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
