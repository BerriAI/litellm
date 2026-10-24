import type { Member } from "@/components/networking";
import { hasProxyWideSpendView, isUserTeamAdminForSingleTeam } from "@/utils/roles";

export interface ClientUsageTeam {
  readonly team_id: string;
  readonly team_alias?: string | null;
  readonly members_with_roles?: readonly Member[] | null;
}

export interface ClientUsageScopeOption {
  value: string;
  label: string;
}

export interface ClientUsageScopeInput {
  userId: string | null;
  userRole: string | null;
  teams: readonly ClientUsageTeam[] | null | undefined;
  requestedTeamId: string | null;
  requestedUserId: string | null;
  ownUsage?: boolean;
}

export interface ClientUsageScope {
  userId: string | null;
  teamId: string | null;
}

export interface ResolvedClientUsageScope {
  scope: ClientUsageScope | null;
  administeredTeams: readonly ClientUsageScopeOption[];
  members: readonly ClientUsageScopeOption[];
  canChooseGlobalUser: boolean;
  error: string | null;
}

const memberLabel = (member: Member): string => {
  const name = member.user_alias?.trim();
  const email = member.user_email?.trim();
  if (name && email && name !== email) return `${name} (${email})`;
  if (name) return name;
  return email || member.user_id || "";
};

const memberOptions = (team: ClientUsageTeam): ClientUsageScopeOption[] => {
  const members = team.members_with_roles ?? [];
  const identified = members.filter((member): member is Member & { user_id: string } => Boolean(member.user_id));
  const unique = identified.filter(
    (member, index) => identified.findIndex((row) => row.user_id === member.user_id) === index,
  );
  return unique.map((member) => ({ value: member.user_id, label: memberLabel(member) }));
};

export const resolveClientUsageScope = ({
  userId,
  userRole,
  teams,
  requestedTeamId,
  requestedUserId,
  ownUsage = false,
}: ClientUsageScopeInput): ResolvedClientUsageScope => {
  const proxyAdmin = Boolean(userId) && hasProxyWideSpendView(userRole);
  const availableTeams = userId
    ? (teams ?? []).filter(
        (team) => proxyAdmin || isUserTeamAdminForSingleTeam(team.members_with_roles ?? null, userId),
      )
    : [];
  const administeredTeams = availableTeams.map((team) => ({
    value: team.team_id,
    label: team.team_alias || team.team_id,
  }));
  const base: ResolvedClientUsageScope = {
    scope: null,
    administeredTeams,
    members: [],
    canChooseGlobalUser: proxyAdmin && !ownUsage && !requestedTeamId,
    error: null,
  };
  if (!userId) return { ...base, error: "Sign in to view client usage" };
  if (ownUsage) return { ...base, scope: { userId, teamId: null } };

  if (requestedTeamId) {
    const selectedTeam = availableTeams.find((team) => team.team_id === requestedTeamId);
    if (!selectedTeam) return { ...base, error: "Choose a team you administer" };
    const members = memberOptions(selectedTeam);
    const memberId = requestedUserId || null;
    if (memberId && !members.some((member) => member.value === memberId)) {
      return { ...base, members, error: "Choose a member of the selected team" };
    }
    return { ...base, members, scope: { teamId: selectedTeam.team_id, userId: memberId } };
  }

  if (proxyAdmin) return { ...base, scope: { teamId: null, userId: requestedUserId || null } };
  if (requestedUserId && requestedUserId !== userId) {
    return { ...base, error: "Choose an administered team before viewing another member's usage" };
  }
  return { ...base, scope: { teamId: null, userId } };
};
