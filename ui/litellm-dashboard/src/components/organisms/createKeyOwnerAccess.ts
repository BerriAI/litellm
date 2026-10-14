import type { SearchSelectOption } from "@/components/shared/SearchSelect";
import { TEAM_ADMIN_CREATE_MEMBER_KEYS_PERMISSION } from "@/components/team/teamAdminEditAccess";
import { isProxyAdminRole, isUserTeamAdminForSingleTeam } from "@/utils/roles";
import type { Member } from "../networking";

export type AnotherUserScope = "any_user" | "team_members" | "none";

export interface TeamMembers {
  readonly members_with_roles?: Member[] | null;
}

export interface KeyOwnerViewer {
  readonly userRole: string | null;
  readonly userId: string | null;
  readonly teamAdminEditableFields: readonly string[];
  readonly teams: readonly TeamMembers[] | null;
}

export const administersTeam = (team: TeamMembers | null, userId: string | null): boolean =>
  userId !== null && isUserTeamAdminForSingleTeam(team?.members_with_roles ?? null, userId);

export const anotherUserScope = ({
  userRole,
  userId,
  teamAdminEditableFields,
  teams,
}: KeyOwnerViewer): AnotherUserScope => {
  if (isProxyAdminRole(userRole ?? "")) return "any_user";
  if (!teamAdminEditableFields.includes(TEAM_ADMIN_CREATE_MEMBER_KEYS_PERMISSION)) return "none";
  return (teams ?? []).some((team) => administersTeam(team, userId)) ? "team_members" : "none";
};

export const effectiveKeyOwner = (requestedOwner: string, scope: AnotherUserScope): string =>
  requestedOwner === "another_user" && scope === "none" ? "you" : requestedOwner;

export const teamMemberOwnerOptions = (team: TeamMembers | null, viewerId: string | null): SearchSelectOption[] => {
  if (!administersTeam(team, viewerId)) return [];
  return (team?.members_with_roles ?? []).flatMap((member) =>
    member.user_id && member.user_id !== viewerId
      ? [
          {
            value: member.user_id,
            label: member.user_email ? `${member.user_email} (${member.user_id})` : member.user_id,
          },
        ]
      : [],
  );
};

export const teamMemberOwnerPlaceholder = (
  options: readonly SearchSelectOption[],
  team: TeamMembers | null,
): string => {
  if (options.length > 0) return "Select a member of the team";
  return team ? "This team has no other members" : "Select a team first";
};

export const isTeamMemberOwner = (ownerId: unknown, team: TeamMembers | null, viewerId: string | null): boolean =>
  teamMemberOwnerOptions(team, viewerId).some((option) => option.value === ownerId);
