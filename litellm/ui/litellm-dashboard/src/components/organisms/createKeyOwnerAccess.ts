import type { TFunction } from "i18next";
import i18n from "@/i18n/i18n";
import type { SearchSelectOption } from "@/components/shared/SearchSelect";
import { TEAM_ADMIN_CREATE_MEMBER_KEYS_PERMISSION } from "@/components/team/teamAdminEditAccess";
import { isProxyAdminRole, isUserTeamAdminForSingleTeam } from "@/utils/roles";
import type { Member } from "../networking";

const KEY_GENERATE_ROUTE = "/key/generate";
const SERVICE_ACCOUNT_KEY_GENERATE_ROUTE = "/key/service-account/generate";

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

export interface KeyCreationTeam extends TeamMembers {
  readonly team_member_permissions?: readonly string[] | null;
}

export const canCreateKeyInTeam = (
  team: KeyCreationTeam,
  { userRole, userId }: Pick<KeyOwnerViewer, "userRole" | "userId">,
  owner: string = "you",
): boolean => {
  if (isProxyAdminRole(userRole ?? "") || administersTeam(team, userId)) return true;
  return (
    userId !== null &&
    team.members_with_roles?.some((member) => member.user_id === userId) === true &&
    team.team_member_permissions?.includes(
      owner === "service_account" ? SERVICE_ACCOUNT_KEY_GENERATE_ROUTE : KEY_GENERATE_ROUTE,
    ) === true
  );
};

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

export const keyOwnerOption = (user: {
  user_id: string;
  user_email?: string | null;
  user_alias?: string | null;
}): SearchSelectOption => {
  const identity = user.user_email?.trim() || user.user_alias?.trim();
  return { value: user.user_id, label: identity ? `${identity} (${user.user_id})` : user.user_id };
};

export const teamMemberOwnerOptions = (team: TeamMembers | null, viewerId: string | null): SearchSelectOption[] => {
  if (!administersTeam(team, viewerId)) return [];
  return (team?.members_with_roles ?? []).flatMap((member) =>
    member.user_id && member.user_id !== viewerId ? [keyOwnerOption({ ...member, user_id: member.user_id })] : [],
  );
};

export const teamMemberOwnerPlaceholder = (
  options: readonly SearchSelectOption[],
  team: TeamMembers | null,
  t: TFunction = i18n.t.bind(i18n),
): string => {
  if (options.length > 0) return t("createKey.ownerAccess.selectMember");
  return team ? t("createKey.ownerAccess.noOtherMembers") : t("createKey.ownerAccess.selectTeamFirst");
};

export const isTeamMemberOwner = (ownerId: unknown, team: TeamMembers | null, viewerId: string | null): boolean =>
  teamMemberOwnerOptions(team, viewerId).some((option) => option.value === ownerId);
