import type { Team } from "@/components/key_team_helpers/key_list";
import { isProxyAdminRole, isUserTeamAdminForSingleTeam } from "@/utils/roles";

type TeamPermissionInfo = Pick<Team, "team_id" | "members_with_roles">;

export const canModifyKey = ({
  userRole,
  userId,
  key,
  teams,
}: {
  userRole: string | null | undefined;
  userId: string | null | undefined;
  key: { team_id: string | null | undefined; user_id: string | null | undefined };
  teams: readonly TeamPermissionInfo[] | null | undefined;
}): boolean => {
  if (isProxyAdminRole(userRole ?? "")) return true;

  const team = teams?.find((candidate) => candidate.team_id === key.team_id);
  const isTeamAdmin = Boolean(userId && team && isUserTeamAdminForSingleTeam(team.members_with_roles, userId));
  const hasMatchingUserId = userId != null && key.user_id != null && userId === key.user_id;
  const isNotInternalViewer = userRole !== "Internal Viewer";
  const isKeyOwner = hasMatchingUserId && isNotInternalViewer;

  return isTeamAdmin || isKeyOwner;
};
