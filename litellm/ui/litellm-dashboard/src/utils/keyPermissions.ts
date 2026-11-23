import type { KeyResponse, Team } from "@/components/key_team_helpers/key_list";
import { isProxyAdminRole, isUserTeamAdminForSingleTeam } from "@/utils/roles";

export const canModifyKey = (
  key: Pick<KeyResponse, "user_id" | "team_id">,
  teams: readonly Pick<Team, "team_id" | "members_with_roles">[] | null | undefined,
  userId: string | null,
  userRole: string | null,
): boolean => {
  const canModifyOwnKey = userId === key.user_id && userRole !== "Internal Viewer";
  return (
    isProxyAdminRole(userRole ?? "") ||
    isUserTeamAdminForSingleTeam(
      teams?.find((team) => team.team_id === key.team_id)?.members_with_roles ?? null,
      userId ?? "",
    ) ||
    canModifyOwnKey
  );
};
