import { TableCell } from "@tremor/react";
import { Team } from "@/components/key_team_helpers/key_list";
import TeamRoleBadge from "@/app/(dashboard)/teams/components/TeamsTable/YourRoleCell/TeamRoleBadge";

interface YourRoleCellProps {
  team: Team;
  userId: string | null;
}

const getUserRole = (team: Team, userId: string | null): string | null => {
  if (!userId) return null;
  const member = team.members_with_roles?.find((m) => m.user_id === userId);
  return member?.role ?? null;
};

const YourRoleCell = ({ team, userId }: YourRoleCellProps) => {
  const roleBadge = TeamRoleBadge(getUserRole(team, userId));

  return <TableCell>{roleBadge}</TableCell>;
};

export default YourRoleCell;
