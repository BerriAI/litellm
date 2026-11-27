export function toMemberMoveBody({ teams }: { teams: string[] }) {
  return { parent_team_ids: teams };
}
