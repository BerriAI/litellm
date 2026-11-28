import type { Member } from "@/components/networking";
import type { AddTeamFields } from "./schema";

export const emptyAddTeam: AddTeamFields = { team_id: "", role: "user" };

export const addTeamToMember = ({ role }: AddTeamFields, userId: string): Member => ({
  role,
  user_id: userId,
});
