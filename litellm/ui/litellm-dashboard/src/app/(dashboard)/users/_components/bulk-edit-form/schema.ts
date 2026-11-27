import { z } from "zod";

export const bulkUserTeamSchema = z.object({
  selectedTeams: z.array(z.string()),
  teamBudget: z.custom<number>((value) => typeof value === "number").nullable(),
  addToTeams: z.boolean(),
  updateAllUsers: z.boolean(),
});

export const bulkUserTeamDefaults = {
  selectedTeams: [],
  teamBudget: null,
  addToTeams: false,
  updateAllUsers: false,
};
