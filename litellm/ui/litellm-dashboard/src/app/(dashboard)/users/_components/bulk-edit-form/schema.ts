import { z } from "zod";

const bulkUserTeamFields = {
  selectedTeams: z.array(z.string()),
  teamBudget: z.custom<number>((value) => typeof value === "number").nullable(),
  addToTeams: z.boolean(),
  updateAllUsers: z.boolean(),
};

export const bulkUserTeamSchema = z.object(bulkUserTeamFields);

export const bulkUserTeamDefaults = {
  selectedTeams: [],
  teamBudget: null,
  addToTeams: false,
  updateAllUsers: false,
};
