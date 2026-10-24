import { z } from "zod";

import { USER_ROLES } from "../userRoles";

const BUDGET_DURATION = /^\d+[dhmwy]$|^\d+mo$/;

const userEmail = z
  .string()
  .min(1, { error: "Email is required", abort: true })
  .refine((email) => email.includes("@") && email.includes("."), "Invalid email format (must contain @ and domain)");

const userRole = z
  .string()
  .min(1, "Role is required")
  .pipe(
    z.enum(USER_ROLES, {
      error: (issue) => `Invalid role "${String(issue.input)}". Must be one of: ${USER_ROLES.join(", ")}`,
    }),
  );

const maxBudget = z
  .string()
  .refine((budget) => budget === "" || !Number.isNaN(parseFloat(budget)), {
    error: (issue) => `Max budget "${String(issue.input)}" must be a number`,
    abort: true,
  })
  .refine((budget) => budget === "" || parseFloat(budget) > 0, "Max budget must be greater than 0");

const budgetDuration = z.string().refine((duration) => duration === "" || BUDGET_DURATION.test(duration), {
  error: (issue) => `Invalid budget duration format "${String(issue.input)}". Use format like "30d", "1mo", "2w", "6h"`,
});

const unknownTeams = (teams: string, knownTeamIds: readonly string[]): string[] =>
  teams
    .split(",")
    .map((team) => team.trim())
    .filter((team) => !knownTeamIds.includes(team));

const teamList = (knownTeamIds: readonly string[]) =>
  z.string().superRefine((teams, ctx) => {
    if (teams === "" || knownTeamIds.length === 0) return;
    const unknown = unknownTeams(teams, knownTeamIds);
    if (unknown.length > 0) ctx.addIssue({ code: "custom", message: `Unknown team(s): ${unknown.join(", ")}` });
  });

export const csvRowSchema = (knownTeamIds: readonly string[]) => {
  const rowShape = {
    user_email: userEmail,
    user_role: userRole,
    max_budget: maxBudget.optional(),
    budget_duration: budgetDuration.optional(),
    teams: teamList(knownTeamIds).optional(),
  };
  return z.object(rowShape);
};
