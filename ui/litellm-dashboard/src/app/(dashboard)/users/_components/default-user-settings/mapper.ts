import { z } from "zod/v4";

import type { components } from "@/lib/http/schema";

import { EMPTY_TEAM_ROW, type DefaultTeamRowValues, type DefaultUserSettingsFormValues } from "./schema";

export type InternalUserSettings = components["schemas"]["InternalUserSettingsResponse"];
export type DefaultInternalUserParams = components["schemas"]["DefaultInternalUserParams"];
type DefaultTeamBody = components["schemas"]["NewUserRequestTeam"];

const teamRowFromServer = z
  .union([
    z.string().transform((teamId): DefaultTeamRowValues => ({ ...EMPTY_TEAM_ROW, team_id: teamId })),
    z
      .object({
        team_id: z.string(),
        max_budget_in_team: z.number().nullish(),
        user_role: z.enum(["user", "admin"]).catch("user"),
      })
      .transform(
        (team): DefaultTeamRowValues => ({
          team_id: team.team_id,
          max_budget_in_team: team.max_budget_in_team?.toString() ?? "",
          user_role: team.user_role,
        }),
      ),
  ])
  .catch(EMPTY_TEAM_ROW);

const serverValuesShape = {
  user_role: z.string().nullish().catch(null),
  max_budget: z.number().nullish().catch(null),
  budget_duration: z.string().nullish().catch(null),
  models: z.array(z.string()).nullish().catch(null),
  teams: z.array(teamRowFromServer).nullish().catch(null),
};

const serverValuesSchema = z.object(serverValuesShape);

export const settingsToForm = (values: InternalUserSettings["values"]): DefaultUserSettingsFormValues => {
  const parsed = serverValuesSchema.parse(values);

  return {
    user_role: parsed.user_role ?? "",
    max_budget: parsed.max_budget?.toString() ?? "",
    budget_duration: parsed.budget_duration ?? "",
    models: parsed.models ?? [],
    teams: parsed.teams ?? [],
  };
};

const DEFAULT_USER_ROLES = ["internal_user", "internal_user_viewer", "proxy_admin", "proxy_admin_viewer"] as const;

const asDefaultUserRole = (raw: string): DefaultInternalUserParams["user_role"] =>
  DEFAULT_USER_ROLES.find((role) => role === raw) ?? null;

const numberOrNull = (raw: string): number | null => (raw.trim() === "" ? null : Number(raw));

const textOrNull = (raw: string): string | null => (raw.trim() === "" ? null : raw);

const listOrNull = <T>(items: readonly T[]): T[] | null => (items.length === 0 ? null : [...items]);

const toTeamBody = (team: DefaultTeamRowValues): DefaultTeamBody => ({
  team_id: team.team_id,
  max_budget_in_team: numberOrNull(team.max_budget_in_team),
  user_role: team.user_role,
});

export const buildBody = (values: DefaultUserSettingsFormValues): DefaultInternalUserParams => ({
  user_role: asDefaultUserRole(values.user_role),
  max_budget: numberOrNull(values.max_budget),
  budget_duration: textOrNull(values.budget_duration),
  models: listOrNull(values.models),
  teams: listOrNull(values.teams.map(toTeamBody)),
});
