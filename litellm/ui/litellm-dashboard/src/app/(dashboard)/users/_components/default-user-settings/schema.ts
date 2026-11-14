import { z } from "zod";

import i18n from "@/i18n/i18n";

const isBlank = (value: string): boolean => value.trim() === "";

const amountOrEmpty = z
  .string()
  .refine((value) => isBlank(value) || (Number.isFinite(Number(value)) && Number(value) >= 0), {
    error: () => i18n.t("users.defaultSettings.errors.nonNegative"),
  });

const defaultTeamRowSchema = z.object({
  team_id: z
    .string()
    .nullable()
    .pipe(
      z
        .string({ error: () => i18n.t("users.defaultSettings.errors.selectTeam") })
        .min(1, { error: () => i18n.t("users.defaultSettings.errors.selectTeam") }),
    ),
  max_budget_in_team: amountOrEmpty,
  user_role: z.enum(["user", "admin"]),
});

export type DefaultTeamRowValues = z.input<typeof defaultTeamRowSchema>;

export const EMPTY_TEAM_ROW: DefaultTeamRowValues = { team_id: null, max_budget_in_team: "", user_role: "user" };

const defaultUserSettingsShape = {
  user_role: z.string(),
  max_budget: amountOrEmpty,
  budget_duration: z.string(),
  models: z.array(z.string()),
  teams: z.array(defaultTeamRowSchema),
};

export const defaultUserSettingsSchema = z.object(defaultUserSettingsShape).superRefine((values, ctx) => {
  const repeatedRows = values.teams.flatMap((team, index) =>
    team.team_id !== "" && values.teams.findIndex((other) => other.team_id === team.team_id) < index ? [index] : [],
  );

  repeatedRows.forEach((index) =>
    ctx.addIssue({
      code: "custom",
      message: i18n.t("users.defaultSettings.errors.duplicateTeam"),
      path: ["teams", index, "team_id"],
    }),
  );
});

export type DefaultUserSettingsFormValues = z.input<typeof defaultUserSettingsSchema>;
export type DefaultUserSettingsSubmitValues = z.output<typeof defaultUserSettingsSchema>;
