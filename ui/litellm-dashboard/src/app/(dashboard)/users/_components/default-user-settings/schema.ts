import { z } from "zod";
import { englishTranslate, type Translate } from "@/i18n/language";

const isBlank = (value: string): boolean => value.trim() === "";

const amountOrEmpty = (t: Translate) =>
  z
    .string()
    .refine(
      (value) => isBlank(value) || (Number.isFinite(Number(value)) && Number(value) >= 0),
      t("Must be a non-negative number"),
    );

const defaultTeamRowSchema = (t: Translate) =>
  z.object({
    team_id: z
      .string()
      .nullable()
      .pipe(z.string({ error: t("Select a team") }).min(1, t("Select a team"))),
    max_budget_in_team: amountOrEmpty(t),
    user_role: z.enum(["user", "admin"]),
  });

export type DefaultTeamRowValues = z.input<ReturnType<typeof defaultTeamRowSchema>>;

export const EMPTY_TEAM_ROW: DefaultTeamRowValues = { team_id: null, max_budget_in_team: "", user_role: "user" };

export const buildDefaultUserSettingsSchema = (t: Translate = englishTranslate) => {
  const shape = {
    user_role: z.string(),
    max_budget: amountOrEmpty(t),
    budget_duration: z.string(),
    models: z.array(z.string()),
    teams: z.array(defaultTeamRowSchema(t)),
  };
  return z.object(shape).superRefine((values, ctx) => {
    const repeatedRows = values.teams.flatMap((team, index) =>
      team.team_id !== "" && values.teams.findIndex((other) => other.team_id === team.team_id) < index ? [index] : [],
    );

    repeatedRows.forEach((index) =>
      ctx.addIssue({
        code: "custom",
        message: t("This team is already listed"),
        path: ["teams", index, "team_id"],
      }),
    );
  });
};

export const defaultUserSettingsSchema = buildDefaultUserSettingsSchema();

export type DefaultUserSettingsFormValues = z.input<typeof defaultUserSettingsSchema>;
export type DefaultUserSettingsSubmitValues = z.output<typeof defaultUserSettingsSchema>;
