import { z } from "zod/v4";

const isBlank = (value: string): boolean => value.trim() === "";

interface DefaultUserSettingsValidationMessages {
  nonNegative: string;
  selectTeam: string;
  duplicateTeam: string;
}

const DEFAULT_VALIDATION_MESSAGES: DefaultUserSettingsValidationMessages = {
  nonNegative: "Must be a non-negative number",
  selectTeam: "Select a team",
  duplicateTeam: "This team is already listed",
};

const createAmountOrEmpty = (message: string) =>
  z.string().refine((value) => isBlank(value) || (Number.isFinite(Number(value)) && Number(value) >= 0), message);

const createDefaultTeamRowSchema = (messages: DefaultUserSettingsValidationMessages) =>
  z.object({
    team_id: z.string().min(1, messages.selectTeam),
    max_budget_in_team: createAmountOrEmpty(messages.nonNegative),
    user_role: z.enum(["user", "admin"]),
  });

const defaultTeamRowSchema = createDefaultTeamRowSchema(DEFAULT_VALIDATION_MESSAGES);

export type DefaultTeamRowValues = z.output<typeof defaultTeamRowSchema>;

export const EMPTY_TEAM_ROW: DefaultTeamRowValues = { team_id: "", max_budget_in_team: "", user_role: "user" };

export const createDefaultUserSettingsSchema = (messages = DEFAULT_VALIDATION_MESSAGES) => {
  const defaultUserSettingsShape = {
    user_role: z.string(),
    max_budget: createAmountOrEmpty(messages.nonNegative),
    budget_duration: z.string(),
    models: z.array(z.string()),
    teams: z.array(createDefaultTeamRowSchema(messages)),
  };

  return z.object(defaultUserSettingsShape).superRefine((values, ctx) => {
    const repeatedRows = values.teams.flatMap((team, index) =>
      team.team_id !== "" && values.teams.findIndex((other) => other.team_id === team.team_id) < index ? [index] : [],
    );

    repeatedRows.forEach((index) =>
      ctx.addIssue({
        code: "custom",
        message: messages.duplicateTeam,
        path: ["teams", index, "team_id"],
      }),
    );
  });
};

export const defaultUserSettingsSchema = createDefaultUserSettingsSchema();

export type DefaultUserSettingsFormValues = z.output<typeof defaultUserSettingsSchema>;
