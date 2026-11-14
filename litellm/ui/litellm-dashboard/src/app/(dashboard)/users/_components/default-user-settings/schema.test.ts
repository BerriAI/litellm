import { describe, expect, it } from "vitest";

import { defaultUserSettingsSchema, type DefaultUserSettingsFormValues } from "./schema";

const values = (overrides: Partial<DefaultUserSettingsFormValues> = {}): DefaultUserSettingsFormValues => ({
  user_role: "internal_user",
  max_budget: "",
  budget_duration: "",
  models: [],
  teams: [],
  ...overrides,
});

const issuesFor = (input: DefaultUserSettingsFormValues) => {
  const result = defaultUserSettingsSchema.safeParse(input);
  return result.success ? [] : result.error.issues.map((issue) => ({ path: issue.path, message: issue.message }));
};

describe("defaultUserSettingsSchema", () => {
  it("accepts settings with no default teams", () => {
    expect(defaultUserSettingsSchema.safeParse(values()).success).toBe(true);
  });

  it("rejects a team row that has no team selected", () => {
    expect(issuesFor(values({ teams: [{ team_id: "", max_budget_in_team: "", user_role: "user" }] }))).toStrictEqual([
      { path: ["teams", 0, "team_id"], message: "Select a team" },
    ]);
  });

  it("rejects the same team appearing twice", () => {
    expect(
      issuesFor(
        values({
          teams: [
            { team_id: "team-1", max_budget_in_team: "", user_role: "user" },
            { team_id: "team-1", max_budget_in_team: "", user_role: "admin" },
          ],
        }),
      ),
    ).toStrictEqual([{ path: ["teams", 1, "team_id"], message: "This team is already listed" }]);
  });

  it("does not treat two blank rows as duplicates of each other", () => {
    const issues = issuesFor(
      values({
        teams: [
          { team_id: "", max_budget_in_team: "", user_role: "user" },
          { team_id: "", max_budget_in_team: "", user_role: "user" },
        ],
      }),
    );

    expect(issues.map((issue) => issue.message)).toStrictEqual(["Select a team", "Select a team"]);
  });

  it("rejects non-numeric budgets on the form and on a team row", () => {
    expect(issuesFor(values({ max_budget: "lots" }))).toStrictEqual([
      { path: ["max_budget"], message: "Must be a non-negative number" },
    ]);
    expect(
      issuesFor(values({ teams: [{ team_id: "team-1", max_budget_in_team: "-5", user_role: "user" }] })),
    ).toStrictEqual([{ path: ["teams", 0, "max_budget_in_team"], message: "Must be a non-negative number" }]);
  });
});
