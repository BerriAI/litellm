import { describe, expect, it } from "vitest";

import { defaultTeamSettingsSchema, type DefaultTeamSettingsFormValues } from "./schema";

const values = (overrides: Partial<DefaultTeamSettingsFormValues> = {}): DefaultTeamSettingsFormValues => ({
  max_budget: "",
  budget_duration: "",
  tpm_limit: "",
  rpm_limit: "",
  models: [],
  team_member_permissions: [],
  ...overrides,
});

const issuesFor = (input: DefaultTeamSettingsFormValues) => {
  const result = defaultTeamSettingsSchema.safeParse(input);
  return result.success ? [] : result.error.issues.map((issue) => ({ path: issue.path, message: issue.message }));
};

describe("defaultTeamSettingsSchema", () => {
  it("accepts fully blank settings", () => {
    expect(defaultTeamSettingsSchema.safeParse(values()).success).toBe(true);
  });

  it("accepts a decimal budget and whole-number rate limits", () => {
    const input = values({
      max_budget: "100.5",
      tpm_limit: "1000",
      rpm_limit: "50",
      models: ["gpt-5.2"],
      team_member_permissions: ["/key/generate"],
    });

    expect(defaultTeamSettingsSchema.safeParse(input).success).toBe(true);
  });

  it("rejects a non-numeric or negative budget", () => {
    expect(issuesFor(values({ max_budget: "lots" }))).toStrictEqual([
      { path: ["max_budget"], message: "Must be a non-negative number" },
    ]);
    expect(issuesFor(values({ max_budget: "-5" }))).toStrictEqual([
      { path: ["max_budget"], message: "Must be a non-negative number" },
    ]);
  });

  it("rejects fractional rate limits", () => {
    expect(issuesFor(values({ tpm_limit: "12.5" }))).toStrictEqual([
      { path: ["tpm_limit"], message: "Must be a non-negative whole number" },
    ]);
    expect(issuesFor(values({ rpm_limit: "0.5" }))).toStrictEqual([
      { path: ["rpm_limit"], message: "Must be a non-negative whole number" },
    ]);
  });

  it("rejects a permission that is not offered to teams, even when the backend enum knows it", () => {
    expect(issuesFor(values({ team_member_permissions: ["/key/health"] }))).toStrictEqual([
      { path: ["team_member_permissions", 0], message: "Unknown permission" },
    ]);
  });
});
