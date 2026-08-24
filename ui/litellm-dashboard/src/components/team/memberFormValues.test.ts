import { describe, expect, it } from "vitest";
import {
  buildMemberFormData,
  buildMemberFormValues,
  emptyMemberFormValues,
  memberFieldNames,
  type MemberFieldsConfig,
} from "./memberFormValues";

const roleOptions = [
  { label: "Admin", value: "admin" },
  { label: "User", value: "user" },
];

const teamConfig: MemberFieldsConfig = {
  roleOptions,
  showEmail: true,
  showUserId: true,
  additionalFields: [
    { name: "max_budget_in_team", label: "Budget", type: "numerical" },
    { name: "budget_duration", label: "Reset", type: "budget-duration" },
    { name: "tpm_limit", label: "TPM", type: "numerical" },
    { name: "rpm_limit", label: "RPM", type: "numerical" },
    { name: "allowed_models", label: "Models", type: "multi-select" },
  ],
};

const orgConfig: MemberFieldsConfig = { roleOptions, showEmail: true, showUserId: true };

describe("memberFieldNames", () => {
  it("lists the rendered fields in submit order", () => {
    expect(memberFieldNames(teamConfig)).toStrictEqual([
      "user_email",
      "user_id",
      "role",
      "max_budget_in_team",
      "budget_duration",
      "tpm_limit",
      "rpm_limit",
      "allowed_models",
    ]);
  });

  it.each([
    [{ roleOptions, showEmail: false, showUserId: true }, ["user_id", "role"]],
    [{ roleOptions, showEmail: true, showUserId: false }, ["user_email", "role"]],
    [{ roleOptions }, ["role"]],
  ])("drops the identity fields the config hides", (config, expected) => {
    expect(memberFieldNames(config as MemberFieldsConfig)).toStrictEqual(expected);
  });
});

describe("buildMemberFormValues", () => {
  it("seeds only the rendered fields, never the rest of the member record", () => {
    expect(
      buildMemberFormValues(
        "edit",
        {
          user_email: "a@b.com",
          user_id: "u1",
          role: "user",
          max_budget_in_team: 12.5,
          budget_duration: "24h",
          tpm_limit: 100,
          rpm_limit: 20,
          allowed_models: ["gpt-4o"],
          spend: 3.21,
          team_id: "t1",
          created_at: "2026-01-01T00:00:00Z",
        },
        teamConfig,
      ),
    ).toStrictEqual({
      user_email: "a@b.com",
      user_id: "u1",
      role: "user",
      max_budget_in_team: 12.5,
      budget_duration: "24h",
      tpm_limit: 100,
      rpm_limit: 20,
      allowed_models: ["gpt-4o"],
    });
  });

  it("keeps a stored budget or limit of 0 as 0 because only null means unlimited", () => {
    expect(
      buildMemberFormValues(
        "edit",
        { user_email: "a@b.com", user_id: "u1", role: "user", max_budget_in_team: 0, tpm_limit: 0, rpm_limit: 0 },
        teamConfig,
      ),
    ).toStrictEqual({
      user_email: "a@b.com",
      user_id: "u1",
      role: "user",
      max_budget_in_team: 0,
      budget_duration: null,
      tpm_limit: 0,
      rpm_limit: 0,
      allowed_models: [],
    });
  });

  it("collapses missing budgets and limits to null and a missing model list to an empty array", () => {
    const unlimitedMember = {
      user_email: "a@b.com",
      user_id: "u1",
      role: "user",
      max_budget_in_team: null,
      budget_duration: null,
      tpm_limit: null,
      rpm_limit: null,
      allowed_models: [],
    };
    expect(
      buildMemberFormValues("edit", { user_email: "a@b.com", user_id: "u1", role: "user" }, teamConfig),
    ).toStrictEqual(unlimitedMember);
  });

  it("falls back to the configured default role when the member has none", () => {
    expect(buildMemberFormValues("edit", { user_id: "u1", role: "" }, { ...orgConfig, defaultRole: "user" }).role).toBe(
      "user",
    );
  });

  it("seeds add mode with the default role and leaves every other field unset", () => {
    expect(buildMemberFormValues("add", null, { ...orgConfig, defaultRole: "user" })).toStrictEqual({
      user_email: undefined,
      user_id: undefined,
      role: "user",
    });
  });

  it("falls back to the first role option when no default is configured", () => {
    expect(buildMemberFormValues("add", null, orgConfig).role).toBe("admin");
  });

  it("ignores the member record in add mode", () => {
    expect(buildMemberFormValues("add", { user_email: "a@b.com", role: "admin" }, orgConfig)).toStrictEqual({
      user_email: undefined,
      user_id: undefined,
      role: "admin",
    });
  });

  it("treats edit mode without a record as add mode", () => {
    expect(buildMemberFormValues("edit", null, { ...orgConfig, defaultRole: "user" }).role).toBe("user");
  });
});

describe("emptyMemberFormValues", () => {
  it("clears every rendered field to the empty value its control understands", () => {
    expect(emptyMemberFormValues(orgConfig)).toStrictEqual({
      user_email: "",
      user_id: "",
      role: "",
    });
  });

  it("clears numeric, duration and multi-select fields to values their controls accept", () => {
    expect(
      emptyMemberFormValues({
        ...orgConfig,
        additionalFields: [
          { name: "max_budget_in_team", label: "Budget", type: "numerical" },
          { name: "budget_duration", label: "Reset", type: "budget-duration" },
          { name: "allowed_models", label: "Models", type: "multi-select" },
        ],
      }),
    ).toStrictEqual({
      user_email: "",
      user_id: "",
      role: "",
      max_budget_in_team: null,
      budget_duration: null,
      allowed_models: [],
    });
  });
});

describe("buildMemberFormData", () => {
  it("trims strings and keeps everything else by reference", () => {
    const models = ["gpt-4o"];

    expect(
      buildMemberFormData({
        user_email: "  a@b.com  ",
        user_id: "  u1  ",
        role: "user",
        max_budget_in_team: 12.5,
        allowed_models: models,
      }),
    ).toStrictEqual({
      user_email: "a@b.com",
      user_id: "u1",
      role: "user",
      max_budget_in_team: 12.5,
      allowed_models: models,
    });
  });

  it.each(["max_budget_in_team", "tpm_limit", "rpm_limit"])("turns a blank %s into null", (key) => {
    expect(buildMemberFormData({ [key]: "   " })[key]).toBeNull();
  });

  it.each(["user_email", "user_id", "budget_duration"])("leaves a blank %s as an empty string", (key) => {
    expect(buildMemberFormData({ [key]: "   " })[key]).toBe("");
  });

  it("keeps a typed numeric value as the raw string it was typed as", () => {
    expect(buildMemberFormData({ max_budget_in_team: "42.56" }).max_budget_in_team).toBe("42.56");
  });

  it("passes null and undefined through untouched", () => {
    expect(buildMemberFormData({ tpm_limit: null, user_email: undefined })).toStrictEqual({
      tpm_limit: null,
      user_email: undefined,
    });
  });

  it("preserves field order", () => {
    expect(Object.keys(buildMemberFormData({ user_email: "a@b.com", user_id: "u1", role: "user" }))).toStrictEqual([
      "user_email",
      "user_id",
      "role",
    ]);
  });
});
