import { describe, expect, it } from "vitest";

import {
  parseSupportedTeamAdminEditableFields,
  parseTeamAdminEditableFields,
  parseTeamEditAccess,
  teamAdminFieldLabel,
  teamAdminSettingsChanges,
} from "./teamAdminEditAccess";

describe("teamAdminFieldLabel", () => {
  it.each([
    ["tpm_limit", "Tokens per minute Limit (TPM)"],
    ["rpm_limit", "Requests per minute Limit (RPM)"],
    ["max_budget", "Max Budget (USD)"],
    ["soft_budget", "Soft Budget (USD)"],
    ["raise_max_budget", "Raise the team's max budget"],
    ["projects", "Create and update projects"],
    ["team_member_budget", "Member Default Budget (USD)"],
    ["team_member_budget_duration", "Member Default Budget Duration"],
    ["team_member_tpm_limit", "Member Default TPM Limit"],
    ["team_member_rpm_limit", "Member Default RPM Limit"],
    ["create_member_keys", "Create keys for team members"],
  ])("names %s the way the team settings form does", (field, label) => {
    expect(teamAdminFieldLabel(field)).toBe(label);
  });

  it("falls back to the raw field name for a field the dashboard has no label for", () => {
    expect(teamAdminFieldLabel("team_alias")).toBe("team_alias");
  });
});

describe("teamAdminSettingsChanges", () => {
  const tpmEnabled = new Set(["tpm_limit"]);
  const stored = { tpm_limit: 1000 };

  it.each([
    ["a typed number string", "5000", 5000],
    ["a number", 1200, 1200],
    ["zero", "0", 0],
    ["an emptied input", "", null],
    ["whitespace", "  ", null],
    ["no limit", null, null],
    ["an unset value", undefined, null],
  ])("sends tpm_limit changed to %s", (_label, tpm_limit, expected) => {
    expect(teamAdminSettingsChanges({ tpm_limit }, stored, tpmEnabled)).toStrictEqual({ tpm_limit: expected });
  });

  it.each([
    ["the stored number", 1000, { tpm_limit: 1000 }],
    ["the stored number typed back in", "1000", { tpm_limit: 1000 }],
    ["an emptied input over no stored limit", "", { tpm_limit: null }],
    ["an unset value over no stored limit", undefined, { tpm_limit: null }],
  ])("sends nothing for %s", (_label, tpm_limit, initialValues) => {
    expect(teamAdminSettingsChanges({ tpm_limit }, initialValues, tpmEnabled)).toStrictEqual({});
  });

  it("leaves tpm_limit out when the proxy did not enable it for team admins", () => {
    expect(teamAdminSettingsChanges({ tpm_limit: "5000" }, stored, new Set(["max_budget"]))).toStrictEqual({});
  });

  const allStored = { tpm_limit: 1000, rpm_limit: 10, max_budget: 20 };

  it("sends every enabled field that changed and skips the ones that did not", () => {
    const values = { tpm_limit: "1000", rpm_limit: "50", max_budget: "12.5" };
    const enabled = new Set(["tpm_limit", "rpm_limit", "max_budget"]);

    expect(teamAdminSettingsChanges(values, allStored, enabled)).toStrictEqual({ rpm_limit: 50, max_budget: 12.5 });
  });

  it("sends a cleared max budget as no budget", () => {
    expect(teamAdminSettingsChanges({ max_budget: "" }, allStored, new Set(["max_budget"]))).toStrictEqual({
      max_budget: null,
    });
  });

  it("leaves out changed fields the proxy did not enable", () => {
    const values = { tpm_limit: "5000", rpm_limit: "50", max_budget: "5" };

    expect(teamAdminSettingsChanges(values, allStored, new Set(["rpm_limit"]))).toStrictEqual({ rpm_limit: 50 });
  });
});

describe("teamAdminSettingsChanges for team member defaults", () => {
  const stored = {
    team_member_budget: 10,
    team_member_budget_duration: "30d",
    team_member_tpm_limit: 100,
    team_member_rpm_limit: 5,
  };
  const enabled = new Set([
    "team_member_budget",
    "team_member_budget_duration",
    "team_member_tpm_limit",
    "team_member_rpm_limit",
  ]);

  const durationEnabled = new Set(["team_member_budget_duration"]);

  it("sends every changed member default and skips the unchanged ones", () => {
    const values = {
      team_member_budget: "25",
      team_member_budget_duration: "7d",
      team_member_tpm_limit: "100",
      team_member_rpm_limit: "9",
    };

    expect(teamAdminSettingsChanges(values, stored, enabled)).toStrictEqual({
      team_member_budget: 25,
      team_member_budget_duration: "7d",
      team_member_rpm_limit: 9,
    });
  });

  it("sends a never-resetting duration as null", () => {
    expect(teamAdminSettingsChanges({ team_member_budget_duration: null }, stored, durationEnabled)).toStrictEqual({
      team_member_budget_duration: null,
    });
  });

  it("sends a never-resetting duration for a team without member defaults yet", () => {
    expect(teamAdminSettingsChanges({ team_member_budget_duration: null }, {}, durationEnabled)).toStrictEqual({
      team_member_budget_duration: null,
    });
  });

  it.each([
    ["the stored duration", "30d", stored],
    ["an untouched duration on a team with a stored duration", undefined, stored],
    ["an untouched duration on a team without member defaults", undefined, {}],
    ["a stored never-resetting duration picked again", null, { team_member_budget_duration: null }],
  ])("sends no duration for %s", (_label, team_member_budget_duration, initialValues) => {
    expect(teamAdminSettingsChanges({ team_member_budget_duration }, initialValues, durationEnabled)).toStrictEqual({});
  });

  it("leaves out member defaults the proxy did not enable", () => {
    const values = { team_member_budget: "25", team_member_budget_duration: "7d" };

    expect(teamAdminSettingsChanges(values, stored, new Set(["max_budget"]))).toStrictEqual({});
  });
});

describe("teamAdminSettingsChanges for the soft budget", () => {
  const stored = { soft_budget: 50 };
  const enabled = new Set(["soft_budget"]);

  it("sends a changed soft budget as a number", () => {
    expect(teamAdminSettingsChanges({ soft_budget: "40" }, stored, enabled)).toStrictEqual({ soft_budget: 40 });
  });

  it("sends a cleared soft budget as no soft budget", () => {
    expect(teamAdminSettingsChanges({ soft_budget: "" }, stored, enabled)).toStrictEqual({ soft_budget: null });
  });

  it("leaves out an unchanged soft budget", () => {
    expect(teamAdminSettingsChanges({ soft_budget: "50" }, stored, enabled)).toStrictEqual({});
  });

  it("leaves out a changed soft budget when the proxy did not enable it", () => {
    expect(teamAdminSettingsChanges({ soft_budget: "40" }, stored, new Set(["max_budget"]))).toStrictEqual({});
  });
});

describe("parseTeamAdminEditableFields", () => {
  it("returns the configured list", () => {
    expect(parseTeamAdminEditableFields({ team_admin_editable_team_fields: ["tpm_limit", "rpm_limit"] })).toEqual([
      "tpm_limit",
      "rpm_limit",
    ]);
  });

  it.each([
    ["no values yet", undefined],
    ["setting missing", {}],
    ["setting is null", { team_admin_editable_team_fields: null }],
    ["setting is a string", { team_admin_editable_team_fields: "tpm_limit" }],
    ["list holds a non-string", { team_admin_editable_team_fields: ["tpm_limit", 7] }],
  ])("fails closed to an empty list when %s", (_label, values) => {
    expect(parseTeamAdminEditableFields(values)).toEqual([]);
  });
});

describe("parseSupportedTeamAdminEditableFields", () => {
  it("reads the enum the proxy advertises on the setting's items schema", () => {
    const schema = {
      properties: {
        team_admin_editable_team_fields: {
          type: "array",
          items: { type: "string", enum: ["max_budget", "tpm_limit"] },
        },
      },
    };
    expect(parseSupportedTeamAdminEditableFields(schema)).toEqual(["max_budget", "tpm_limit"]);
  });

  it.each([
    ["schema not loaded", undefined],
    ["property absent", { properties: {} }],
    ["items has no enum", { properties: { team_admin_editable_team_fields: { items: { type: "string" } } } }],
    ["enum is not a string list", { properties: { team_admin_editable_team_fields: { items: { enum: [1] } } } }],
  ])("returns no supported fields when %s", (_label, schema) => {
    expect(parseSupportedTeamAdminEditableFields(schema)).toEqual([]);
  });
});

describe("parseTeamEditAccess", () => {
  it.each([
    ["unrestricted", { kind: "unrestricted" }],
    ["team_admin_disabled", { kind: "team_admin_disabled" }],
    ["none", { kind: "none" }],
  ])("passes the proxy's %s verdict through", (_kind, verdict) => {
    expect(parseTeamEditAccess(verdict)).toEqual(verdict);
  });

  it("hands a team admin the fields the proxy enabled", () => {
    expect(parseTeamEditAccess({ kind: "team_admin", editable_fields: ["tpm_limit"] })).toEqual({
      kind: "team_admin",
      editableFields: new Set(["tpm_limit"]),
      mayRaiseMaxBudget: false,
    });
  });

  it.each([
    ["the proxy allows raising", { may_raise_max_budget: true }, true],
    ["the proxy forbids raising", { may_raise_max_budget: false }, false],
    ["an older proxy omits the flag", {}, false],
    ["the flag is not a boolean", { may_raise_max_budget: "yes" }, false],
  ])("lets a team admin raise the max budget only when %s", (_label, flag, mayRaiseMaxBudget) => {
    expect(parseTeamEditAccess({ kind: "team_admin", editable_fields: ["max_budget"], ...flag })).toEqual({
      kind: "team_admin",
      editableFields: new Set(["max_budget"]),
      mayRaiseMaxBudget,
    });
  });

  it.each([
    ["the proxy sent nothing", undefined],
    ["the kind is unknown", { kind: "owner" }],
    ["a team admin verdict lacks its field list", { kind: "team_admin" }],
    ["the field list holds a non-string", { kind: "team_admin", editable_fields: [7] }],
  ])("fails closed to no access when %s", (_label, value) => {
    expect(parseTeamEditAccess(value)).toEqual({ kind: "none" });
  });
});
