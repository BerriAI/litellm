import { describe, expect, it } from "vitest";

import {
  parseSupportedTeamAdminEditableFields,
  parseTeamAdminEditableFields,
  parseTeamEditAccess,
  teamAdminFieldLabel,
  teamAdminSettingsChanges,
} from "./teamAdminEditAccess";

describe("teamAdminFieldLabel", () => {
  it("names tpm_limit the way the team settings form does", () => {
    expect(teamAdminFieldLabel("tpm_limit")).toBe("Tokens per minute Limit (TPM)");
  });

  it("falls back to the raw field name for a field the dashboard has no label for", () => {
    expect(teamAdminFieldLabel("max_budget")).toBe("max_budget");
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
