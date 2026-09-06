import { describe, expect, it } from "vitest";

import {
  parseSupportedTeamAdminEditableFields,
  parseTeamAdminEditableFields,
  resolveTeamEditAccess,
} from "./teamAdminEditAccess";

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

describe("resolveTeamEditAccess", () => {
  it("does not restrict callers who edit as proxy or org admin", () => {
    expect(resolveTeamEditAccess(false, { team_admin_editable_team_fields: [] })).toEqual({ kind: "unrestricted" });
  });

  it("disables editing for a team admin when no field is enabled", () => {
    expect(resolveTeamEditAccess(true, { team_admin_editable_team_fields: [] })).toEqual({
      kind: "team_admin_disabled",
    });
    expect(resolveTeamEditAccess(true, undefined)).toEqual({ kind: "team_admin_disabled" });
  });

  it("hands a team admin the enabled fields", () => {
    expect(resolveTeamEditAccess(true, { team_admin_editable_team_fields: ["tpm_limit"] })).toEqual({
      kind: "team_admin",
      editableFields: new Set(["tpm_limit"]),
    });
  });
});
