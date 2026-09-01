import { describe, expect, it } from "vitest";

import { getAuditObjectName } from "./auditObjectLabel";

describe("getAuditObjectName", () => {
  it("reads the model name from before_value for a deleted model", () => {
    expect(
      getAuditObjectName({
        table_name: "LiteLLM_ProxyModelTable",
        before_value: { model_name: "gpt-5.6", model_id: "abc" },
        updated_values: {},
      }),
    ).toBe("gpt-5.6");
  });

  it("prefers updated_values over before_value", () => {
    expect(
      getAuditObjectName({
        table_name: "LiteLLM_TeamTable",
        before_value: { team_alias: "old" },
        updated_values: { team_alias: "new" },
      }),
    ).toBe("new");
  });

  it("maps each table to its human name field", () => {
    expect(
      getAuditObjectName({
        table_name: "LiteLLM_VerificationToken",
        before_value: { key_alias: "k" },
        updated_values: {},
      }),
    ).toBe("k");
    expect(
      getAuditObjectName({
        table_name: "LiteLLM_UserTable",
        before_value: { user_email: "a@b.c" },
        updated_values: {},
      }),
    ).toBe("a@b.c");
    expect(
      getAuditObjectName({
        table_name: "LiteLLM_OrganizationTable",
        before_value: { organization_alias: "org" },
        updated_values: {},
      }),
    ).toBe("org");
  });

  it("returns null for unknown tables, missing or blank names", () => {
    expect(
      getAuditObjectName({ table_name: "LiteLLM_Config", before_value: { model_name: "x" }, updated_values: {} }),
    ).toBeNull();
    expect(
      getAuditObjectName({ table_name: "LiteLLM_ProxyModelTable", before_value: {}, updated_values: {} }),
    ).toBeNull();
    expect(
      getAuditObjectName({
        table_name: "LiteLLM_ProxyModelTable",
        before_value: { model_name: "  " },
        updated_values: {},
      }),
    ).toBeNull();
  });
});
