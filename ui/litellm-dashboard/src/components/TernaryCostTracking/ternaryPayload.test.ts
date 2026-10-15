import { describe, expect, it } from "vitest";

import {
  buildCreatePayload,
  buildUpdatePayload,
  createSchema,
  EMPTY_TERNARY_FORM_VALUES,
  formValuesFromSettings,
  updateSchema,
  type TernaryFormValues,
} from "./ternaryPayload";

const filled: TernaryFormValues = {
  api_key: " tern-key ",
  connection_id: " conn-1 ",
  base_url: "https://api.ternary.app ",
  frequency: "daily",
  interval_seconds: "",
};

const issuesFor = (schema: typeof createSchema | typeof updateSchema, values: TernaryFormValues) => {
  const result = schema.safeParse(values);
  return result.success ? [] : result.error.issues.map((issue) => issue.path.join("."));
};

describe("buildCreatePayload", () => {
  it("trims every field and sends no interval for a daily export", () => {
    expect(buildCreatePayload(filled)).toEqual({
      api_key: "tern-key",
      connection_id: "conn-1",
      base_url: "https://api.ternary.app",
      frequency: "daily",
      interval_seconds: null,
    });
  });

  it("sends the interval as a number for an interval export", () => {
    expect(buildCreatePayload({ ...filled, frequency: "interval", interval_seconds: "300" }).interval_seconds).toBe(
      300,
    );
  });

  it("drops a typed interval when the frequency is daily", () => {
    expect(buildCreatePayload({ ...filled, interval_seconds: "300" }).interval_seconds).toBeNull();
  });
});

describe("buildUpdatePayload", () => {
  it("omits a blank API key so the stored one is kept", () => {
    expect(buildUpdatePayload({ ...filled, api_key: "  " })).not.toHaveProperty("api_key");
  });

  it("sends a new API key when one is typed", () => {
    expect(buildUpdatePayload(filled).api_key).toBe("tern-key");
  });
});

describe("form schemas", () => {
  it("requires an API key only when creating", () => {
    const withoutKey = { ...filled, api_key: "" };
    expect(issuesFor(createSchema, withoutKey)).toEqual(["api_key"]);
    expect(issuesFor(updateSchema, withoutKey)).toEqual([]);
  });

  it("requires a positive whole number of seconds for an interval export", () => {
    const interval = { ...filled, frequency: "interval" as const };
    expect(issuesFor(createSchema, { ...interval, interval_seconds: "" })).toEqual(["interval_seconds"]);
    expect(issuesFor(createSchema, { ...interval, interval_seconds: "0" })).toEqual(["interval_seconds"]);
    expect(issuesFor(createSchema, { ...interval, interval_seconds: "1.5" })).toEqual(["interval_seconds"]);
    expect(issuesFor(createSchema, { ...interval, interval_seconds: "60" })).toEqual([]);
  });

  it("rejects a blank connection ID or base URL", () => {
    expect(issuesFor(createSchema, { ...filled, connection_id: " ", base_url: "" })).toEqual([
      "connection_id",
      "base_url",
    ]);
  });

  it("starts a new integration on the default base URL and daily exports", () => {
    expect(issuesFor(createSchema, { ...EMPTY_TERNARY_FORM_VALUES, api_key: "k", connection_id: "c" })).toEqual([]);
  });
});

describe("formValuesFromSettings", () => {
  it("prefills everything except the API key", () => {
    expect(
      formValuesFromSettings({
        api_key_masked: "te****ey",
        connection_id: "conn-1",
        base_url: "https://eu.example.com",
        frequency: "interval",
        interval_seconds: 120,
        status: "configured",
      }),
    ).toEqual({
      api_key: "",
      connection_id: "conn-1",
      base_url: "https://eu.example.com",
      frequency: "interval",
      interval_seconds: "120",
    });
  });
});
