import { describe, expect, it } from "vitest";
import {
  EMPTY_KILL_SWITCH_FORM,
  buildKillSwitchFromForm,
  parseKillSwitchForForm,
  validateKillSwitchBody,
  type KillSwitchConfig,
  type KillSwitchFormValue,
} from "./kill_switch_config";

const fullConfig: KillSwitchConfig = {
  url: "https://ops.example.com/kill?env=prod",
  method: "DELETE",
  headers: { "X-Env": "prod" },
  query_params: { agent: "billing-bot" },
  body: { reason: "manual stop", force: true },
  auth: { type: "api_key", header_name: "X-Ops-Key", api_key: "k-456" },
};

describe("buildKillSwitchFromForm", () => {
  it("returns undefined when the form never touched the kill switch", () => {
    expect(buildKillSwitchFromForm(undefined)).toBeUndefined();
  });

  it("returns null when the URL is blank so the backend clears the config", () => {
    const blankUrlForm: KillSwitchFormValue = {
      ...EMPTY_KILL_SWITCH_FORM,
      url: "   ",
      auth_type: "bearer",
      auth_token: "t",
    };
    expect(buildKillSwitchFromForm(blankUrlForm)).toBeNull();
  });

  it("builds the full config, dropping rows without a key and parsing the JSON body", () => {
    const fullForm: KillSwitchFormValue = {
      url: " https://ops.example.com/kill?env=prod ",
      method: "DELETE",
      headers: [
        { key: "X-Env", value: "prod" },
        { key: "  ", value: "ignored" },
      ],
      query_params: [{ key: "agent", value: "billing-bot" }],
      body: '{"reason": "manual stop", "force": true}',
      auth_type: "api_key",
      auth_header_name: "X-Ops-Key",
      auth_api_key: "k-456",
    };
    expect(buildKillSwitchFromForm(fullForm)).toEqual(fullConfig);
  });

  it.each([
    [{ auth_type: "none" as const, auth_token: "leftover" }, null],
    [
      { auth_type: "bearer" as const, auth_token: "tok-123" },
      { type: "bearer", token: "tok-123" },
    ],
    [
      { auth_type: "api_key" as const, auth_api_key: "k" },
      { type: "api_key", header_name: "X-API-Key", api_key: "k" },
    ],
    [
      { auth_type: "basic" as const, auth_username: "ops", auth_password: "pw" },
      { type: "basic", username: "ops", password: "pw" },
    ],
  ])("maps auth form fields %j to %j", (authFields, expectedAuth) => {
    const authForm: KillSwitchFormValue = { ...EMPTY_KILL_SWITCH_FORM, url: "https://x.example", ...authFields };
    expect(buildKillSwitchFromForm(authForm)?.auth).toEqual(expectedAuth);
  });

  it("sends an empty body as null and defaults the method to POST", () => {
    const expected: KillSwitchConfig = {
      url: "https://x.example",
      method: "POST",
      headers: {},
      query_params: {},
      body: null,
      auth: null,
    };
    expect(buildKillSwitchFromForm({ url: "https://x.example", body: "  " })).toEqual(expected);
  });
});

describe("validateKillSwitchBody", () => {
  it.each(["", "   ", undefined, '{"a": 1}'])("accepts %j", (text) => {
    expect(validateKillSwitchBody(text)).toBe(true);
  });

  it.each(["[1, 2]", '"text"', "42", "null"])("rejects non-object JSON %s", (text) => {
    expect(validateKillSwitchBody(text)).toBe("Body must be a JSON object");
  });

  it("rejects malformed JSON with the parser message", () => {
    expect(validateKillSwitchBody("{not json")).toMatch(/JSON/);
  });
});

describe("parseKillSwitchForForm", () => {
  it("returns the empty form for a missing config", () => {
    expect(parseKillSwitchForForm(null)).toEqual(EMPTY_KILL_SWITCH_FORM);
    expect(parseKillSwitchForForm(undefined)).toEqual(EMPTY_KILL_SWITCH_FORM);
  });

  it("round-trips a full config through the form representation", () => {
    expect(buildKillSwitchFromForm(parseKillSwitchForForm(fullConfig))).toEqual(fullConfig);
  });

  it("keeps the redacted secret marker in the auth field so the backend restores it", () => {
    const parsed = parseKillSwitchForForm({
      url: "https://x.example",
      method: "POST",
      auth: { type: "bearer", token: "redacted-marker" },
    });
    expect(parsed.auth_type).toBe("bearer");
    expect(parsed.auth_token).toBe("redacted-marker");
    expect(parsed.auth_password).toBe("");
  });
});
