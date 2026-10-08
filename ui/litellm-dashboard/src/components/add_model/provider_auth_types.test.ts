import { describe, expect, it } from "vitest";

import {
  PROVIDER_AUTH_TYPES,
  authTypeFieldKeys,
  authTypesFor,
  hiddenAuthFieldKeys,
  inferAuthTypeId,
  type ProviderAuthType,
} from "./provider_auth_types";
import { Providers } from "../provider_info_helpers";

describe("provider auth types", () => {
  it("resolves provider enum keys and display names", () => {
    expect(authTypesFor("MICROSOFT_365_COPILOT")).toEqual(PROVIDER_AUTH_TYPES.MICROSOFT_365_COPILOT);
    expect(authTypesFor(Providers.MICROSOFT_365_COPILOT)).toEqual(PROVIDER_AUTH_TYPES.MICROSOFT_365_COPILOT);
    expect(authTypesFor(null)).toEqual([]);
    expect(authTypesFor("unknown")).toEqual([]);
  });

  it("returns the unique field keys for every auth type", () => {
    expect(authTypeFieldKeys("MICROSOFT_365_COPILOT")).toEqual([
      "token_exchange_endpoint",
      "token_exchange_profile",
      "client_id",
      "client_secret",
      "token_exchange_scope",
      "token_exchange_audience",
      "api_key",
    ]);
  });

  it("infers auth types from required fields and falls back when only optional defaults are present", () => {
    const authTypes = authTypesFor("MICROSOFT_365_COPILOT");
    const delegatedValues: Record<string, unknown> = {
      token_exchange_profile: "jwt_bearer_obo",
      token_exchange_scope: "https://graph.microsoft.com/.default",
      api_key: "delegated-token",
    };
    const emptyValues: Record<string, unknown> = {
      token_exchange_endpoint: "",
      client_id: null,
      client_secret: undefined,
      token_exchange_profile: "jwt_bearer_obo",
      token_exchange_scope: "https://graph.microsoft.com/.default",
      api_key: "",
    };

    expect(inferAuthTypeId(authTypes, delegatedValues)).toBe("static_token");
    expect(inferAuthTypeId(authTypes, emptyValues)).toBe("oauth_token_exchange");
    expect(inferAuthTypeId([], {})).toBe("");
  });

  it("hides only non-selected keys not shared with the selected type", () => {
    const authTypes: readonly ProviderAuthType[] = [
      {
        id: "first",
        label: "First",
        description: "First auth type",
        fieldKeys: ["shared", "first_key"],
        requiredFieldKeys: ["first_key"],
      },
      {
        id: "second",
        label: "Second",
        description: "Second auth type",
        fieldKeys: ["shared", "second_key"],
        requiredFieldKeys: ["second_key"],
      },
      {
        id: "third",
        label: "Third",
        description: "Third auth type",
        fieldKeys: ["third_key"],
        requiredFieldKeys: ["third_key"],
      },
    ];

    expect(hiddenAuthFieldKeys(authTypes, "first")).toEqual(["second_key", "third_key"]);
  });

  it("defaults GitHub Copilot to shared device login and recognises a stored per-user credential", () => {
    const authTypes = authTypesFor("GITHUB_COPILOT");

    expect(inferAuthTypeId(authTypes, { api_key: "" })).toBe("shared_device_login");
    expect(inferAuthTypeId(authTypes, { github_copilot_auth_type: "per_user_oauth" })).toBe("per_user_oauth");
    expect(hiddenAuthFieldKeys(authTypes, "per_user_oauth")).toEqual(["api_base", "api_key"]);
    expect(hiddenAuthFieldKeys(authTypes, "shared_device_login")).toEqual(["github_copilot_auth_type"]);
  });
});
