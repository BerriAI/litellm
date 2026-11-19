import { describe, expect, it } from "vitest";
import { PROFILE_KIND, newProfileDraft } from "./additionalAuthProfiles";
import { buildEditServerPayload, type EditServerFormValues, type EditServerUiState } from "./editServerPayload";
import { baseUi } from "./editServerPayload.differential.cases";

const values: EditServerFormValues = {
  server_name: "srv",
  alias: "srv_alias",
  description: "a server",
  transport: "http",
  auth_type: "none",
};

const build = (ui: Partial<EditServerUiState> = {}) => buildEditServerPayload(values, { ...baseUi, ...ui });

const payloadOf = (result: ReturnType<typeof build>) => {
  expect(result.kind).toBe("ok");
  if (result.kind !== "ok") throw new Error("unreachable");
  return result.payload;
};

const profile = { ...newProfileDraft(), id: "wiki-bots", kind: PROFILE_KIND.TOKEN, authentication_token: "tok" };

describe("buildEditServerPayload additional_auth", () => {
  it("leaves the key out when profiles were not touched, so the stored ones stay", () => {
    const untouched = JSON.stringify(payloadOf(build()));
    expect(JSON.stringify(payloadOf(build({ additionalAuth: null })))).toBe(untouched);
    expect(Object.hasOwn(payloadOf(build({ additionalAuth: null })), "additional_auth")).toBe(false);
  });

  it("sends the edited list, which replaces the stored profiles", () => {
    expect(payloadOf(build({ additionalAuth: [profile] })).additional_auth).toEqual([
      { id: "wiki-bots", auth_type: "token", authentication_token: "tok" },
    ]);
  });

  it("sends an empty list when the admin removed every profile", () => {
    expect(payloadOf(build({ additionalAuth: [] })).additional_auth).toEqual([]);
  });

  it("refuses a profile whose id collides with another one in the form, ignoring case", () => {
    const result = build({ additionalAuth: [profile, { ...profile, id: "WIKI-bots" }] });
    expect(result).toEqual({ kind: "invalid_additional_auth", profileId: "wiki-bots" });
  });
});
