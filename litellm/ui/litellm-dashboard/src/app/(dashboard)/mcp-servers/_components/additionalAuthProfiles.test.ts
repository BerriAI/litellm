import { describe, expect, it } from "vitest";
import {
  PROFILE_KIND,
  additionalAuthEntry,
  draftFromProfile,
  draftsToProfiles,
  firstInvalidProfileId,
  newProfileDraft,
  profileProblem,
  type AdditionalAuthDraft,
} from "./additionalAuthProfiles";

const clientCredentials = (overrides: Partial<AdditionalAuthDraft> = {}): AdditionalAuthDraft => ({
  ...newProfileDraft(),
  id: "tickets-bots",
  client_id: "bot-app",
  client_secret: "s3cret",
  token_url: "https://auth.example.com/token",
  ...overrides,
});

const staticToken = (overrides: Partial<AdditionalAuthDraft> = {}): AdditionalAuthDraft => ({
  ...newProfileDraft(),
  id: "wiki-bots",
  kind: PROFILE_KIND.BEARER_TOKEN,
  authentication_token: "tok",
  ...overrides,
});

describe("draftsToProfiles", () => {
  it("sends a client-credentials profile as oauth2 with the client_credentials flow", () => {
    expect(
      draftsToProfiles([
        clientCredentials({ scopes: " read  write\nadmin ", token_endpoint_auth_method: "client_secret_post" }),
      ]),
    ).toEqual([
      {
        id: "tickets-bots",
        auth_type: "oauth2",
        oauth2_flow: "client_credentials",
        client_id: "bot-app",
        client_secret: "s3cret",
        token_url: "https://auth.example.com/token",
        scopes: ["read", "write", "admin"],
        token_endpoint_auth_method: "client_secret_post",
      },
    ]);
  });

  it("sends a static profile with only its token", () => {
    expect(draftsToProfiles([staticToken({ client_id: "ignored", token_url: "ignored" })])).toEqual([
      { id: "wiki-bots", auth_type: "bearer_token", authentication_token: "tok" },
    ]);
  });

  it("sends secrets exactly as typed and trims only the identifiers around them", () => {
    const [oauth, token] = draftsToProfiles([
      clientCredentials({
        client_id: " bot-app ",
        client_secret: " s3cret ",
        token_url: " https://auth.example.com/token ",
      }),
      staticToken({ authentication_token: " tok " }),
    ]);
    expect(oauth).toMatchObject({
      client_id: "bot-app",
      client_secret: " s3cret ",
      token_url: "https://auth.example.com/token",
    });
    expect(token).toMatchObject({ authentication_token: " tok " });
  });

  it("leaves out blank optional fields instead of sending empty strings", () => {
    const [profile] = draftsToProfiles([clientCredentials({ scopes: "  ", token_endpoint_auth_method: "" })]);
    expect(Object.hasOwn(profile, "scopes")).toBe(false);
    expect(Object.hasOwn(profile, "token_endpoint_auth_method")).toBe(false);
  });

  it("clears the optional settings an admin emptied on a stored profile, since an omitted one is kept", () => {
    const emptied = { isNew: false, client_secret: "", scopes: "", token_endpoint_auth_method: "" };
    const [profile] = draftsToProfiles([clientCredentials(emptied)]);
    expect(profile).toMatchObject({ scopes: [], token_endpoint_auth_method: null });
    expect(Object.hasOwn(profile, "client_secret")).toBe(false);
  });
});

describe("draftFromProfile", () => {
  it("never carries a secret from a loaded profile into the draft", () => {
    const loaded = {
      id: "tickets-bots",
      auth_type: "oauth2",
      oauth2_flow: "client_credentials",
      client_id: "bot-app",
      client_secret: "REDACTED-OR-NOT",
      token_url: "https://auth.example.com/token",
      scopes: ["read", "write"],
      authentication_token: "REDACTED-OR-NOT",
    };
    const draft = draftFromProfile(loaded);
    expect(draft).toMatchObject({ isNew: false, kind: "client_credentials", scopes: "read write" });
    expect(draft.client_secret).toBe("");
    expect(draft.authentication_token).toBe("");
  });

  it("keeps a saved scope that contains a comma when the profile is saved again", () => {
    const loaded = { id: "tickets-bots", auth_type: "oauth2" as const, scopes: ["read:tickets,wiki", "write"] };
    const [profile] = draftsToProfiles([draftFromProfile(loaded)]);
    expect(profile.scopes).toEqual(["read:tickets,wiki", "write"]);
  });

  it("round-trips a loaded static profile without a token so the stored one is kept", () => {
    const draft = draftFromProfile({ id: "wiki-bots", auth_type: "api_key", authentication_token: "x" });
    expect(draftsToProfiles([draft])).toEqual([{ id: "wiki-bots", auth_type: "api_key" }]);
  });
});

describe("profileProblem", () => {
  it.each([
    ["bad characters", "atl bots", "id_format"],
    ["non-ASCII letters", "café", "id_format"],
    ["over 64 characters", "a".repeat(65), "id_format"],
    ["the reserved id in any case", "Default", "id_reserved"],
  ])("rejects %s", (_label, id, expected) => {
    expect(profileProblem(clientCredentials({ id }), [clientCredentials({ id })])).toBe(expected);
  });

  it("accepts the 64-character boundary", () => {
    const draft = clientCredentials({ id: "a".repeat(64) });
    expect(profileProblem(draft, [draft])).toBeNull();
  });

  it("treats ids that differ only by case as duplicates", () => {
    const drafts = [clientCredentials({ id: "TICKETS-Bots" }), staticToken({ id: "tickets-bots" })];
    expect(profileProblem(drafts[0], drafts)).toBe("id_duplicate");
    expect(firstInvalidProfileId(drafts)).toBe("TICKETS-Bots");
  });

  it("requires the credential fields of a new profile but not of a loaded one", () => {
    expect(profileProblem(clientCredentials({ client_secret: "" }), [clientCredentials()])).toBe("missing_field");
    expect(profileProblem(staticToken({ authentication_token: " " }), [staticToken()])).toBe("missing_field");
    const loaded = clientCredentials({ client_secret: "", isNew: false });
    expect(profileProblem(loaded, [loaded])).toBeNull();
  });
});

describe("additionalAuthEntry", () => {
  it("adds no key when profiles were never touched", () => {
    expect(Object.hasOwn(additionalAuthEntry(null, { omitWhenEmpty: false }), "additional_auth")).toBe(false);
    expect(Object.hasOwn(additionalAuthEntry(undefined, { omitWhenEmpty: true }), "additional_auth")).toBe(false);
  });

  it("omits an empty list on create but sends it on edit, where it clears the stored profiles", () => {
    expect(Object.hasOwn(additionalAuthEntry([], { omitWhenEmpty: true }), "additional_auth")).toBe(false);
    expect(additionalAuthEntry([], { omitWhenEmpty: false })).toEqual({ additional_auth: [] });
  });

  it("sends the built profiles when there are some", () => {
    expect(additionalAuthEntry([staticToken()], { omitWhenEmpty: true })).toEqual({
      additional_auth: [{ id: "wiki-bots", auth_type: "bearer_token", authentication_token: "tok" }],
    });
  });
});
