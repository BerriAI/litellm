import { describe, expect, it } from "vitest";
import { Member } from "@/components/networking";
import { isTeamAdminEditingMemberKey, KEY_BUDGET_FIELDS, teamAdminMemberKeyPayload } from "./teamAdminMemberKeyPayload";

const members = (role: string): Member[] => [{ user_id: "admin-user", role, user_email: null } as unknown as Member];

const baseArgs = {
  userRole: "Internal User",
  userId: "admin-user",
  keyUserId: "member-user",
  keyTeamId: "team-1",
};

describe("isTeamAdminEditingMemberKey", () => {
  it("is false for a proxy admin", () => {
    expect(isTeamAdminEditingMemberKey({ ...baseArgs, userRole: "Admin", teamMembers: members("admin") })).toBe(false);
  });

  it("is false when the caller owns the key", () => {
    expect(isTeamAdminEditingMemberKey({ ...baseArgs, keyUserId: "admin-user", teamMembers: members("admin") })).toBe(
      false,
    );
  });

  it("is false for a personal key with no team", () => {
    expect(isTeamAdminEditingMemberKey({ ...baseArgs, keyTeamId: null, teamMembers: members("admin") })).toBe(false);
  });

  it("is false when the caller is not a team admin", () => {
    expect(isTeamAdminEditingMemberKey({ ...baseArgs, teamMembers: members("user") })).toBe(false);
    expect(isTeamAdminEditingMemberKey({ ...baseArgs, teamMembers: null })).toBe(false);
    expect(isTeamAdminEditingMemberKey({ ...baseArgs, teamMembers: undefined })).toBe(false);
  });

  it("is true for a team admin editing another member's team key", () => {
    expect(isTeamAdminEditingMemberKey({ ...baseArgs, teamMembers: members("admin") })).toBe(true);
  });
});

describe("teamAdminMemberKeyPayload", () => {
  it("keeps only dirty budget fields from the form values plus the key", () => {
    const formValues = {
      key: "sk-1",
      max_budget: 25,
      soft_budget: 10,
      key_alias: "renamed",
      metadata: { tags: ["a"] },
      tpm_limit: null,
    };
    const result = teamAdminMemberKeyPayload(formValues, ["max_budget", "soft_budget"]);
    expect(result).toEqual({
      kind: "ok",
      payload: { key: "sk-1", max_budget: 25, soft_budget: 10 },
    });
  });

  it("drops budget fields present in the form but not dirty", () => {
    const formValues = {
      key: "sk-1",
      max_budget: 25,
      budget_duration: "30d",
      budget_limits: [{ budget_duration: "1d", max_budget: 5 }],
    };
    const result = teamAdminMemberKeyPayload(formValues, ["max_budget"]);
    expect(result).toEqual({ kind: "ok", payload: { key: "sk-1", max_budget: 25 } });
  });

  it("drops budget_duration when it is an empty string but keeps null", () => {
    const cleared = teamAdminMemberKeyPayload({ key: "sk-1", budget_duration: "" }, ["budget_duration"]);
    expect(cleared).toEqual({ kind: "ok", payload: { key: "sk-1" } });
    const kept = teamAdminMemberKeyPayload({ key: "sk-1", budget_duration: null }, ["budget_duration"]);
    expect(kept).toEqual({ kind: "ok", payload: { key: "sk-1", budget_duration: null } });
  });

  it("keeps budget_limits when present", () => {
    const windows = [{ budget_duration: "1d", max_budget: 5 }];
    const result = teamAdminMemberKeyPayload({ key: "sk-1", budget_limits: windows }, ["budget_limits"]);
    expect(result).toEqual({ kind: "ok", payload: { key: "sk-1", budget_limits: windows } });
  });

  it("is blocked when a dirty field is not a budget field, naming it", () => {
    const result = teamAdminMemberKeyPayload({ key: "sk-1", key_alias: "renamed" }, ["key_alias", "max_budget"]);
    expect(result).toEqual({ kind: "blocked", fields: ["key_alias"] });
  });

  it("is ok when every dirty field is a budget field and ignores token/key", () => {
    const result = teamAdminMemberKeyPayload({ key: "sk-1", max_budget: 5 }, ["token", "key", "max_budget"]);
    expect(result).toEqual({ kind: "ok", payload: { key: "sk-1", max_budget: 5 } });
  });

  it("covers exactly the backend budget field set", () => {
    expect([...KEY_BUDGET_FIELDS].sort()).toEqual(
      ["budget_duration", "budget_limits", "max_budget", "soft_budget"].sort(),
    );
  });
});
