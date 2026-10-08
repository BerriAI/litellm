import { describe, expect, it } from "vitest";
import { canModifyKey } from "./canModifyKey";

const team = {
  team_id: "team-1",
  members_with_roles: [
    { user_id: "team-admin", role: "admin" },
    { user_id: "team-member", role: "user" },
  ],
};
const proxyAdminContext = {
  userRole: "Admin",
  userId: "admin",
  key: { user_id: "owner", team_id: null },
  teams: [],
};
const teamAdminContext = {
  userRole: "Internal User",
  userId: "team-admin",
  key: { user_id: "team-member", team_id: "team-1" },
  teams: [team],
};
const keyOwnerContext = {
  userRole: "Internal User",
  userId: "owner",
  key: { user_id: "owner", team_id: null },
  teams: [],
};
const internalViewerContext = {
  userRole: "Internal Viewer",
  userId: "owner",
  key: { user_id: "owner", team_id: null },
  teams: [],
};
const unauthorizedContext = {
  userRole: "Internal User",
  userId: "other",
  key: { user_id: "owner", team_id: "team-1" },
  teams: [team],
};

describe("canModifyKey", () => {
  it("allows proxy admins", () => {
    expect(canModifyKey(proxyAdminContext)).toBe(true);
  });

  it("allows admins of the key's team", () => {
    expect(canModifyKey(teamAdminContext)).toBe(true);
  });

  it("allows the key owner unless they are an internal viewer", () => {
    expect(canModifyKey(keyOwnerContext)).toBe(true);
    expect(canModifyKey(internalViewerContext)).toBe(false);
  });

  it("rejects users without access to the key", () => {
    expect(canModifyKey(unauthorizedContext)).toBe(false);
  });
});
