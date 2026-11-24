import { describe, expect, it } from "vitest";
import { canModifyKey } from "./keyPermissions";

const key = { user_id: "owner", team_id: "key-team" };

describe("canModifyKey", () => {
  it.each(["Admin", "proxy_admin"])("allows %s to modify another user's key", (role) => {
    expect(canModifyKey(key, [], "administrator", role)).toBe(true);
  });

  it("allows an internal user to modify their own key but not another user's key", () => {
    expect(canModifyKey(key, [], "owner", "Internal User")).toBe(true);
    expect(canModifyKey(key, [], "another-user", "Internal User")).toBe(false);
  });

  it("denies an internal viewer even when they own the key", () => {
    expect(canModifyKey(key, [], "owner", "Internal Viewer")).toBe(false);
  });

  it.each([
    { teamId: "key-team", memberRole: "admin", allowed: true },
    { teamId: "other-team", memberRole: "admin", allowed: false },
    { teamId: "key-team", memberRole: "user", allowed: false },
  ] as const)(
    "checks the $memberRole membership on $teamId before allowing a team key change",
    ({ teamId, memberRole, allowed }) => {
      const teams = [
        {
          team_id: teamId,
          members_with_roles: [{ user_id: "member", user_email: "member@example.com", role: memberRole }],
        },
      ];
      expect(canModifyKey(key, teams, "member", "Internal User")).toBe(allowed);
    },
  );

  it("does not grant access through missing teams or a missing session", () => {
    expect(canModifyKey(key, undefined, "another-user", "Internal User")).toBe(false);
    expect(canModifyKey(key, null, null, null)).toBe(false);
  });
});
