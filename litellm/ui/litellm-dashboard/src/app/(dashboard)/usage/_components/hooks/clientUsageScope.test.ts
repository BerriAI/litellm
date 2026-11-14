import { describe, expect, it } from "vitest";
import { resolveClientUsageScope, type ClientUsageScopeInput, type ClientUsageTeam } from "./clientUsageScope";

const teams: readonly ClientUsageTeam[] = [
  {
    team_id: "engineering",
    team_alias: "Engineering",
    members_with_roles: [
      { user_id: "john", role: "admin", user_alias: "John", user_email: "john@example.com" },
      { user_id: "alice", role: "user", user_email: "alice@example.com" },
      { user_id: "alice", role: "user", user_email: "duplicate@example.com" },
      { user_id: "id-only", role: "user" },
      { user_id: null, role: "user", user_email: "invited@example.com" },
    ],
  },
  {
    team_id: "design",
    team_alias: "Design",
    members_with_roles: [
      { user_id: "john", role: "admin" },
      { user_id: "bob", role: "user", user_alias: "Bob" },
    ],
  },
  {
    team_id: "external",
    members_with_roles: [
      { user_id: "john", role: "user" },
      { user_id: "outside-admin", role: "admin" },
    ],
  },
];

const defaults: ClientUsageScopeInput = {
  userId: "john",
  userRole: "Internal User",
  teams,
  requestedTeamId: null,
  requestedUserId: null,
};

const scopeFor = (overrides: Partial<ClientUsageScopeInput> = {}) =>
  resolveClientUsageScope({ ...defaults, ...overrides });

describe("resolveClientUsageScope", () => {
  it("should default a signed-in user to personal usage", () => {
    const result = scopeFor({ userId: "alice" });
    expect(result.scope).toEqual({ userId: "alice", teamId: null });
    expect(result.canChooseGlobalUser).toBe(false);
    expect(result.administeredTeams).toEqual([]);
    expect(result.members).toEqual([]);
    expect(result.error).toBeNull();
  });

  it.each(["Admin", "Admin Viewer", "proxy_admin", "proxy_admin_viewer"])(
    "should permit proxy-wide and selected-user read scopes for %s",
    (userRole) => {
      const allUsers = scopeFor({ userRole });
      const selectedUser = scopeFor({ userRole, requestedUserId: "any-proxy-user" });
      expect(allUsers.scope).toEqual({ userId: null, teamId: null });
      expect(allUsers.canChooseGlobalUser).toBe(true);
      expect(allUsers.administeredTeams.map((team) => team.value)).toEqual(["engineering", "design", "external"]);
      expect(selectedUser.scope).toEqual({ userId: "any-proxy-user", teamId: null });
    },
  );

  it.each(["Org Admin", "org_admin", "Internal User", "internal_user", "team_admin"])(
    "should not grant proxy-wide reads from the %s role name",
    (userRole) => {
      const result = scopeFor({ userRole, teams: [], requestedUserId: "outside-user" });
      expect(result.canChooseGlobalUser).toBe(false);
      expect(result.scope).toBeNull();
      expect(result.error).toBe("Choose an administered team before viewing another member's usage");
    },
  );

  it("should offer only administered teams and keep a team administrator in personal scope until a team is selected", () => {
    const result = scopeFor();
    expect(result.scope).toEqual({ userId: "john", teamId: null });
    expect(result.administeredTeams).toEqual([
      { value: "engineering", label: "Engineering" },
      { value: "design", label: "Design" },
    ]);
    expect(result.canChooseGlobalUser).toBe(false);
  });

  it("should expose only identified members of the selected administered team with their actual labels", () => {
    const result = scopeFor({ requestedTeamId: "engineering" });
    expect(result.scope).toEqual({ teamId: "engineering", userId: null });
    expect(result.members).toEqual([
      { value: "john", label: "John (john@example.com)" },
      { value: "alice", label: "alice@example.com" },
      { value: "id-only", label: "id-only" },
    ]);
    expect(result.error).toBeNull();
  });

  it("should keep the team filter when selecting a team member or the administrator's own team usage", () => {
    const alice = scopeFor({ requestedTeamId: "engineering", requestedUserId: "alice" });
    const john = scopeFor({ requestedTeamId: "engineering", requestedUserId: "john" });
    expect(alice.scope).toEqual({ teamId: "engineering", userId: "alice" });
    expect(john.scope).toEqual({ teamId: "engineering", userId: "john" });
  });

  it("should reject a member from another administered team instead of querying that person's global usage", () => {
    const result = scopeFor({ requestedTeamId: "engineering", requestedUserId: "bob" });
    expect(result.scope).toBeNull();
    expect(result.error).toBe("Choose a member of the selected team");
    expect(result.members.map((member) => member.value)).not.toContain("bob");
  });

  it.each(["external", "missing-team"])("should reject the unauthorized team %s", (requestedTeamId) => {
    const result = scopeFor({ requestedTeamId });
    expect(result.scope).toBeNull();
    expect(result.members).toEqual([]);
    expect(result.error).toBe("Choose a team you administer");
  });

  it("should reject a formerly valid selection as soon as team membership is removed", () => {
    const result = scopeFor({ teams: [], requestedTeamId: "engineering", requestedUserId: "alice" });
    expect(result.scope).toBeNull();
    expect(result.administeredTeams).toEqual([]);
    expect(result.members).toEqual([]);
  });

  it("should not use a missing team list to authorize a saved team or member selection", () => {
    const result = scopeFor({ teams: undefined, requestedTeamId: "engineering", requestedUserId: "alice" });
    expect(result.scope).toBeNull();
    expect(result.error).toBe("Choose a team you administer");
  });

  it("should restrict proxy-admin member options to the selected team too", () => {
    const selected = { userRole: "Admin", requestedTeamId: "external" };
    const allowed = scopeFor({ ...selected, requestedUserId: "outside-admin" });
    const denied = scopeFor({ ...selected, requestedUserId: "alice" });
    expect(allowed.scope).toEqual({ teamId: "external", userId: "outside-admin" });
    expect(allowed.canChooseGlobalUser).toBe(false);
    expect(denied.scope).toBeNull();
  });

  it("should force personal usage for every role when ownUsage is set", () => {
    const requested = { ownUsage: true, requestedTeamId: "missing-team", requestedUserId: "outside-user" };
    for (const userRole of ["Admin", "Internal User"]) {
      const result = scopeFor({ ...requested, userRole });
      expect(result.scope).toEqual({ userId: "john", teamId: null });
      expect(result.members).toEqual([]);
      expect(result.canChooseGlobalUser).toBe(false);
      expect(result.error).toBeNull();
    }
  });

  it("should fail closed without a signed-in identity even if a proxy role or null-ID membership is present", () => {
    const result = scopeFor({ userId: null, userRole: "Admin", requestedTeamId: "engineering" });
    expect(result.scope).toBeNull();
    expect(result.administeredTeams).toEqual([]);
    expect(result.members).toEqual([]);
    expect(result.canChooseGlobalUser).toBe(false);
    expect(result.error).toBe("Sign in to view client usage");
  });

  it("should not mutate the supplied team membership data", () => {
    const before = JSON.stringify(teams);
    scopeFor({ requestedTeamId: "engineering", requestedUserId: "alice" });
    expect(JSON.stringify(teams)).toBe(before);
  });
});
