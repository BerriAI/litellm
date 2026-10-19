import { describe, expect, it } from "vitest";

import type { Member } from "../networking";
import {
  administersTeam,
  anotherUserScope,
  effectiveKeyOwner,
  isTeamMemberOwner,
  teamMemberOwnerOptions,
  teamMemberOwnerPlaceholder,
  type TeamMembers,
} from "./createKeyOwnerAccess";

const VIEWER = "viewer";

const member = (user_id: string | null, role: string, user_email: string | null = null): Member => ({
  user_id,
  role,
  user_email,
});

const administered: TeamMembers & { team_id: string } = {
  team_id: "administered",
  members_with_roles: [member(VIEWER, "admin"), member("alice", "user", "alice@example.com"), member("bob", "user")],
};

const joined: TeamMembers & { team_id: string } = {
  team_id: "joined",
  members_with_roles: [member(VIEWER, "user"), member("carol", "admin", "carol@example.com")],
};

const foreign: TeamMembers & { team_id: string } = {
  team_id: "foreign",
  members_with_roles: [member("dave", "admin")],
};

const viewer = {
  userRole: "Internal User",
  userId: VIEWER,
  teamAdminEditableFields: ["create_member_keys"],
  teams: [joined, administered],
};

describe("anotherUserScope", () => {
  it("lets a team admin pick among team members once the proxy enables create_member_keys", () => {
    expect(anotherUserScope(viewer)).toBe("team_members");
  });

  it.each([
    ["nothing is enabled", []],
    ["only other team admin permissions are enabled", ["max_budget", "member_key_budgets", "projects"]],
  ])("hides Another User from a team admin when %s", (_label, teamAdminEditableFields) => {
    expect(anotherUserScope({ ...viewer, teamAdminEditableFields })).toBe("none");
  });

  it.each([
    ["only a plain member of their teams", [joined]],
    ["an admin of no listed team", [foreign]],
    ["in no team", []],
    ["still waiting on the team list", null],
  ])("hides Another User with the permission on from a user who is %s", (_label, teams) => {
    expect(anotherUserScope({ ...viewer, teams })).toBe("none");
  });

  it("hides Another User when the session carries no user id", () => {
    expect(anotherUserScope({ ...viewer, userId: null })).toBe("none");
  });

  it.each([
    ["with the permission off and no teams", { teamAdminEditableFields: [], teams: [] }],
    ["with the permission on and an administered team", {}],
    ["before the team list has loaded", { teams: null }],
  ])("lets a proxy admin pick any user %s", (_label, overrides) => {
    expect(anotherUserScope({ ...viewer, userRole: "Admin", ...overrides })).toBe("any_user");
  });
});

describe("effectiveKeyOwner", () => {
  it.each(["any_user", "team_members"] as const)("keeps another_user for a viewer scoped to %s", (scope) => {
    expect(effectiveKeyOwner("another_user", scope)).toBe("another_user");
  });

  it("falls back to you when another_user is requested by a viewer who may not pick one", () => {
    expect(effectiveKeyOwner("another_user", "none")).toBe("you");
  });

  it.each(["you", "service_account", "agent"])(
    "leaves %s alone for a viewer who may not pick another user",
    (owner) => {
      expect(effectiveKeyOwner(owner, "none")).toBe(owner);
    },
  );
});

describe("administersTeam", () => {
  it("narrows a team list to the teams the viewer administers", () => {
    const teams = [joined, administered, foreign];

    expect(teams.filter((team) => administersTeam(team, VIEWER)).map((team) => team.team_id)).toStrictEqual([
      "administered",
    ]);
  });

  it.each([
    ["no team", null],
    ["a team with no member list", {}],
    ["a team whose member list is null", { members_with_roles: null }],
  ])("is false for %s", (_label, team) => {
    expect(administersTeam(team, VIEWER)).toBe(false);
  });
});

describe("teamMemberOwnerOptions", () => {
  it("lists the selected team's other members, labelled by email when the team has one", () => {
    expect(teamMemberOwnerOptions(administered, VIEWER)).toStrictEqual([
      { value: "alice", label: "alice@example.com (alice)" },
      { value: "bob", label: "bob" },
    ]);
  });

  it("skips a member row that has no user id", () => {
    const team = {
      members_with_roles: [member(VIEWER, "admin"), member(null, "user", "pending@example.com"), member("bob", "user")],
    };

    expect(teamMemberOwnerOptions(team, VIEWER)).toStrictEqual([{ value: "bob", label: "bob" }]);
  });

  it.each([
    ["no team is selected", null],
    ["the selected team is one the viewer only belongs to", joined],
    ["the selected team is one the viewer is not in", foreign],
  ])("offers nobody when %s", (_label, team) => {
    expect(teamMemberOwnerOptions(team, VIEWER)).toStrictEqual([]);
  });
});

describe("teamMemberOwnerPlaceholder", () => {
  const soloTeam = { members_with_roles: [member(VIEWER, "admin")] };

  it.each([
    ["asks for a team before one is selected", null, "Select a team first"],
    ["says so when the viewer is the team's only member", soloTeam, "This team has no other members"],
    ["asks for a member once the team has someone to pick", administered, "Select a member of the team"],
  ])("%s", (_label, team, expected) => {
    expect(teamMemberOwnerPlaceholder(teamMemberOwnerOptions(team, VIEWER), team)).toBe(expected);
  });
});

describe("isTeamMemberOwner", () => {
  it("keeps an owner who belongs to the selected team", () => {
    expect(isTeamMemberOwner("alice", administered, VIEWER)).toBe(true);
  });

  it.each([
    ["is in another team only", "carol", administered],
    ["was picked before the team was cleared", "alice", null],
    ["belongs to a team the viewer does not administer", "carol", joined],
    ["is the viewer, who already has the You option", VIEWER, administered],
    ["is not set", null, administered],
  ])("rejects an owner who %s", (_label, ownerId, team) => {
    expect(isTeamMemberOwner(ownerId, team, VIEWER)).toBe(false);
  });
});
