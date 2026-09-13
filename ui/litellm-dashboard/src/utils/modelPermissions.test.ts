import { describe, expect, it } from "vitest";

import { Team } from "@/components/networking";
import { canCreateModels, canModifyModel, modelCreationScope } from "./modelPermissions";

const teamWhere = (userId: string, role: string, teamId = "team-1"): Team[] =>
  [{ team_id: teamId, members_with_roles: [{ user_id: userId, user_email: "t@test.com", role }] }] as unknown as Team[];

const PROXY_ADMIN = { userRole: "Admin", userID: "u-admin", isViewOnly: false };
const TEAM_ADMIN = { userRole: "Internal User", userID: "u-team-admin", isViewOnly: false };
const MEMBER = { userRole: "Internal User", userID: "u-member", isViewOnly: false };
// proxy_admin_viewer sessions: effectiveSessionRole masquerades the role as "Admin".
const VIEW_ONLY_ADMIN = { userRole: "Admin", userID: "u-viewer", isViewOnly: true };

const noLimits = { disabledForInternalUsers: false };

describe("modelCreationScope", () => {
  it("lets a proxy admin create without naming a team", () => {
    expect(modelCreationScope(PROXY_ADMIN, { teams: null, ...noLimits })).toBe("unscoped-ok");
  });

  // Live-verified: POST /model/new from a team admin 403s without model_info.team_id and
  // returns 200 with it, so the form must make the team mandatory rather than optional.
  it("requires a team admin to name a team", () => {
    expect(modelCreationScope(TEAM_ADMIN, { teams: teamWhere("u-team-admin", "admin"), ...noLimits })).toBe(
      "team-required",
    );
  });

  it("forbids a plain team member", () => {
    expect(modelCreationScope(MEMBER, { teams: teamWhere("u-member", "user"), ...noLimits })).toBe("forbidden");
  });

  // The admin setting is scoped to internal users and must never lock out a proxy admin.
  it("honours the internal-user kill switch without touching proxy admins", () => {
    const limits = { teams: teamWhere("u-team-admin", "admin"), disabledForInternalUsers: true };
    expect(modelCreationScope(TEAM_ADMIN, limits)).toBe("forbidden");
    expect(modelCreationScope(PROXY_ADMIN, limits)).toBe("unscoped-ok");
  });

  // org_admin and Admin Viewer are in all_admin_roles but are not PROXY_ADMIN to the API, so
  // an unscoped create from them 403s. Treating them as admins here is what let a form submit
  // a payload the backend always rejected.
  it("does not treat an org admin as able to create unscoped", () => {
    const orgAdmin = { userRole: "org_admin", userID: "u-org", isViewOnly: false };
    expect(modelCreationScope(orgAdmin, { teams: teamWhere("u-org", "admin"), ...noLimits })).toBe("team-required");
  });

  // Server-side, POST /model/new 403s the viewer roles, so the "Admin" the masquerade
  // reports must not read as a proxy admin here.
  it("forbids a view-only admin session despite the masqueraded Admin role", () => {
    expect(modelCreationScope(VIEW_ONLY_ADMIN, { teams: [], ...noLimits })).toBe("forbidden");
    expect(canCreateModels(VIEW_ONLY_ADMIN, { teams: [], ...noLimits })).toBe(false);
  });

  // _check_proxy_admin_viewer_access (route_checks.py) 403s /model/new on the session role
  // alone, before the team-scoped carve-out in ModelManagementAuthChecks can run.
  it("forbids a view-only admin even when they admin a team", () => {
    expect(modelCreationScope(VIEW_ONLY_ADMIN, { teams: teamWhere("u-viewer", "admin"), ...noLimits })).toBe(
      "forbidden",
    );
  });
});

describe("canModifyModel", () => {
  const teamRow = { teamId: "team-1", isDbModel: true };

  // config.yaml rows: PATCH /model/{id}/update 404s and POST /model/delete 400s for everyone.
  it("refuses a config-defined row even to a proxy admin", () => {
    expect(canModifyModel(PROXY_ADMIN, null, { teamId: "team-1", isDbModel: false })).toBe(false);
  });

  it("lets a proxy admin act on any DB row", () => {
    expect(canModifyModel(PROXY_ADMIN, null, teamRow)).toBe(true);
  });

  // The regression this whole owner exists for. Live-verified: a model created by the proxy
  // admin (created_by=default_user_id) was PATCHed and DELETEd 200 by a team admin who did
  // not create it. Authorizing on created_by hid controls the API accepts.
  it("lets a team admin act on their team's row they did not create", () => {
    expect(canModifyModel(TEAM_ADMIN, teamWhere("u-team-admin", "admin"), teamRow)).toBe(true);
  });

  it("refuses a plain member of the owning team", () => {
    expect(canModifyModel(MEMBER, teamWhere("u-member", "user"), teamRow)).toBe(false);
  });

  it("refuses a team admin of a different team", () => {
    expect(canModifyModel(TEAM_ADMIN, teamWhere("u-team-admin", "admin", "other-team"), teamRow)).toBe(false);
  });

  // Unscoped rows can only have been created by a proxy admin, and only one can edit them.
  it("refuses a team admin on an unscoped row", () => {
    expect(canModifyModel(TEAM_ADMIN, teamWhere("u-team-admin", "admin"), { teamId: null, isDbModel: true })).toBe(
      false,
    );
  });

  it("does not treat two absent identities as a match", () => {
    expect(canModifyModel({ userRole: "Internal User", userID: null, isViewOnly: false }, null, teamRow)).toBe(false);
  });

  // PATCH /model/{id}/update and POST /model/delete 403 the viewer roles like /model/new does.
  it("refuses a view-only admin session on a DB row", () => {
    expect(canModifyModel(VIEW_ONLY_ADMIN, null, teamRow)).toBe(false);
  });

  // The route RBAC blocks /model/update and /model/delete for the viewer role before the
  // team-scoped carve-out runs, so team-admin membership changes nothing here either.
  it("refuses a view-only user even when they admin the owning team", () => {
    expect(canModifyModel(VIEW_ONLY_ADMIN, teamWhere("u-viewer", "admin"), teamRow)).toBe(false);
  });
});
