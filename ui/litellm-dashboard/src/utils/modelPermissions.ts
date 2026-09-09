import { Team } from "@/components/networking";

import { isProxyAdminRole, isUserTeamAdminForAnyTeam, isUserTeamAdminForSingleTeam } from "./roles";

/**
 * The dashboard's mirror of the two server layers that gate model writes: the role-level
 * route RBAC (`_check_proxy_admin_viewer_access` in litellm/proxy/auth/route_checks.py),
 * which 403s /model/new, /model/update, and /model/delete for every view-only session
 * before the endpoint runs, and ModelManagementAuthChecks in
 * litellm/proxy/management_endpoints/model_management_endpoints.py behind it.
 *
 * Past that route gate, both questions below are answered by exactly two inputs: the
 * caller's role, and whether the caller admins the team named in `model_info.team_id`.
 * `created_by` is written at creation and never read by an auth check, so it is deliberately
 * absent here; gating on it hid controls from team admins the API accepts, and showed
 * controls to former team admins the API rejects.
 */
export interface ModelActor {
  userRole: string | null;
  userID: string | null;
  /**
   * From useAuthorized(). A proxy_admin_viewer session masquerades as "Admin" in userRole
   * (effectiveSessionRole, for read parity), yet every management write 403s it, so the role
   * alone cannot answer a write question.
   */
  isViewOnly: boolean;
}

const isWritableProxyAdmin = ({ userRole, isViewOnly }: ModelActor): boolean =>
  !isViewOnly && userRole != null && isProxyAdminRole(userRole);

/** How this actor must scope a deployment they create, or that they may not create one. */
export type ModelWriteScope = "forbidden" | "unscoped-ok" | "team-required";

export interface ModelCreationLimits {
  teams: Team[] | null;
  /** The admin setting that withdraws model creation from internal users. */
  disabledForInternalUsers: boolean;
}

const isTeamAdminOf = (teams: Team[] | null, userID: string, teamId: string): boolean => {
  const team = teams?.find((candidate) => candidate.team_id === teamId);
  return team != null && isUserTeamAdminForSingleTeam(team.members_with_roles, userID);
};

/** The team member permission that opens the team's auto-routers to non-admin members. */
export const AUTO_ROUTER_MANAGEMENT_PERMISSION = "/model/auto_router_management";

const grantsAutoRouterManagement = (team: Team, userID: string): boolean =>
  team.team_member_permissions?.includes(AUTO_ROUTER_MANAGEMENT_PERMISSION) === true &&
  team.members_with_roles?.some((member) => member.user_id === userID) === true;

const isAutoRouterManagerOf = (teams: Team[] | null, userID: string, teamId: string): boolean => {
  const team = teams?.find((candidate) => candidate.team_id === teamId);
  return team != null && grantsAutoRouterManagement(team, userID);
};

const isAutoRouterManagerOfAnyTeam = (teams: Team[] | null, userID: string): boolean =>
  teams?.some((team) => grantsAutoRouterManagement(team, userID)) === true;

/**
 * POST /model/new takes a proxy admin unconditionally, or a team admin whose payload names a
 * team; an unscoped create from anyone else is a 403. A view-only session is 403d by the
 * route RBAC on its role alone, so team-admin membership cannot rescue it. Returning the
 * requirement rather than a pair of booleans keeps "may not create" and "may create
 * unscoped" from being confused.
 */
export const modelCreationScope = (
  actor: ModelActor,
  { teams, disabledForInternalUsers }: ModelCreationLimits,
): ModelWriteScope => {
  if (actor.isViewOnly) {
    return "forbidden";
  }
  if (isWritableProxyAdmin(actor)) {
    return "unscoped-ok";
  }
  if (disabledForInternalUsers) {
    return "forbidden";
  }
  if (actor.userID != null && isUserTeamAdminForAnyTeam(teams, actor.userID)) {
    return "team-required";
  }
  return "forbidden";
};

/**
 * The same POST /model/new, for an `auto_router/` deployment: everything modelCreationScope
 * admits, plus a non-admin member of a team that grants auto-router management, who must
 * scope the router to that team. The server judges the deployment's kind, so this scope
 * must never feed the Add Model form.
 */
export const autoRouterCreationScope = (actor: ModelActor, limits: ModelCreationLimits): ModelWriteScope => {
  const scope = modelCreationScope(actor, limits);
  if (scope !== "forbidden" || actor.isViewOnly || limits.disabledForInternalUsers) {
    return scope;
  }
  return actor.userID != null && isAutoRouterManagerOfAnyTeam(limits.teams, actor.userID)
    ? "team-required"
    : "forbidden";
};

export const canCreateModels = (actor: ModelActor, limits: ModelCreationLimits): boolean =>
  modelCreationScope(actor, limits) !== "forbidden";

export interface ModelRowOrigin {
  teamId: string | null | undefined;
  /** False for config.yaml rows, which update and delete both refuse whoever asks. */
  isDbModel: boolean;
  /** An `auto_router/` deployment, the only kind the member permission reaches. */
  isAutoRouter?: boolean;
  /**
   * Who created the row. Irrelevant to admins (see the note on this file), but the member
   * permission reaches only the auto-routers the member created themselves.
   */
  createdBy?: string | null;
}

/** May this actor edit or delete this specific deployment? */
export const canModifyModel = (
  actor: ModelActor,
  teams: Team[] | null,
  { teamId, isDbModel, isAutoRouter = false, createdBy = null }: ModelRowOrigin,
): boolean => {
  if (actor.isViewOnly || !isDbModel) {
    return false;
  }
  if (isWritableProxyAdmin(actor)) {
    return true;
  }
  if (actor.userID == null || teamId == null) {
    return false;
  }
  if (isTeamAdminOf(teams, actor.userID, teamId)) {
    return true;
  }
  return isAutoRouter && createdBy === actor.userID && isAutoRouterManagerOf(teams, actor.userID, teamId);
};
