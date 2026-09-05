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

export const canCreateModels = (actor: ModelActor, limits: ModelCreationLimits): boolean =>
  modelCreationScope(actor, limits) !== "forbidden";

export interface ModelRowOrigin {
  teamId: string | null | undefined;
  /** False for config.yaml rows, which update and delete both refuse whoever asks. */
  isDbModel: boolean;
}

/** May this actor edit or delete this specific deployment? */
export const canModifyModel = (
  actor: ModelActor,
  teams: Team[] | null,
  { teamId, isDbModel }: ModelRowOrigin,
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
  return isTeamAdminOf(teams, actor.userID, teamId);
};
