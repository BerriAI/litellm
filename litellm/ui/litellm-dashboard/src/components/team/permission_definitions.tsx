import type { ParseKeys } from "i18next";
import i18n from "@/i18n/i18n";

export interface PermissionInfo {
  method: string;
  endpoint: string;
  description: string;
  route: string;
}

/**
 * Map of permission endpoint patterns to their descriptions
 */
export const PERMISSION_DESCRIPTIONS: Record<string, ParseKeys> = {
  "/auto_router/manage": "teams.memberPermissions.descriptions.autoRouterManage",
  "/key/generate": "teams.memberPermissions.descriptions.keyGenerate",
  "/key/service-account/generate": "teams.memberPermissions.descriptions.keyServiceAccountGenerate",
  "/key/update": "teams.memberPermissions.descriptions.keyUpdate",
  "/key/delete": "teams.memberPermissions.descriptions.keyDelete",
  "/key/info": "teams.memberPermissions.descriptions.keyInfo",
  "/key/regenerate": "teams.memberPermissions.descriptions.keyRegenerate",
  "/key/{key_id}/regenerate": "teams.memberPermissions.descriptions.keyIdRegenerate",
  "/key/list": "teams.memberPermissions.descriptions.keyList",
  "/key/block": "teams.memberPermissions.descriptions.keyBlock",
  "/key/unblock": "teams.memberPermissions.descriptions.keyUnblock",
  "/key/access_group_assignment": "teams.memberPermissions.descriptions.keyAccessGroupAssignment",
  "/team/daily/activity": "teams.memberPermissions.descriptions.teamDailyActivity",
  "/spend/logs": "teams.memberPermissions.descriptions.spendLogs",
};

/**
 * Determines the HTTP method for a given permission endpoint
 */
export const getMethodForEndpoint = (endpoint: string): string => {
  if (
    endpoint.includes("/info") ||
    endpoint.includes("/list") ||
    endpoint.includes("/activity") ||
    endpoint === "/spend/logs"
  ) {
    return "GET";
  }
  return "POST";
};

/**
 * Parses a permission string into a structured PermissionInfo object
 */
export const getPermissionInfo = (permission: string): PermissionInfo => {
  const method = getMethodForEndpoint(permission);
  const endpoint = permission;

  // Find exact match or fallback to default description
  let descriptionKey = Object.hasOwn(PERMISSION_DESCRIPTIONS, permission)
    ? PERMISSION_DESCRIPTIONS[permission]
    : undefined;

  // If no exact match, try to find a partial match based on patterns
  if (!descriptionKey) {
    for (const [pattern, key] of Object.entries(PERMISSION_DESCRIPTIONS)) {
      if (permission.includes(pattern)) {
        descriptionKey = key;
        break;
      }
    }
  }

  // Fallback if no match found
  const description = descriptionKey
    ? i18n.t(descriptionKey)
    : i18n.t("teams.memberPermissions.descriptions.fallback", { permission });

  return {
    method,
    endpoint,
    description,
    route: permission,
  };
};
