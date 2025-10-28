export interface PermissionInfo {
  method: string;
  endpoint: string;
  description: string;
  route: string;
}

/**
 * Map of permission endpoint patterns to their descriptions
 */
export const PERMISSION_DESCRIPTIONS: Record<string, string> = {
  "/key/generate": "Member can generate a virtual key for this team",
  "/key/service-account/generate":
    "Member can generate a service account key (not belonging to any user) for this team",
  "/key/update": "Member can update a virtual key belonging to this team",
  "/key/delete": "Member can delete a virtual key belonging to this team",
  "/key/info": "Member can get info about a virtual key belonging to this team",
  "/key/regenerate": "Member can regenerate a virtual key belonging to this team",
  "/key/{key_id}/regenerate": "Member can regenerate a virtual key belonging to this team",
  "/key/list": "Member can list virtual keys belonging to this team",
  "/key/block": "Member can block a virtual key belonging to this team",
  "/key/unblock": "Member can unblock a virtual key belonging to this team",
};

/**
 * Determines the HTTP method for a given permission endpoint
 */
export const getMethodForEndpoint = (endpoint: string): string => {
  if (endpoint.includes("/info") || endpoint.includes("/list")) {
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
  let description = PERMISSION_DESCRIPTIONS[permission];

  // If no exact match, try to find a partial match based on patterns
  if (!description) {
    for (const [pattern, desc] of Object.entries(PERMISSION_DESCRIPTIONS)) {
      if (permission.includes(pattern)) {
        description = desc;
        break;
      }
    }
  }

  // Fallback if no match found
  if (!description) {
    description = `Access ${permission}`;
  }

  return {
    method,
    endpoint,
    description,
    route: permission,
  };
};
