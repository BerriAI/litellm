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
  '/key/generate': 'Generate Key Fn',
  '/key/update': 'Update Key Fn',
  '/key/delete': 'Delete Key Fn',
  '/key/info': 'Info Key Fn',
  '/key/regenerate': 'Regenerate Key Fn',
  '/key/{key_id}/regenerate': 'Regenerate Key Fn',
  '/key/list': 'List Keys',
  '/key/block': 'Block Key',
  '/key/unblock': 'Unblock Key'
};

/**
 * Determines the HTTP method for a given permission endpoint
 */
export const getMethodForEndpoint = (endpoint: string): string => {
  if (endpoint.includes('/info') || endpoint.includes('/list')) {
    return 'GET';
  }
  return 'POST';
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
    route: permission
  };
}; 