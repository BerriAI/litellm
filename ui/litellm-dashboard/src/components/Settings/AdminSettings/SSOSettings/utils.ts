import { SSOSettingsValues } from "@/app/(dashboard)/hooks/sso/useSSOSettings";

/**
 * Processes SSO settings form values and transforms them into the payload format expected by the API
 * Handles role mappings transformation and field extraction
 */
export const processSSOSettingsPayload = (formValues: Record<string, any>): Record<string, any> => {
  const {
    proxy_admin_teams,
    admin_viewer_teams,
    internal_user_teams,
    internal_viewer_teams,
    default_role,
    group_claim,
    use_role_mappings,
    use_team_mappings,
    team_ids_jwt_field,
    ...rest
  } = formValues;

  const payload: any = {
    ...rest,
  };

  // Add role mappings only if use_role_mappings is checked AND provider supports role mappings
  const provider = rest.sso_provider;
  const supportsRoleMappings = provider === "okta" || provider === "generic";
  if (use_role_mappings && supportsRoleMappings) {
    // Helper function to split comma-separated string into array
    const splitTeams = (teams: string | undefined): string[] => {
      if (!teams || teams.trim() === "") return [];
      return teams
        .split(",")
        .map((team) => team.trim())
        .filter((team) => team.length > 0);
    };

    // Map default role display values to backend values
    const defaultRoleMapping: Record<string, string> = {
      internal_user_viewer: "internal_user_viewer",
      internal_user: "internal_user",
      proxy_admin_viewer: "proxy_admin_viewer",
      proxy_admin: "proxy_admin",
    };

    payload.role_mappings = {
      provider: "generic",
      group_claim,
      default_role: defaultRoleMapping[default_role] || "internal_user",
      roles: {
        proxy_admin: splitTeams(proxy_admin_teams),
        proxy_admin_viewer: splitTeams(admin_viewer_teams),
        internal_user: splitTeams(internal_user_teams),
        internal_user_viewer: splitTeams(internal_viewer_teams),
      },
    };
  }

  // Add team mappings only if use_team_mappings is checked AND provider supports team mappings
  const supportsTeamMappings = provider === "okta" || provider === "generic";
  if (use_team_mappings && supportsTeamMappings) {
    payload.team_mappings = {
      team_ids_jwt_field: team_ids_jwt_field,
    };
  }

  return payload;
};

// Determine the SSO provider based on the configuration
export const detectSSOProvider = (values: SSOSettingsValues): string | null => {
  if (values.google_client_id) return "google";
  if (values.microsoft_client_id) return "microsoft";
  if (values.generic_client_id) {
    // Check if it looks like Okta/Auth0 based on endpoints
    if (
      values.generic_authorization_endpoint?.includes("okta") ||
      values.generic_authorization_endpoint?.includes("auth0")
    ) {
      return "okta";
    }
    return "generic";
  }
  return null;
};
