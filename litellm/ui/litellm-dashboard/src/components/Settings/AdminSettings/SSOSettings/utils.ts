import { SSOSettingsValues } from "@/app/(dashboard)/hooks/sso/useSSOSettings";

const COGNITO_HOSTED_DOMAIN = /^(.+)\.auth\.([a-z0-9-]+)\.amazoncognito\.com$/;

export const cognitoBaseUrl = (domain: string, region: string | undefined): string => {
  const host = domain
    .trim()
    .replace(/^https?:\/\//, "")
    .replace(/\/+$/, "");
  return host.includes(".") ? `https://${host}` : `https://${host}.auth.${(region ?? "").trim()}.amazoncognito.com`;
};

export const cognitoEndpoints = (domain: string, region: string | undefined) => {
  const base = cognitoBaseUrl(domain, region);
  return {
    generic_authorization_endpoint: `${base}/oauth2/authorize`,
    generic_token_endpoint: `${base}/oauth2/token`,
    generic_userinfo_endpoint: `${base}/oauth2/userInfo`,
  };
};

// shortcut: only Cognito hosted domains (*.amazoncognito.com) are recognized, a custom domain reads back as generic
export const parseCognitoDomain = (
  authorizationEndpoint: string | null | undefined,
): { cognito_domain: string; cognito_region: string } | null => {
  if (!authorizationEndpoint) return null;
  const url = URL.canParse(authorizationEndpoint) ? new URL(authorizationEndpoint) : null;
  const match = url?.pathname === "/oauth2/authorize" ? COGNITO_HOSTED_DOMAIN.exec(url.hostname) : null;
  return match ? { cognito_domain: match[1], cognito_region: match[2] } : null;
};

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
    cognito_domain,
    cognito_region,
    ...rest
  } = formValues;

  const payload: any = {
    ...rest,
    ...(rest.sso_provider === "cognito" && cognito_domain
      ? {
          ...cognitoEndpoints(cognito_domain, cognito_region),
          generic_scope: "openid email profile",
          generic_user_id_attribute: "sub",
          generic_user_email_attribute: "email",
        }
      : {}),
  };

  if (typeof payload.saml_allow_unsolicited === "boolean") {
    payload.saml_allow_unsolicited = payload.saml_allow_unsolicited ? "true" : "false";
  }

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

const isEmptySSOValue = (value: unknown): boolean => value === undefined || value === null || value === "";

// An unchecked saml_allow_unsolicited is the same as never having set it
const comparableSSOValue = (key: string, value: unknown): unknown =>
  isEmptySSOValue(value) || (key === "saml_allow_unsolicited" && value === "false") ? undefined : value;

// /update/sso_settings is a PATCH: send only the fields that changed, with null for the ones that were emptied
export const diffSSOSettingsPayload = (
  before: Record<string, unknown>,
  after: Record<string, unknown>,
): Record<string, unknown> =>
  Object.fromEntries(
    [...new Set([...Object.keys(before), ...Object.keys(after)])]
      .filter(
        (key) =>
          JSON.stringify(comparableSSOValue(key, before[key])) !== JSON.stringify(comparableSSOValue(key, after[key])),
      )
      .map((key) => [key, isEmptySSOValue(after[key]) ? null : after[key]]),
  );

export const CLEARED_SSO_SETTINGS = {
  google_client_id: null,
  google_client_secret: null,
  microsoft_client_id: null,
  microsoft_client_secret: null,
  microsoft_tenant: null,
  generic_client_id: null,
  generic_client_secret: null,
  generic_authorization_endpoint: null,
  generic_token_endpoint: null,
  generic_userinfo_endpoint: null,
  generic_scope: null,
  generic_user_id_attribute: null,
  generic_user_email_attribute: null,
  saml_idp_metadata_url: null,
  saml_idp_metadata_xml: null,
  saml_sp_entity_id: null,
  saml_allow_unsolicited: null,
  proxy_base_url: null,
  user_email: null,
  ui_access_mode: null,
  role_mappings: null,
  team_mappings: null,
} as const;

// Determine the SSO provider based on the configuration
export const detectSSOProvider = (values: SSOSettingsValues): string | null => {
  if (values.google_client_id) return "google";
  if (values.microsoft_client_id) return "microsoft";
  if (values.generic_client_id) {
    if (parseCognitoDomain(values.generic_authorization_endpoint)) return "cognito";
    // Check if it looks like Okta/Auth0 based on endpoints
    if (
      values.generic_authorization_endpoint?.includes("okta") ||
      values.generic_authorization_endpoint?.includes("auth0")
    ) {
      return "okta";
    }
    return "generic";
  }
  if (values.saml_idp_metadata_url || values.saml_idp_metadata_xml) return "saml";
  return null;
};
