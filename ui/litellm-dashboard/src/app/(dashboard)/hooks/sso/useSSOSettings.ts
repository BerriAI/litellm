import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { getSSOSettings } from "@/components/networking";
import { useQuery, UseQueryResult } from "@tanstack/react-query";
import { createQueryKeys } from "../common/queryKeysFactory";

export interface SSOFieldSchema {
  description: string;
  properties: {
    [key: string]: {
      description: string;
      type: string;
    };
  };
}

export interface SSOSettingsValues {
  google_client_id: string | null;
  google_client_secret: string | null;
  microsoft_client_id: string | null;
  microsoft_client_secret: string | null;
  microsoft_tenant: string | null;
  generic_client_id: string | null;
  generic_client_secret: string | null;
  generic_authorization_endpoint: string | null;
  generic_token_endpoint: string | null;
  generic_userinfo_endpoint: string | null;
  proxy_base_url: string | null;
  user_email: string | null;
  ui_access_mode: string | null;
  role_mappings: RoleMappings;
  team_mappings: TeamMappings;
}

export interface RoleMappings {
  provider: string;
  group_claim: string;
  default_role: "internal_user" | "internal_user_viewer" | "proxy_admin" | "proxy_admin_viewer";
  roles: {
    [key: string]: string[];
  };
}

export interface TeamMappings {
  team_ids_jwt_field: string;
}

export interface SSOSettingsResponse {
  values: SSOSettingsValues;
  field_schema: SSOFieldSchema;
}

const ssoKeys = createQueryKeys("sso");

export const useSSOSettings = (): UseQueryResult<SSOSettingsResponse> => {
  const { accessToken, userId, userRole } = useAuthorized();
  return useQuery<SSOSettingsResponse>({
    queryKey: ssoKeys.detail("settings"),
    queryFn: async () => await getSSOSettings(accessToken!),
    enabled: Boolean(accessToken && userId && userRole),
  });
};
