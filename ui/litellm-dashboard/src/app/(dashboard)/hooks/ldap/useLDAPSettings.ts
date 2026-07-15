import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { getLDAPSettings } from "@/components/networking";
import { useQuery, UseQueryResult } from "@tanstack/react-query";
import { createQueryKeys } from "../common/queryKeysFactory";

export interface LDAPFieldSchema {
  description: string;
  properties: {
    [key: string]: {
      description: string;
      type: string;
    };
  };
}

export interface LDAPSettingsValues {
  ldap_enabled: boolean;
  ldap_url: string | null;
  ldap_base_dn: string | null;
  ldap_search_base: string | null;
  ldap_bind_dn: string | null;
  ldap_bind_password: string | null;
  ldap_user_search_filter: string;
  ldap_user_id_attribute: string | null;
  ldap_email_attribute: string;
  ldap_display_name_attribute: string;
  ldap_group_attribute: string;
  ldap_admin_group_dn: string | null;
  ldap_use_ssl: boolean;
  ldap_start_tls: boolean;
  ldap_allow_insecure: boolean;
}

export interface LDAPSettingsResponse {
  values: LDAPSettingsValues;
  field_schema: LDAPFieldSchema;
}

export const ldapKeys = createQueryKeys("ldap");

export const useLDAPSettings = (): UseQueryResult<LDAPSettingsResponse> => {
  const { accessToken, userId, userRole } = useAuthorized();
  return useQuery<LDAPSettingsResponse>({
    queryKey: ldapKeys.detail("settings"),
    queryFn: async () => await getLDAPSettings(accessToken!),
    enabled: Boolean(accessToken && userId && userRole),
  });
};
