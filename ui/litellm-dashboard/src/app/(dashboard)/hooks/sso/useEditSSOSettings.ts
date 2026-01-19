import { useMutation, UseMutationResult } from "@tanstack/react-query";
import { updateSSOSettings } from "@/components/networking";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export interface EditSSOSettingsParams {
  google_client_id?: string | null;
  google_client_secret?: string | null;
  microsoft_client_id?: string | null;
  microsoft_client_secret?: string | null;
  microsoft_tenant?: string | null;
  generic_client_id?: string | null;
  generic_client_secret?: string | null;
  generic_authorization_endpoint?: string | null;
  generic_token_endpoint?: string | null;
  generic_userinfo_endpoint?: string | null;
  proxy_base_url?: string | null;
  user_email?: string | null;
  sso_provider?: string | null;
  role_mappings?: any;
  [key: string]: any;
}

export interface EditSSOSettingsResponse {
  [key: string]: any;
}

export const useEditSSOSettings = (): UseMutationResult<EditSSOSettingsResponse, Error, EditSSOSettingsParams> => {
  const { accessToken } = useAuthorized();

  return useMutation<EditSSOSettingsResponse, Error, EditSSOSettingsParams>({
    mutationFn: async (params: EditSSOSettingsParams) => {
      if (!accessToken) {
        throw new Error("Access token is required");
      }
      return await updateSSOSettings(accessToken, params);
    },
  });
};
