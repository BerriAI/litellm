import { fetchProxySettings } from "@/utils/proxyUtils";
import { useQuery } from "@tanstack/react-query";
import { createQueryKeys } from "../common/queryKeysFactory";

export const proxySettingsKeys = createQueryKeys("proxySettings");

export interface ProxySettings {
  PROXY_BASE_URL: string;
  PROXY_LOGOUT_URL: string;
  LITELLM_UI_API_DOC_BASE_URL?: string | null;
}

const EMPTY_PROXY_SETTINGS: ProxySettings = {
  PROXY_BASE_URL: "",
  PROXY_LOGOUT_URL: "",
  LITELLM_UI_API_DOC_BASE_URL: null,
};

export default function useProxySettings(accessToken: string | null): ProxySettings {
  const { data } = useQuery({
    queryKey: [...proxySettingsKeys.all, accessToken],
    queryFn: () => fetchProxySettings(accessToken),
    enabled: Boolean(accessToken),
  });
  return data ?? EMPTY_PROXY_SETTINGS;
}
