import { getProxyBaseUrl, getProxyUISettings } from "@/components/networking";
import { useQuery } from "@tanstack/react-query";
import { createQueryKeys } from "../common/queryKeysFactory";

export const proxySettingsKeys = createQueryKeys("proxySettings");

export interface ProxySettings {
  PROXY_BASE_URL: string;
  PROXY_LOGOUT_URL: string;
  LITELLM_UI_API_DOC_BASE_URL?: string | null;
  DISABLE_EXPENSIVE_DB_QUERIES?: boolean;
  NUM_SPEND_LOGS_ROWS?: number;
}

const EMPTY_PROXY_SETTINGS: ProxySettings = {
  PROXY_BASE_URL: "",
  PROXY_LOGOUT_URL: "",
  LITELLM_UI_API_DOC_BASE_URL: null,
};

export function useProxySettingsQuery(accessToken: string | null) {
  const managementBaseUrl = getProxyBaseUrl();
  return useQuery({
    queryKey: [...proxySettingsKeys.all, managementBaseUrl, accessToken],
    queryFn: () => {
      if (getProxyBaseUrl() !== managementBaseUrl) throw new Error("Gateway changed while loading settings.");
      return accessToken ? getProxyUISettings(accessToken) : null;
    },
    enabled: Boolean(accessToken),
  });
}

export default function useProxySettings(accessToken: string | null): ProxySettings {
  const { data } = useProxySettingsQuery(accessToken);
  return data ?? EMPTY_PROXY_SETTINGS;
}
