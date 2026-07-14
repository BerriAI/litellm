import { useMemo } from "react";
import { ConfigType, useProxyConfig } from "@/app/(dashboard)/hooks/proxyConfig/useProxyConfig";

const FLAG_FIELD = "enable_ptu_cost_attribution";

export const useIsPtuCostAttributionEnabled = (): { enabled: boolean; isLoading: boolean } => {
  const { data, isLoading } = useProxyConfig(ConfigType.GENERAL_SETTINGS);

  const enabled = useMemo(() => {
    if (!data) {
      return false;
    }
    const entry = data.find((item) => item.field_name === FLAG_FIELD);
    return Boolean(entry?.field_value);
  }, [data]);

  return { enabled, isLoading };
};
