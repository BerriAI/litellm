import { useMemo } from "react";
import { useUISettings } from "@/app/(dashboard)/hooks/uiSettings/useUISettings";

const FLAG_FIELD = "enable_ptu_cost_attribution";

export const useIsPtuCostAttributionEnabled = (): { enabled: boolean; isLoading: boolean } => {
  const { data, isLoading } = useUISettings();

  const enabled = useMemo(() => {
    return Boolean(data?.values?.[FLAG_FIELD]);
  }, [data]);

  return { enabled, isLoading };
};
