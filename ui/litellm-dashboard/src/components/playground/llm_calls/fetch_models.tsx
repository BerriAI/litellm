// fetch_models.ts

import { useModelHub } from "@/app/(dashboard)/hooks/models/useModels";
import { useMemo } from "react";

export interface ModelGroup {
  model_group: string;
  mode?: string;
}

/**
 * Hook that fetches available models using modelHubCall and formats them for the selection dropdown.
 */
export const useAvailableModels = (
  accessToken: string | null,
): {
  models: ModelGroup[];
  isLoading: boolean;
  error: Error | null;
} => {
  const { data: fetchedModels, isLoading, error } = useModelHub(accessToken);

  const models = useMemo(() => {
    if (!fetchedModels?.data || fetchedModels.data.length === 0) {
      return [];
    }

    const formattedModels: ModelGroup[] = fetchedModels.data.map((item: any) => ({
      model_group: item.model_group,
      mode: item?.mode,
    }));

    formattedModels.sort((a, b) => a.model_group.localeCompare(b.model_group));
    return formattedModels;
  }, [fetchedModels]);

  return {
    models,
    isLoading,
    error: error as Error | null,
  };
};
