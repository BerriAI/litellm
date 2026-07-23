import { useMemo } from "react";
import { useModelsInfo } from "@/app/(dashboard)/hooks/models/useModels";

export interface ModelDashboardData {
  availableModelGroups: string[];
  availableModelAccessGroups: string[];
  allModelsOnProxy: string[];
  isLoading: boolean;
}

export function useModelDashboardData(): ModelDashboardData {
  const { data: modelDataResponse, isLoading } = useModelsInfo();

  const availableModelGroups = useMemo(() => {
    const groups = new Set<string>(modelDataResponse?.data?.map((model) => model.model_name) ?? []);
    return Array.from(groups).sort();
  }, [modelDataResponse?.data]);

  const availableModelAccessGroups = useMemo(() => {
    const groups = new Set<string>(
      modelDataResponse?.data?.flatMap((model) => model.model_info?.access_groups ?? []) ?? [],
    );
    return Array.from(groups);
  }, [modelDataResponse?.data]);

  const allModelsOnProxy = useMemo(
    () => modelDataResponse?.data?.map((model) => model.model_name) ?? [],
    [modelDataResponse?.data],
  );

  return { availableModelGroups, availableModelAccessGroups, allModelsOnProxy, isLoading };
}
