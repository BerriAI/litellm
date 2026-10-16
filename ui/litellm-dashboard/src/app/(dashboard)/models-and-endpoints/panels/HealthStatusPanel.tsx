"use client";

import { useCallback, useMemo } from "react";
import HealthCheckComponent from "@/components/model_dashboard/HealthCheckComponent";
import { useHealthTableUrlState } from "@/components/model_dashboard/useHealthTableUrlState";
import { getDisplayModelName } from "@/components/view_model/model_name_display";
import { useModelsInfo } from "@/app/(dashboard)/hooks/models/useModels";
import { useModelCostMap } from "@/app/(dashboard)/hooks/models/useModelCostMap";
import { useTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { transformModelData } from "@/app/(dashboard)/models-and-endpoints/utils/modelDataTransformer";
import { useModelDetailRouting } from "@/app/(dashboard)/models-and-endpoints/detailNavigation";

export default function HealthStatusPanel() {
  const { accessToken } = useAuthorized();
  const { data: teams } = useTeams();
  const { data: modelCostMapData } = useModelCostMap();
  const { openModel } = useModelDetailRouting();
  const { pagination, onPaginationChange } = useHealthTableUrlState();
  const {
    data: healthModelDataResponse,
    isLoading,
    isError,
  } = useModelsInfo(pagination.pageIndex + 1, pagination.pageSize);
  const pageUnavailable = isLoading || (isError && !healthModelDataResponse);

  const getProviderFromModel = useCallback(
    (model: string) => {
      if (modelCostMapData && typeof modelCostMapData === "object" && model in modelCostMapData) {
        return modelCostMapData[model]["litellm_provider"];
      }
      return "openai";
    },
    [modelCostMapData],
  );

  const processedHealthModelData = useMemo(() => {
    if (!healthModelDataResponse?.data) {
      return { data: [] };
    }
    return transformModelData(healthModelDataResponse, getProviderFromModel);
  }, [healthModelDataResponse, getProviderFromModel]);

  const healthModelIdsOnProxy = useMemo<string[]>(
    () =>
      healthModelDataResponse?.data
        ?.map((model: any) => model.model_info?.id)
        .filter((id: string | undefined): id is string => Boolean(id)) ?? [],
    [healthModelDataResponse?.data],
  );

  return (
    <HealthCheckComponent
      accessToken={accessToken}
      modelData={processedHealthModelData}
      all_models_on_proxy={healthModelIdsOnProxy}
      getDisplayModelName={getDisplayModelName}
      setSelectedModelId={openModel}
      teams={teams ?? null}
      isLoading={pageUnavailable}
      pagination={pagination}
      onPaginationChange={onPaginationChange}
      rowCount={healthModelDataResponse?.total_count ?? 0}
    />
  );
}
