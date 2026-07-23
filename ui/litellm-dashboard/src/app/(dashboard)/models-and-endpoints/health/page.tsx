"use client";

import { useMemo, useState } from "react";
import type { PaginationState } from "@tanstack/react-table";
import HealthCheckComponent from "@/components/model_dashboard/HealthCheckComponent";
import { getDisplayModelName } from "@/components/view_model/model_name_display";
import { useModelsInfo } from "@/app/(dashboard)/hooks/models/useModels";
import { useModelCostMap } from "@/app/(dashboard)/hooks/models/useModelCostMap";
import { useTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { transformModelData } from "@/app/(dashboard)/models-and-endpoints/utils/modelDataTransformer";
import { useModelDetailRouting } from "@/app/(dashboard)/models-and-endpoints/detailNavigation";

const HEALTH_PAGE_SIZE = 50;

export default function HealthStatusPage() {
  const { accessToken } = useAuthorized();
  const { data: teams } = useTeams();
  const { data: modelCostMapData } = useModelCostMap();
  const { openModel } = useModelDetailRouting();
  const [pagination, setPagination] = useState<PaginationState>({ pageIndex: 0, pageSize: HEALTH_PAGE_SIZE });
  const { data: healthModelDataResponse, isLoading } = useModelsInfo(pagination.pageIndex + 1, pagination.pageSize);

  const getProviderFromModel = (model: string) => {
    if (modelCostMapData && typeof modelCostMapData === "object" && model in modelCostMapData) {
      return modelCostMapData[model]["litellm_provider"];
    }
    return "openai";
  };

  const processedHealthModelData = useMemo(() => {
    if (!healthModelDataResponse?.data) {
      return { data: [] };
    }
    return transformModelData(healthModelDataResponse, getProviderFromModel);
  }, [healthModelDataResponse?.data]);

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
      isLoading={isLoading}
      pagination={pagination}
      onPaginationChange={setPagination}
      rowCount={healthModelDataResponse?.total_count ?? 0}
    />
  );
}
