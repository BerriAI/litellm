"use client";

import { useCallback } from "react";
import { useSearchParams } from "next/navigation";
import AllModelsTab from "@/app/(dashboard)/models-and-endpoints/components/AllModelsTab";
import { ALL_MODEL_GROUPS_VALUE } from "@/app/(dashboard)/models-and-endpoints/components/AllModelsTable";
import { useModelDashboardData } from "@/app/(dashboard)/models-and-endpoints/useModelDashboardData";
import { useModelDetailRouting } from "@/app/(dashboard)/models-and-endpoints/detailNavigation";
import { navigateWithParams } from "@/app/(dashboard)/navigateWithParams";

const useUrlFilter = (paramName: string): [string | null, (value: string | null) => void] => {
  const searchParams = useSearchParams();
  const value = searchParams?.get(paramName) ?? null;
  const setValue = useCallback(
    (nextValue: string | null) => {
      navigateWithParams((params) => {
        if (nextValue && nextValue !== ALL_MODEL_GROUPS_VALUE) {
          params.set(paramName, nextValue);
        } else {
          params.delete(paramName);
        }
      }, "replace");
    },
    [paramName],
  );
  return [value, setValue];
};

export default function AllModelsPanel() {
  const [selectedModelGroup, setSelectedModelGroup] = useUrlFilter("model_group");
  const [selectedModelAccessGroupFilter, setSelectedModelAccessGroupFilter] = useUrlFilter("model_access_group");
  const { availableModelGroups, availableModelAccessGroups } = useModelDashboardData();
  const { openModel, openTeam } = useModelDetailRouting();

  return (
    <AllModelsTab
      selectedModelGroup={selectedModelGroup}
      setSelectedModelGroup={setSelectedModelGroup}
      selectedModelAccessGroupFilter={selectedModelAccessGroupFilter}
      setSelectedModelAccessGroupFilter={setSelectedModelAccessGroupFilter}
      availableModelGroups={availableModelGroups}
      availableModelAccessGroups={availableModelAccessGroups}
      setSelectedModelId={openModel}
      setSelectedTeamId={openTeam}
    />
  );
}
