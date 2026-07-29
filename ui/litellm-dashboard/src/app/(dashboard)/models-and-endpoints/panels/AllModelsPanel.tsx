"use client";

import { useCallback } from "react";
import { useSearchParams } from "next/navigation";
import AllModelsTab from "@/app/(dashboard)/models-and-endpoints/components/AllModelsTab";
import { ALL_MODEL_GROUPS_VALUE } from "@/app/(dashboard)/models-and-endpoints/components/AllModelsTable";
import { useModelDashboardData } from "@/app/(dashboard)/models-and-endpoints/useModelDashboardData";
import { useModelDetailRouting } from "@/app/(dashboard)/models-and-endpoints/detailNavigation";
import { navigateWithParams } from "@/app/(dashboard)/navigateWithParams";

export default function AllModelsPanel() {
  const searchParams = useSearchParams();
  const selectedModelGroup = searchParams?.get("model_group") ?? null;
  const setSelectedModelGroup = useCallback((modelGroup: string) => {
    navigateWithParams((params) => {
      if (modelGroup && modelGroup !== ALL_MODEL_GROUPS_VALUE) {
        params.set("model_group", modelGroup);
      } else {
        params.delete("model_group");
      }
    }, "replace");
  }, []);
  const { availableModelGroups, availableModelAccessGroups } = useModelDashboardData();
  const { openModel, openTeam } = useModelDetailRouting();

  return (
    <AllModelsTab
      selectedModelGroup={selectedModelGroup}
      setSelectedModelGroup={setSelectedModelGroup}
      availableModelGroups={availableModelGroups}
      availableModelAccessGroups={availableModelAccessGroups}
      setSelectedModelId={openModel}
      setSelectedTeamId={openTeam}
    />
  );
}
