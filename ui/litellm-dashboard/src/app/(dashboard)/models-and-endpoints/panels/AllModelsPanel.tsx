"use client";

import AllModelsTab from "@/app/(dashboard)/models-and-endpoints/components/AllModelsTab";
import { ALL_MODEL_GROUPS_VALUE } from "@/app/(dashboard)/models-and-endpoints/components/AllModelsTable";
import { useModelDashboardData } from "@/app/(dashboard)/models-and-endpoints/useModelDashboardData";
import {
  useModelDetailRouting,
  useModelGroupFilterRouting,
} from "@/app/(dashboard)/models-and-endpoints/detailNavigation";

export default function AllModelsPanel() {
  const { modelGroup, setModelGroup } = useModelGroupFilterRouting();
  const { availableModelGroups, availableModelAccessGroups } = useModelDashboardData();
  const { openModel, openTeam } = useModelDetailRouting();

  return (
    <AllModelsTab
      selectedModelGroup={modelGroup}
      setSelectedModelGroup={(group) => setModelGroup(group === ALL_MODEL_GROUPS_VALUE ? null : group)}
      availableModelGroups={availableModelGroups}
      availableModelAccessGroups={availableModelAccessGroups}
      setSelectedModelId={openModel}
      setSelectedTeamId={openTeam}
    />
  );
}
