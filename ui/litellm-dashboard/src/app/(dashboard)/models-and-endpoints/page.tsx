"use client";

import { useState } from "react";
import AllModelsTab from "@/app/(dashboard)/models-and-endpoints/components/AllModelsTab";
import { useModelDashboardData } from "@/app/(dashboard)/models-and-endpoints/useModelDashboardData";
import { useModelDetailRouting } from "@/app/(dashboard)/models-and-endpoints/detailNavigation";

export default function AllModelsPage() {
  const [selectedModelGroup, setSelectedModelGroup] = useState<string | null>(null);
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
