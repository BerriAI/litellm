"use client";

import type { ReactNode } from "react";
import ModelsAndEndpointsView from "@/app/(dashboard)/models-and-endpoints/ModelsAndEndpointsView";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useTeams } from "@/app/(dashboard)/hooks/teams/useTeams";

export default function ModelsAndEndpointsLayout({ children }: { children: ReactNode }) {
  const { premiumUser } = useAuthorized();
  const { data: teams } = useTeams();
  return (
    <>
      <ModelsAndEndpointsView premiumUser={premiumUser} teams={teams ?? null} />
      {children}
    </>
  );
}
