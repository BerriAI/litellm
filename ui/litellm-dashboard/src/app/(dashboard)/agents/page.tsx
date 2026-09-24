"use client";

import AgentsPanel from "./_components/AgentsPanel";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useTeams } from "@/app/(dashboard)/hooks/teams/useTeams";

export default function Agents() {
  const { accessToken, userRole } = useAuthorized();
  const { data: teams } = useTeams();
  return <AgentsPanel accessToken={accessToken} userRole={userRole} teams={teams ?? null} />;
}
