"use client";

import NewUsagePage from "@/components/UsagePage/components/UsagePageView";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import { useOrganizations } from "@/app/(dashboard)/hooks/organizations/useOrganizations";

export default function UsagePage() {
  useAuthorized();
  const { data: teams } = useTeams();
  const { data: organizations } = useOrganizations();
  return <NewUsagePage teams={teams ?? []} organizations={organizations ?? []} />;
}
