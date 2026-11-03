"use client";

import NewUsagePage from "./_components/components/UsagePageView";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import { useOrganizations } from "@/app/(dashboard)/hooks/organizations/useOrganizations";

import { useQueryStates } from "nuqs";
import { useUISettings } from "@/app/(dashboard)/hooks/uiSettings/useUISettings";
import { Skeleton } from "@/components/ui/skeleton";
import { dateRangeFromParams, usageUrlParsers } from "./_components/usageUrlState";

export default function UsagePage() {
  useAuthorized();
  const { data: teams, isLoading: teamsLoading } = useTeams();
  const { data: organizations } = useOrganizations();
  const { data: settings, isLoading: settingsLoading } = useUISettings();
  const [params] = useQueryStates(usageUrlParsers);
  if (settingsLoading && !dateRangeFromParams(params)) {
    return (
      <div role="status" aria-label="Loading usage settings" className="p-6">
        <Skeleton className="h-64 w-full" />
      </div>
    );
  }
  const defaultTimeRange: unknown = settings?.values?.usage_default_time_range;
  return (
    <NewUsagePage
      teams={teams ?? []}
      organizations={organizations ?? []}
      teamsLoading={teamsLoading}
      defaultTimeRange={typeof defaultTimeRange === "string" ? defaultTimeRange : undefined}
    />
  );
}
