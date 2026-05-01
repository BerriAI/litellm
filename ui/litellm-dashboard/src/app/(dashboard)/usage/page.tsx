"use client";

import UsagePageView from "@/components/UsagePage/components/UsagePageView";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import useTeams from "@/app/(dashboard)/hooks/useTeams";
import { useOrganizations } from "@/app/(dashboard)/hooks/organizations/useOrganizations";

const UsagePage = () => {
  const { accessToken, userRole, userId, premiumUser } = useAuthorized();
  const { teams } = useTeams();
  const { data: organizations } = useOrganizations();

  return <UsagePageView teams={teams ?? []} organizations={organizations ?? []} />;
};

export default UsagePage;
