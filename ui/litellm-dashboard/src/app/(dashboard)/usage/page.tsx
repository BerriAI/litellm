"use client";

import UsagePageView from "@/components/Usage/components/UsagePageView";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import useTeams from "@/app/(dashboard)/hooks/useTeams";

const UsagePage = () => {
  const { accessToken, userRole, userId, premiumUser } = useAuthorized();
  const { teams } = useTeams();

  return <UsagePageView teams={teams ?? []} organizations={[]} />;
};

export default UsagePage;
