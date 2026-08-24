"use client";

import OldTeams from "@/components/OldTeams";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function TeamsPage() {
  const { accessToken, userId, userRole, premiumUser } = useAuthorized();
  return <OldTeams accessToken={accessToken} userID={userId} userRole={userRole} premiumUser={premiumUser ?? false} />;
}
