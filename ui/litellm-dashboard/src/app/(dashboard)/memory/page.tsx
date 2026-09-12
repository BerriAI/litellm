"use client";

import { MemoryView } from "./_components/MemoryView";
import { AdminOnlyNotice } from "@/components/shared/AdminOnlyNotice";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import useCan from "@/app/(dashboard)/hooks/useCan";
import { useTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import useIsOrgAdmin from "@/app/(dashboard)/hooks/useIsOrgAdmin";
import { isProxyAdminRole, isUserTeamAdminForAnyTeam } from "@/utils/roles";
import { MemoryPolicies, MemoryPreference } from "./_components/MemorySettings";
import { AutomaticMemoryEntries } from "./_components/AutomaticMemoryEntries";

export default function Memory() {
  const { accessToken, userRole, userId, isViewOnly } = useAuthorized();
  const canViewMemory = useCan("viewMemory");
  const teams = useTeams();
  const orgAdmin = useIsOrgAdmin();
  const proxyAdmin =
    isProxyAdminRole(userRole ?? "") || userRole === "Admin Viewer" || userRole === "proxy_admin_viewer";
  const canManage = proxyAdmin || orgAdmin || isUserTeamAdminForAnyTeam(teams.data ?? null, userId ?? "");

  if (!canViewMemory) {
    return <AdminOnlyNotice pageTitle="Memory" />;
  }

  return (
    <div className="space-y-6 p-6">
      <h1 className="text-2xl font-semibold">Memory</h1>
      {userId && <MemoryPreference userId={userId} readOnly={isViewOnly} />}
      {userId && <AutomaticMemoryEntries userId={userId} readOnly={isViewOnly} />}
      {canManage && userId && <MemoryPolicies userId={userId} proxyAdmin={proxyAdmin} readOnly={isViewOnly} />}
      {proxyAdmin && <MemoryView accessToken={accessToken} userID={userId} userRole={userRole} />}
    </div>
  );
}
