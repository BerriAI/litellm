"use client";

import { MemoryView } from "./_components/MemoryView";
import { AdminOnlyNotice } from "@/components/shared/AdminOnlyNotice";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import useCan from "@/app/(dashboard)/hooks/useCan";
import { useTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import useIsOrgAdmin from "@/app/(dashboard)/hooks/useIsOrgAdmin";
import { isProxyAdminRole, isUserTeamAdminForAnyTeam } from "@/utils/roles";
import { useState } from "react";
import { MemoryAdministration } from "./_components/MemorySettings";
import { AutomaticMemoryEntries } from "./_components/AutomaticMemoryEntries";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

export default function Memory() {
  const [view, setView] = useState("v2");
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
    <Tabs value={view} onValueChange={(value) => setView(value)} className="gap-6 px-8 py-8">
      {(proxyAdmin || canManage) && (
        <TabsList variant="line" aria-label="Memory sections">
          <TabsTrigger value="v2">Memories</TabsTrigger>
          {canManage && <TabsTrigger value="administration">Administration</TabsTrigger>}
          {proxyAdmin && <TabsTrigger value="v1">Memory API (V1)</TabsTrigger>}
        </TabsList>
      )}
      <TabsContent value="v2" className="space-y-8">
        {userId && (
          <AutomaticMemoryEntries key={userId} userId={userId} proxyAdmin={proxyAdmin} readOnly={isViewOnly} />
        )}
      </TabsContent>
      {canManage && userId && (
        <TabsContent value="administration">
          {view === "administration" && (
            <MemoryAdministration userId={userId} proxyAdmin={proxyAdmin} readOnly={isViewOnly} />
          )}
        </TabsContent>
      )}
      {proxyAdmin && (
        <TabsContent value="v1">
          {view === "v1" && <MemoryView accessToken={accessToken} userID={userId} userRole={userRole} />}
        </TabsContent>
      )}
    </Tabs>
  );
}
