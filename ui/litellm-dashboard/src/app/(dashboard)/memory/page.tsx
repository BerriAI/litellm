"use client";

import { MemoryView } from "./_components/MemoryView";
import { AdminOnlyNotice } from "@/components/shared/AdminOnlyNotice";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import useCan from "@/app/(dashboard)/hooks/useCan";
import { useTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import useIsOrgAdmin from "@/app/(dashboard)/hooks/useIsOrgAdmin";
import { isProxyAdminRole, isUserTeamAdminForAnyTeam } from "@/utils/roles";
import { useState } from "react";
import { ChevronDown } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { MemoryPolicies } from "./_components/MemorySettings";
import { AutomaticMemoryEntries } from "./_components/AutomaticMemoryEntries";

export default function Memory() {
  const [advanced, setAdvanced] = useState(false);
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
    <div className="space-y-8 px-8 py-8">
      {userId && <AutomaticMemoryEntries key={userId} userId={userId} proxyAdmin={proxyAdmin} readOnly={isViewOnly} />}
      {canManage && userId && (
        <Collapsible open={advanced} onOpenChange={setAdvanced} className="border-t pt-4">
          <CollapsibleTrigger render={<Button variant="ghost" className="gap-2 text-muted-foreground" />}>
            <ChevronDown className={`size-4 transition-transform ${advanced ? "rotate-180" : ""}`} />
            Advanced settings
          </CollapsibleTrigger>
          <CollapsibleContent>
            {advanced && (
              <div className="space-y-6 pt-4">
                <MemoryPolicies userId={userId} proxyAdmin={proxyAdmin} readOnly={isViewOnly} />
                {proxyAdmin && <MemoryView accessToken={accessToken} userID={userId} userRole={userRole} />}
              </div>
            )}
          </CollapsibleContent>
        </Collapsible>
      )}
    </div>
  );
}
