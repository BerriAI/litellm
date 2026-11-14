"use client";

import AgentsPanel from "./_components/AgentsPanel";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import ExternalCloudAgents from "@/components/agents/cloud/ExternalCloudAgents";
import { Button } from "@/components/ui/button";
import { useUrlTab } from "@/hooks/useUrlTab";
import { isProxyAdminRole } from "@/utils/roles";
import { useTranslation } from "react-i18next";

export default function Agents() {
  const { accessToken, userRole } = useAuthorized();
  const { data: teams } = useTeams();
  const { t } = useTranslation();
  const [tab, setTab] = useUrlTab(["registered", "external"], "registered", "agents_tab");
  const external = Boolean(accessToken && isProxyAdminRole(userRole ?? ""));
  return (
    <div className="min-w-0 flex-1 overflow-y-auto">
      {external && (
        <div role="tablist" aria-label={t("agents.panel.title")} className="flex gap-2 border-b px-8 pt-6 pb-3">
          <Button
            role="tab"
            aria-selected={tab === "registered"}
            variant={tab === "registered" ? "default" : "ghost"}
            onClick={() => setTab("registered")}
          >
            {t("agents.external.registered")}
          </Button>
          <Button
            role="tab"
            aria-selected={tab === "external"}
            variant={tab === "external" ? "default" : "ghost"}
            onClick={() => setTab("external")}
          >
            {t("agents.external.title")}
          </Button>
        </div>
      )}
      {external && tab === "external" && accessToken ? (
        <ExternalCloudAgents accessToken={accessToken} />
      ) : (
        <AgentsPanel accessToken={accessToken} userRole={userRole} teams={teams ?? null} />
      )}
    </div>
  );
}
