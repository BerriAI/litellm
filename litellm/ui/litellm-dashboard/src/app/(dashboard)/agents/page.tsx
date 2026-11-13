"use client";
import { useTranslation } from "react-i18next";

import AgentsPanel from "./_components/AgentsPanel";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import { parseAsStringLiteral, useQueryState } from "nuqs";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { isProxyAdminRole } from "@/utils/roles";
import ExternalAgentsPanel from "@/components/agents/external/ExternalAgentsPanel";

const sections = ["registered", "external"] as const;

export default function Agents() {
  const { t } = useTranslation();

  const { accessToken, userRole } = useAuthorized();
  const { data: teams } = useTeams();
  const [section, setSection] = useQueryState(
    "agent_section",
    parseAsStringLiteral(sections).withDefault("registered"),
  );
  const admin = isProxyAdminRole(userRole ?? "");
  return (
    <div className="flex w-full min-w-0 flex-col overflow-y-auto">
      {admin && (
        <Tabs
          className="px-8 pt-6"
          value={section}
          onValueChange={(value) => {
            if (sections.includes(value as typeof section)) void setSection(value as typeof section);
          }}
        >
          <TabsList variant="line">
            <TabsTrigger value="registered">{t("agents.external.registeredAgents")}</TabsTrigger>
            <TabsTrigger value="external">{t("agents.external.externalAgents")}</TabsTrigger>
          </TabsList>
        </Tabs>
      )}
      {admin && section === "external" ? (
        <ExternalAgentsPanel token={accessToken ?? ""} teams={teams ?? []} />
      ) : (
        <AgentsPanel accessToken={accessToken} userRole={userRole} teams={teams ?? null} />
      )}
    </div>
  );
}
