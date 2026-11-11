import React from "react";
import type { TFunction } from "i18next";
import { useTranslation } from "react-i18next";
import type { components } from "@/lib/http/schema";
import { useQuery } from "@tanstack/react-query";
import { apiClient } from "@/components/networking";
import { Button } from "@/components/ui/button";
import { readAgentIdentity } from "./agent_identity";

const authenticationMessage = (t: TFunction, error: boolean, lastAuthenticated?: string | null): string => {
  if (error) return t("agents.identityDetails.evidence.loadFailed");
  if (lastAuthenticated)
    return t("agents.identityDetails.evidence.lastMatch", { date: new Date(lastAuthenticated).toLocaleString() });
  return t("agents.identityDetails.evidence.awaiting");
};

export const AgentIdentityDetails = ({
  agentId,
  identity: value,
  accessToken,
  isAdmin,
}: {
  agentId: string;
  identity: unknown;
  accessToken: string | null;
  isAdmin: boolean;
}) => {
  const { t } = useTranslation();
  const identity = readAgentIdentity(value);
  const { data, isError, isFetching, refetch } = useQuery({
    queryKey: ["agent-identity", agentId, identity],
    queryFn: () =>
      apiClient.get<components["schemas"]["ManagedAgentIdentityStatus"]>(
        `/v1/agents/${encodeURIComponent(agentId)}/identity`,
        {
          accessToken: accessToken ?? "",
        },
      ),
    enabled: Boolean(isAdmin && accessToken && identity),
  });

  if (!identity || !isAdmin) return null;
  const executionLabel = data?.enabled
    ? t("agents.identityDetails.execution.enabled")
    : t("agents.identityDetails.execution.disabled");
  return (
    <section
      aria-label={t("agents.identityDetails.regionLabel")}
      className="mb-6 space-y-2 rounded-lg border border-border p-4"
    >
      <h3 className="font-medium">{t("agents.identityDetails.title")}</h3>
      <p className="text-sm">
        {t("agents.identityDetails.tenant")} <span className="font-mono">{identity.tenant_id}</span>
      </p>
      <>
        <p className="text-sm">
          {t("agents.identityDetails.clientId")} <span className="font-mono">{identity.client_id}</span>
        </p>
        <p className="text-sm">
          {t("agents.identityDetails.servicePrincipal")}{" "}
          {identity.service_principal_id || t("agents.identityDetails.notConfigured")}
        </p>
      </>
      <p className="text-sm">
        {t("agents.identityDetails.executionSummary", {
          execution: data ? executionLabel : t("agents.identityDetails.loading"),
          mode: data?.execution_mode ?? t("agents.identityDetails.loading"),
        })}
      </p>
      <p className="text-sm">
        {data?.identity?.active === false
          ? t("agents.identityDetails.evidence.unbound")
          : authenticationMessage(t, isError, data?.last_authenticated_at)}
      </p>
      <p className="text-xs text-muted-foreground">{t("agents.identityDetails.evidenceNote")}</p>
      <div className="flex items-center gap-4">
        <Button
          variant="outline"
          size="sm"
          disabled={isFetching}
          onClick={() => {
            void refetch();
          }}
        >
          {t("agents.identityDetails.refresh")}
        </Button>
        <a className="text-sm underline" href="/ui/logs/">
          {t("agents.identityDetails.viewLogs")}
        </a>
      </div>
    </section>
  );
};
