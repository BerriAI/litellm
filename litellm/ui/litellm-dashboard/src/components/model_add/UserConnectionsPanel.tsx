"use client";

import { useQueryClient } from "@tanstack/react-query";
import { useCallback, useState } from "react";
import { useTranslation } from "react-i18next";

import { userConnectionsKeys, useUserConnections } from "@/app/(dashboard)/hooks/credentials/useUserConnections";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { userConnectionDeleteCall, UserProviderConnection } from "@/components/networking";
import { Providers } from "@/components/provider_info_helpers";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { toast } from "@/lib/toast";

import GithubCopilotConnectModal from "./GithubCopilotConnectModal";

const providerLabel = (provider: string): string =>
  Providers[provider.toUpperCase() as keyof typeof Providers] ??
  Object.entries(Providers).find(([key]) => key.toLowerCase() === provider.toLowerCase())?.[1] ??
  provider;

export default function UserConnectionsPanel() {
  const { t } = useTranslation();
  const { accessToken } = useAuthorized();
  const queryClient = useQueryClient();
  const { data, isLoading, isError, refetch } = useUserConnections();
  const [connectingCredential, setConnectingCredential] = useState<string | null>(null);
  const connections: readonly UserProviderConnection[] = data?.connections ?? [];
  const showEmpty = !isLoading && !isError && connections.length === 0;

  const refreshConnections = useCallback(
    () => queryClient.invalidateQueries({ queryKey: userConnectionsKeys.all }),
    [queryClient],
  );

  const disconnect = async (credentialName: string) => {
    if (!accessToken) {
      return;
    }
    try {
      await userConnectionDeleteCall(accessToken, credentialName);
      toast.success(t("modelAdd.userConnectionsPanel.disconnected"));
      setConnectingCredential(null);
      await refreshConnections();
    } catch (error) {
      toast.error(t("modelAdd.userConnectionsPanel.disconnectFailed"));
    }
  };

  return (
    <section className="mx-auto flex w-full flex-col gap-3 p-2">
      <div>
        <h3 className="text-base font-semibold">{t("modelAdd.userConnectionsPanel.title")}</h3>
        <p className="text-sm text-muted-foreground">{t("modelAdd.userConnectionsPanel.description")}</p>
      </div>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>{t("modelAdd.userConnectionsPanel.credential")}</TableHead>
            <TableHead>{t("modelAdd.userConnectionsPanel.provider")}</TableHead>
            <TableHead>{t("modelAdd.userConnectionsPanel.status")}</TableHead>
            <TableHead className="text-right">{t("modelAdd.userConnectionsPanel.actions")}</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {isLoading && (
            <TableRow>
              <TableCell colSpan={4} className="text-sm text-muted-foreground">
                {t("modelAdd.userConnectionsPanel.loading")}
              </TableCell>
            </TableRow>
          )}
          {isError && (
            <TableRow>
              <TableCell colSpan={4} className="text-sm">
                <div className="flex items-center justify-between gap-2">
                  <span className="text-muted-foreground">{t("modelAdd.userConnectionsPanel.loadFailed")}</span>
                  <Button variant="outline" size="sm" onClick={() => refetch()}>
                    {t("modelAdd.userConnectionsPanel.retry")}
                  </Button>
                </div>
              </TableCell>
            </TableRow>
          )}
          {showEmpty && (
            <TableRow>
              <TableCell colSpan={4} className="text-sm text-muted-foreground">
                {t("modelAdd.userConnectionsPanel.empty")}
              </TableCell>
            </TableRow>
          )}
          {connections.map((connection) => (
            <TableRow key={connection.credential_name}>
              <TableCell>{connection.credential_name}</TableCell>
              <TableCell>{providerLabel(connection.provider)}</TableCell>
              <TableCell>
                {connection.connected ? (
                  <Badge>{t("modelAdd.userConnectionsPanel.connectedAs", { login: connection.github_login })}</Badge>
                ) : (
                  <Badge variant="outline">{t("modelAdd.userConnectionsPanel.notConnected")}</Badge>
                )}
              </TableCell>
              <TableCell className="text-right">
                {connection.connected ? (
                  <Button variant="outline" size="sm" onClick={() => disconnect(connection.credential_name)}>
                    {t("modelAdd.userConnectionsPanel.disconnect")}
                  </Button>
                ) : (
                  <Button size="sm" onClick={() => setConnectingCredential(connection.credential_name)}>
                    {t("modelAdd.userConnectionsPanel.connect")}
                  </Button>
                )}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
      {connectingCredential !== null && (
        <GithubCopilotConnectModal
          credentialName={connectingCredential}
          onClose={() => setConnectingCredential(null)}
          onConnected={refreshConnections}
          onDisconnect={() => disconnect(connectingCredential)}
        />
      )}
    </section>
  );
}
