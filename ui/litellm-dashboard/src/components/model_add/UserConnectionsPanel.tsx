"use client";

import { useQueryClient } from "@tanstack/react-query";
import { useCallback, useState } from "react";

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
      toast.success("GitHub Copilot disconnected");
      setConnectingCredential(null);
      await refreshConnections();
    } catch (error) {
      toast.error("Failed to disconnect GitHub Copilot");
    }
  };

  return (
    <section className="mx-auto flex w-full flex-col gap-3 p-2">
      <div>
        <h3 className="text-base font-semibold">Your connections</h3>
        <p className="text-sm text-muted-foreground">
          Credentials that use your own account. Connect once, and your requests to models on that credential use your
          own access.
        </p>
      </div>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Credential</TableHead>
            <TableHead>Provider</TableHead>
            <TableHead>Status</TableHead>
            <TableHead className="text-right">Actions</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {isLoading && (
            <TableRow>
              <TableCell colSpan={4} className="text-sm text-muted-foreground">
                Loading connections...
              </TableCell>
            </TableRow>
          )}
          {isError && (
            <TableRow>
              <TableCell colSpan={4} className="text-sm">
                <div className="flex items-center justify-between gap-2">
                  <span className="text-muted-foreground">Could not load your connections.</span>
                  <Button variant="outline" size="sm" onClick={() => refetch()}>
                    Retry
                  </Button>
                </div>
              </TableCell>
            </TableRow>
          )}
          {showEmpty && (
            <TableRow>
              <TableCell colSpan={4} className="text-sm text-muted-foreground">
                No credentials need a personal connection.
              </TableCell>
            </TableRow>
          )}
          {connections.map((connection) => (
            <TableRow key={connection.credential_name}>
              <TableCell>{connection.credential_name}</TableCell>
              <TableCell>{providerLabel(connection.provider)}</TableCell>
              <TableCell>
                {connection.connected ? (
                  <Badge>Connected as @{connection.github_login}</Badge>
                ) : (
                  <Badge variant="outline">Not connected</Badge>
                )}
              </TableCell>
              <TableCell className="text-right">
                {connection.connected ? (
                  <Button variant="outline" size="sm" onClick={() => disconnect(connection.credential_name)}>
                    Disconnect
                  </Button>
                ) : (
                  <Button size="sm" onClick={() => setConnectingCredential(connection.credential_name)}>
                    Connect
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
