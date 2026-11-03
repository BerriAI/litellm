"use client";

import React, { useState } from "react";
import type { TFunction } from "i18next";
import { useTranslation } from "react-i18next";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { RefreshCw, ShieldOff } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import {
  AlertDialog,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { fetchMCPServerUserCredentials, revokeMCPServerUserCredential } from "@/components/networking";
import type { MCPServerUserCredentialListItem } from "@/components/mcp_tools/types";
import { createQueryKeys } from "@/app/(dashboard)/hooks/common/queryKeysFactory";

const mcpServerUserCredentialKeys = createQueryKeys("mcpServerUserCredentials");

export function credentialTypeLabel(
  credentialType: MCPServerUserCredentialListItem["credential_type"],
  t: TFunction,
): string {
  return credentialType === "oauth2" ? "OAuth2" : t("mcpServers.serverUserCredentialsPanel.credentialTypes.byokApiKey");
}

export function formatTimestamp(value: string | null): string {
  if (value === null) return "-";
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleString();
}

function CredentialsBody({
  items,
  error,
  isLoading,
  onRevoke,
}: {
  items: MCPServerUserCredentialListItem[] | undefined;
  error: Error | null;
  isLoading: boolean;
  onRevoke: ((item: MCPServerUserCredentialListItem) => void) | null;
}) {
  const { t } = useTranslation();
  if (isLoading) {
    return (
      <div
        role="status"
        className="flex items-center justify-center gap-3 rounded-lg border border-dashed border-border bg-card p-12"
      >
        <UiLoadingSpinner className="size-6 text-muted-foreground" />
        <p className="text-sm text-muted-foreground">{t("mcpServers.serverUserCredentialsPanel.loading")}</p>
      </div>
    );
  }
  if (error) {
    return (
      <Alert variant="destructive">
        <AlertTitle>{t("mcpServers.serverUserCredentialsPanel.loadError")}</AlertTitle>
        <AlertDescription>{error.message}</AlertDescription>
      </Alert>
    );
  }
  if (!items) return null;
  if (items.length === 0) {
    return (
      <div className="rounded-lg border border-dashed border-border bg-card p-12 text-center">
        <p className="text-sm text-muted-foreground">{t("mcpServers.serverUserCredentialsPanel.empty")}</p>
      </div>
    );
  }
  return (
    <section
      aria-label={t("mcpServers.serverUserCredentialsPanel.tableLabel")}
      className="rounded-lg border border-border bg-card"
    >
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>{t("mcpServers.serverUserCredentialsPanel.columns.user")}</TableHead>
            <TableHead>{t("mcpServers.serverUserCredentialsPanel.columns.type")}</TableHead>
            <TableHead>{t("mcpServers.serverUserCredentialsPanel.columns.connected")}</TableHead>
            <TableHead>{t("mcpServers.serverUserCredentialsPanel.columns.expires")}</TableHead>
            <TableHead>{t("mcpServers.serverUserCredentialsPanel.columns.updated")}</TableHead>
            {onRevoke ? (
              <TableHead className="text-right">{t("mcpServers.serverUserCredentialsPanel.columns.actions")}</TableHead>
            ) : null}
          </TableRow>
        </TableHeader>
        <TableBody>
          {items.map((item) => (
            <TableRow key={item.user_id}>
              <TableCell className="font-mono text-xs">{item.user_id}</TableCell>
              <TableCell>
                <Badge variant="secondary">{credentialTypeLabel(item.credential_type, t)}</Badge>
              </TableCell>
              <TableCell className="text-xs">{formatTimestamp(item.connected_at)}</TableCell>
              <TableCell className="text-xs">{formatTimestamp(item.expires_at)}</TableCell>
              <TableCell className="text-xs">{formatTimestamp(item.updated_at)}</TableCell>
              {onRevoke ? (
                <TableCell className="text-right">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => onRevoke(item)}
                    aria-label={t("mcpServers.serverUserCredentialsPanel.revokeAriaLabel", { user: item.user_id })}
                  >
                    <ShieldOff className="size-4" />
                    {t("mcpServers.serverUserCredentialsPanel.revoke")}
                  </Button>
                </TableCell>
              ) : null}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </section>
  );
}

interface MCPServerUserCredentialsPanelProps {
  serverId: string;
  accessToken: string | null;
  canRevoke: boolean;
}

export function MCPServerUserCredentialsPanel({
  serverId,
  accessToken,
  canRevoke,
}: MCPServerUserCredentialsPanelProps) {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const [pendingItem, setPendingItem] = useState<MCPServerUserCredentialListItem | null>(null);
  const queryKey = mcpServerUserCredentialKeys.detail(serverId);
  const { data, error, isLoading, isFetching, refetch } = useQuery<MCPServerUserCredentialListItem[], Error>({
    queryKey,
    queryFn: () => fetchMCPServerUserCredentials(accessToken!, serverId),
    enabled: !!accessToken,
  });
  const revoke = useMutation<void, Error, MCPServerUserCredentialListItem>({
    mutationFn: (item) => revokeMCPServerUserCredential(accessToken!, serverId, item.user_id, item.credential_type),
    onSettled: () => queryClient.invalidateQueries({ queryKey }),
  });
  const confirmRevoke = () => {
    if (pendingItem === null) return;
    revoke.mutate(pendingItem);
    setPendingItem(null);
  };

  return (
    <div className="space-y-4" data-testid="mcp-server-user-credentials-panel">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-lg font-medium">{t("mcpServers.serverUserCredentialsPanel.title")}</h2>
          <p className="text-sm text-muted-foreground">{t("mcpServers.serverUserCredentialsPanel.description")}</p>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={() => refetch()}
          disabled={isFetching}
          aria-label={t("mcpServers.serverUserCredentialsPanel.refreshAriaLabel")}
        >
          <RefreshCw className={`size-4 ${isFetching ? "animate-spin" : ""}`} />
          {t("mcpServers.serverUserCredentialsPanel.refresh")}
        </Button>
      </div>

      {revoke.isError ? (
        <Alert variant="destructive">
          <AlertTitle>{t("mcpServers.serverUserCredentialsPanel.revokeError")}</AlertTitle>
          <AlertDescription>{revoke.error.message}</AlertDescription>
        </Alert>
      ) : null}
      {revoke.isSuccess ? (
        <Alert>
          <AlertTitle>{t("mcpServers.serverUserCredentialsPanel.revokeSuccess.title")}</AlertTitle>
          <AlertDescription>
            {t("mcpServers.serverUserCredentialsPanel.revokeSuccess.description", {
              type: credentialTypeLabel(revoke.variables.credential_type, t),
              user: revoke.variables.user_id,
            })}
          </AlertDescription>
        </Alert>
      ) : null}

      <CredentialsBody items={data} error={error} isLoading={isLoading} onRevoke={canRevoke ? setPendingItem : null} />

      <AlertDialog open={pendingItem !== null} onOpenChange={(open) => !open && setPendingItem(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{t("mcpServers.serverUserCredentialsPanel.revokeDialog.title")}</AlertDialogTitle>
            <AlertDialogDescription>
              {pendingItem
                ? `${t("mcpServers.serverUserCredentialsPanel.revokeDialog.target", {
                    type: credentialTypeLabel(pendingItem.credential_type, t),
                    user: pendingItem.user_id,
                  })} `
                : ""}
              {t("mcpServers.serverUserCredentialsPanel.revokeDialog.consequence")}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <Button variant="outline" onClick={() => setPendingItem(null)}>
              {t("mcpServers.serverUserCredentialsPanel.revokeDialog.cancel")}
            </Button>
            <Button variant="destructive" onClick={confirmRevoke} disabled={revoke.isPending}>
              {t("mcpServers.serverUserCredentialsPanel.revokeDialog.confirm")}
            </Button>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}

export default MCPServerUserCredentialsPanel;
