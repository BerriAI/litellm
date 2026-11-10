"use client";

import { useTerminology } from "@/contexts/TerminologyContext";
import React, { useState } from "react";
import type { TFunction } from "i18next";
import { useTranslation } from "react-i18next";
import i18n from "@/i18n/i18n";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { RefreshCw, Unplug } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import {
  AlertDialog,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { fetchMCPGatewaySessions, terminateMCPGatewaySessions } from "@/components/networking";
import type {
  MCPGatewaySessionGroupCount,
  MCPGatewaySessionSelector,
  MCPGatewaySessionsResponse,
  MCPGatewaySessionsTerminateResponse,
} from "@/components/mcp_tools/types";
import { createQueryKeys } from "@/app/(dashboard)/hooks/common/queryKeysFactory";

const mcpGatewaySessionKeys = createQueryKeys("mcpGatewaySessions");
const REFETCH_INTERVAL_MS = 15000;

export function formatIdleSeconds(idleSeconds: number, t: TFunction = i18n.t.bind(i18n)): string {
  const total = Math.max(0, Math.floor(idleSeconds));
  if (total < 60) return t("mcpServers.gatewaySessionsTab.idle.seconds", { seconds: total });
  const minutes = Math.floor(total / 60);
  const seconds = total % 60;
  return seconds === 0
    ? t("mcpServers.gatewaySessionsTab.idle.minutes", { minutes })
    : t("mcpServers.gatewaySessionsTab.idle.minutesSeconds", { minutes, seconds });
}

function groupLabel(label: string | null, t: TFunction): string {
  if (label === null) return t("mcpServers.gatewaySessionsTab.unknown");
  return label === "" ? '""' : label;
}

export function describeSelector(selector: MCPGatewaySessionSelector, t: TFunction): string {
  if (selector.user_id !== undefined)
    return t("mcpServers.gatewaySessionsTab.selector.user", { user: groupLabel(selector.user_id, t) });
  return t("mcpServers.gatewaySessionsTab.selector.session", { sessionId: selector.session_id_prefix });
}

export function describeTerminateResult(
  result: MCPGatewaySessionsTerminateResponse,
  t: TFunction = i18n.t.bind(i18n),
): string {
  return t("mcpServers.gatewaySessionsTab.terminateResult", {
    count: result.terminated_sessions,
    pid: result.worker_pid,
  });
}

function StatCard({ label, value }: { label: string; value: number }) {
  return (
    <div className="bg-card border border-border rounded-lg px-4 py-3">
      <div className="text-2xl font-bold text-foreground">{value}</div>
      <div className="text-xs text-muted-foreground mt-0.5">{label}</div>
    </div>
  );
}

function DisconnectUserButton({
  userId,
  onDisconnectUser,
}: {
  userId: string | null;
  onDisconnectUser: (userId: string) => void;
}) {
  const { t } = useTranslation();
  if (userId === null || userId === "") return null;
  return (
    <Button
      variant="outline"
      size="sm"
      onClick={() => onDisconnectUser(userId)}
      aria-label={t("mcpServers.gatewaySessionsTab.disconnectUser.ariaLabel", { user: groupLabel(userId, t) })}
    >
      <Unplug className="size-4" />
      {t("mcpServers.gatewaySessionsTab.disconnectUser.label")}
    </Button>
  );
}

function GroupCountTable({
  title,
  groups,
  labelHeader,
  onDisconnectUser,
}: {
  title: string;
  groups: MCPGatewaySessionGroupCount[];
  labelHeader: string;
  onDisconnectUser?: (userId: string) => void;
}) {
  const { t } = useTranslation();
  return (
    <section aria-label={title} className="rounded-lg border border-border bg-card">
      <h3 className="border-b border-border px-4 py-2 text-sm font-semibold text-foreground">{title}</h3>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>{labelHeader}</TableHead>
            <TableHead className="text-right">{t("mcpServers.gatewaySessionsTab.columns.sessions")}</TableHead>
            {onDisconnectUser ? (
              <TableHead className="text-right">{t("mcpServers.gatewaySessionsTab.columns.actions")}</TableHead>
            ) : null}
          </TableRow>
        </TableHeader>
        <TableBody>
          {groups.map((group) => (
            <TableRow key={group.label ?? "__unknown__"}>
              <TableCell className="font-mono text-xs">{groupLabel(group.label, t)}</TableCell>
              <TableCell className="text-right">{group.count}</TableCell>
              {onDisconnectUser ? (
                <TableCell className="text-right">
                  <DisconnectUserButton userId={group.label} onDisconnectUser={onDisconnectUser} />
                </TableCell>
              ) : null}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </section>
  );
}

function SessionsBody({
  data,
  error,
  isLoading,
  onDisconnect,
}: {
  data: MCPGatewaySessionsResponse | undefined;
  error: Error | null;
  isLoading: boolean;
  onDisconnect: ((selector: MCPGatewaySessionSelector) => void) | null;
}) {
  const term = useTerminology();
  const { t } = useTranslation();
  if (isLoading) {
    return (
      <div
        role="status"
        className="flex items-center justify-center gap-3 rounded-lg border border-dashed border-border bg-card p-12"
      >
        <UiLoadingSpinner className="size-6 text-muted-foreground" />
        <p className="text-sm text-muted-foreground">{t("mcpServers.gatewaySessionsTab.loading")}</p>
      </div>
    );
  }
  if (error) {
    return (
      <Alert variant="destructive">
        <AlertTitle>{t("mcpServers.gatewaySessionsTab.loadError.title")}</AlertTitle>
        <AlertDescription>{error.message}</AlertDescription>
      </Alert>
    );
  }
  if (!data) return null;
  if (data.total_sessions === 0) {
    return (
      <div className="rounded-lg border border-dashed border-border bg-card p-12 text-center">
        <p className="text-sm text-muted-foreground">
          {t("mcpServers.gatewaySessionsTab.empty.noConnections", { pid: data.worker_pid })}{" "}
          {term(t("mcpServers.gatewaySessionsTab.empty.connectHint"))}
        </p>
      </div>
    );
  }
  return (
    <>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        <StatCard label={t("mcpServers.gatewaySessionsTab.stats.liveSessions")} value={data.total_sessions} />
        <StatCard label={term(t("mcpServers.gatewaySessionsTab.stats.aiClients"))} value={data.by_client.length} />
        <StatCard label={t("mcpServers.gatewaySessionsTab.stats.users")} value={data.by_user.length} />
      </div>
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <GroupCountTable
          title={term(t("mcpServers.gatewaySessionsTab.groups.byClient"))}
          labelHeader={t("mcpServers.gatewaySessionsTab.columns.client")}
          groups={data.by_client}
        />
        <GroupCountTable
          title={t("mcpServers.gatewaySessionsTab.groups.byUser")}
          labelHeader={t("mcpServers.gatewaySessionsTab.columns.user")}
          groups={data.by_user}
          onDisconnectUser={onDisconnect ? (userId) => onDisconnect({ user_id: userId }) : undefined}
        />
      </div>
      <section
        aria-label={t("mcpServers.gatewaySessionsTab.liveSessions.regionLabel")}
        className="rounded-lg border border-border bg-card"
      >
        <h3 className="border-b border-border px-4 py-2 text-sm font-semibold text-foreground">
          {t("mcpServers.gatewaySessionsTab.liveSessions.heading", { pid: data.worker_pid })}
        </h3>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>{t("mcpServers.gatewaySessionsTab.columns.session")}</TableHead>
              <TableHead>{t("mcpServers.gatewaySessionsTab.columns.client")}</TableHead>
              <TableHead>{t("mcpServers.gatewaySessionsTab.columns.user")}</TableHead>
              <TableHead>{t("mcpServers.gatewaySessionsTab.columns.keyAlias")}</TableHead>
              <TableHead>{t("mcpServers.gatewaySessionsTab.columns.team")}</TableHead>
              <TableHead>{t("mcpServers.gatewaySessionsTab.columns.clientIp")}</TableHead>
              <TableHead className="text-right">{t("mcpServers.gatewaySessionsTab.columns.idle")}</TableHead>
              <TableHead className="text-right">{t("mcpServers.gatewaySessionsTab.columns.inFlight")}</TableHead>
              {onDisconnect ? (
                <TableHead className="text-right">{t("mcpServers.gatewaySessionsTab.columns.actions")}</TableHead>
              ) : null}
            </TableRow>
          </TableHeader>
          <TableBody>
            {data.sessions.map((session, index) => (
              <TableRow key={`${session.session_id_prefix}-${index}`}>
                <TableCell className="font-mono text-xs">{session.session_id_prefix}</TableCell>
                <TableCell>
                  {session.client_name === null ? (
                    <span className="text-muted-foreground">{t("mcpServers.gatewaySessionsTab.unknown")}</span>
                  ) : (
                    <>
                      <span className="font-mono text-xs">{groupLabel(session.client_name, t)}</span>
                      {session.client_version ? (
                        <span className="ml-1 text-xs text-muted-foreground">v{session.client_version}</span>
                      ) : null}
                    </>
                  )}
                </TableCell>
                <TableCell>
                  {session.user_id === null ? (
                    <span className="text-muted-foreground">{t("mcpServers.gatewaySessionsTab.unknown")}</span>
                  ) : (
                    <>
                      <span className="font-mono text-xs">{session.user_id}</span>
                      {session.user_email ? (
                        <span className="ml-1 text-xs text-muted-foreground">{session.user_email}</span>
                      ) : null}
                    </>
                  )}
                </TableCell>
                <TableCell className="text-xs">{session.key_alias ?? "-"}</TableCell>
                <TableCell className="text-xs">{session.team_alias ?? session.team_id ?? "-"}</TableCell>
                <TableCell className="font-mono text-xs">{session.client_ip || "-"}</TableCell>
                <TableCell className="text-right text-xs">{formatIdleSeconds(session.idle_seconds, t)}</TableCell>
                <TableCell className="text-right text-xs">{session.in_flight_requests}</TableCell>
                {onDisconnect ? (
                  <TableCell className="text-right">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => onDisconnect({ session_id_prefix: session.session_id_prefix })}
                      aria-label={t("mcpServers.gatewaySessionsTab.disconnectSession.ariaLabel", {
                        sessionId: session.session_id_prefix,
                      })}
                    >
                      <Unplug className="size-4" />
                      {t("mcpServers.gatewaySessionsTab.disconnectSession.label")}
                    </Button>
                  </TableCell>
                ) : null}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </section>
    </>
  );
}

interface MCPGatewaySessionsTabProps {
  accessToken: string | null;
  canTerminate: boolean;
}

export function MCPGatewaySessionsTab({ accessToken, canTerminate }: MCPGatewaySessionsTabProps) {
  const term = useTerminology();
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const [pendingSelector, setPendingSelector] = useState<MCPGatewaySessionSelector | null>(null);
  const queryOptions = {
    queryKey: mcpGatewaySessionKeys.lists(),
    queryFn: () => fetchMCPGatewaySessions(accessToken!),
    enabled: !!accessToken,
    refetchInterval: REFETCH_INTERVAL_MS,
  };
  const { data, error, isLoading, isFetching, refetch } = useQuery<MCPGatewaySessionsResponse, Error>(queryOptions);
  const terminate = useMutation<MCPGatewaySessionsTerminateResponse, Error, MCPGatewaySessionSelector>({
    mutationFn: (selector) => terminateMCPGatewaySessions(accessToken!, selector),
    onSettled: () => queryClient.invalidateQueries({ queryKey: mcpGatewaySessionKeys.lists() }),
  });
  const confirmDisconnect = () => {
    if (pendingSelector === null) return;
    terminate.mutate(pendingSelector);
    setPendingSelector(null);
  };

  return (
    <div className="mt-4 space-y-4" data-testid="mcp-gateway-sessions-tab">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-base font-semibold text-foreground">{t("mcpServers.gatewaySessionsTab.header.title")}</h2>
          <p className="text-sm text-muted-foreground">{term(t("mcpServers.gatewaySessionsTab.header.description"))}</p>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={() => refetch()}
          disabled={isFetching}
          aria-label={t("mcpServers.gatewaySessionsTab.refresh.ariaLabel")}
        >
          <RefreshCw className={`size-4 ${isFetching ? "animate-spin" : ""}`} />
          {t("mcpServers.gatewaySessionsTab.refresh.label")}
        </Button>
      </div>

      {terminate.isError ? (
        <Alert variant="destructive">
          <AlertTitle>{t("mcpServers.gatewaySessionsTab.terminateError.title")}</AlertTitle>
          <AlertDescription>{terminate.error.message}</AlertDescription>
        </Alert>
      ) : null}
      {terminate.isSuccess ? (
        <Alert>
          <AlertTitle>{t("mcpServers.gatewaySessionsTab.terminateSuccess.title")}</AlertTitle>
          <AlertDescription>
            {describeTerminateResult(terminate.data, t)} {t("mcpServers.gatewaySessionsTab.terminateSuccess.followUp")}
          </AlertDescription>
        </Alert>
      ) : null}

      <SessionsBody
        data={data}
        error={error}
        isLoading={isLoading}
        onDisconnect={canTerminate ? setPendingSelector : null}
      />

      <AlertDialog open={pendingSelector !== null} onOpenChange={(open) => !open && setPendingSelector(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{t("mcpServers.gatewaySessionsTab.confirm.title")}</AlertDialogTitle>
            <AlertDialogDescription>
              {pendingSelector
                ? `${t("mcpServers.gatewaySessionsTab.confirm.target", { target: describeSelector(pendingSelector, t) })} `
                : ""}
              {t("mcpServers.gatewaySessionsTab.confirm.consequence")}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <Button variant="outline" onClick={() => setPendingSelector(null)}>
              {t("mcpServers.gatewaySessionsTab.confirm.cancel")}
            </Button>
            <Button variant="destructive" onClick={confirmDisconnect} disabled={terminate.isPending}>
              {t("mcpServers.gatewaySessionsTab.confirm.confirm")}
            </Button>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}

export default MCPGatewaySessionsTab;
