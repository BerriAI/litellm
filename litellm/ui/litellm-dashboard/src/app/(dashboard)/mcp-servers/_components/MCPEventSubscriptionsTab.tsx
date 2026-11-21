"use client";

import React, { useState } from "react";
import { useTranslation } from "react-i18next";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { RefreshCw, Send, XCircle } from "lucide-react";
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
import {
  fetchMCPEventSubscriptions,
  sendMCPEventTestDelivery,
  terminateMCPEventSubscription,
} from "@/components/networking";
import type {
  MCPEventSubscription,
  MCPEventSubscriptionsResponse,
  MCPEventTestDeliveryResult,
} from "@/components/mcp_tools/types";
import { createQueryKeys } from "@/app/(dashboard)/hooks/common/queryKeysFactory";
import { argumentsSummary, canSendTestEvent, canTerminate, statusTone } from "./eventSubscriptionView";

const eventSubscriptionKeys = createQueryKeys("mcpEventSubscriptions");
const REFETCH_INTERVAL_MS = 15000;

const TONE_VARIANTS = {
  success: "default",
  warning: "secondary",
  muted: "outline",
  destructive: "destructive",
} as const;

function SubscriptionRow({
  subscription,
  canManage,
  onTest,
  onTerminate,
}: {
  subscription: MCPEventSubscription;
  canManage: boolean;
  onTest: (subscription: MCPEventSubscription) => void;
  onTerminate: (subscription: MCPEventSubscription) => void;
}) {
  const { t } = useTranslation();
  return (
    <TableRow>
      <TableCell className="font-mono text-xs">{subscription.event_name}</TableCell>
      <TableCell className="font-mono text-xs">{subscription.server_id}</TableCell>
      <TableCell className="text-xs">{argumentsSummary(subscription.arguments)}</TableCell>
      <TableCell className="font-mono text-xs">{subscription.callback_host}</TableCell>
      <TableCell>
        <Badge variant={TONE_VARIANTS[statusTone(subscription.status)]}>
          {t(`mcpServers.eventSubscriptionsTab.status.${subscription.status}`)}
        </Badge>
      </TableCell>
      <TableCell className="text-xs">{subscription.user_id ?? subscription.team_id ?? "-"}</TableCell>
      <TableCell className="text-xs">{subscription.last_delivery_at ?? "-"}</TableCell>
      <TableCell className="text-xs">
        {subscription.last_error ? t(`mcpServers.eventSubscriptionsTab.failure.${subscription.last_error}`) : "-"}
      </TableCell>
      <TableCell className="text-xs">{subscription.refresh_before ?? t("mcpServers.eventSubscriptionsTab.noExpiry")}</TableCell>
      {canManage ? (
        <TableCell className="space-x-2 text-right">
          <Button
            variant="outline"
            size="sm"
            disabled={!canSendTestEvent(subscription)}
            onClick={() => onTest(subscription)}
            aria-label={t("mcpServers.eventSubscriptionsTab.test.ariaLabel", { id: subscription.id })}
          >
            <Send className="size-4" />
            {t("mcpServers.eventSubscriptionsTab.test.label")}
          </Button>
          <Button
            variant="outline"
            size="sm"
            disabled={!canTerminate(subscription)}
            onClick={() => onTerminate(subscription)}
            aria-label={t("mcpServers.eventSubscriptionsTab.terminate.ariaLabel", { id: subscription.id })}
          >
            <XCircle className="size-4" />
            {t("mcpServers.eventSubscriptionsTab.terminate.label")}
          </Button>
        </TableCell>
      ) : null}
    </TableRow>
  );
}

function SubscriptionsTable({
  data,
  canManage,
  onTest,
  onTerminate,
}: {
  data: MCPEventSubscriptionsResponse;
  canManage: boolean;
  onTest: (subscription: MCPEventSubscription) => void;
  onTerminate: (subscription: MCPEventSubscription) => void;
}) {
  const { t } = useTranslation();
  if (data.subscriptions.length === 0) {
    return (
      <div className="rounded-lg border border-dashed border-border bg-card p-12 text-center">
        <p className="text-sm text-muted-foreground">{t("mcpServers.eventSubscriptionsTab.empty")}</p>
      </div>
    );
  }
  return (
    <section
      aria-label={t("mcpServers.eventSubscriptionsTab.regionLabel")}
      className="rounded-lg border border-border bg-card"
    >
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>{t("mcpServers.eventSubscriptionsTab.columns.event")}</TableHead>
            <TableHead>{t("mcpServers.eventSubscriptionsTab.columns.server")}</TableHead>
            <TableHead>{t("mcpServers.eventSubscriptionsTab.columns.arguments")}</TableHead>
            <TableHead>{t("mcpServers.eventSubscriptionsTab.columns.callback")}</TableHead>
            <TableHead>{t("mcpServers.eventSubscriptionsTab.columns.status")}</TableHead>
            <TableHead>{t("mcpServers.eventSubscriptionsTab.columns.owner")}</TableHead>
            <TableHead>{t("mcpServers.eventSubscriptionsTab.columns.lastDelivery")}</TableHead>
            <TableHead>{t("mcpServers.eventSubscriptionsTab.columns.lastError")}</TableHead>
            <TableHead>{t("mcpServers.eventSubscriptionsTab.columns.expires")}</TableHead>
            {canManage ? (
              <TableHead className="text-right">{t("mcpServers.eventSubscriptionsTab.columns.actions")}</TableHead>
            ) : null}
          </TableRow>
        </TableHeader>
        <TableBody>
          {data.subscriptions.map((subscription) => (
            <SubscriptionRow
              key={subscription.id}
              subscription={subscription}
              canManage={canManage}
              onTest={onTest}
              onTerminate={onTerminate}
            />
          ))}
        </TableBody>
      </Table>
    </section>
  );
}

function TestResultAlert({ result }: { result: MCPEventTestDeliveryResult }) {
  const { t } = useTranslation();
  if (result.delivered) {
    return (
      <Alert>
        <AlertTitle>{t("mcpServers.eventSubscriptionsTab.test.delivered", { status: result.status ?? "" })}</AlertTitle>
      </Alert>
    );
  }
  return (
    <Alert variant="destructive">
      <AlertTitle>{t("mcpServers.eventSubscriptionsTab.test.failed")}</AlertTitle>
      <AlertDescription>
        {result.error
          ? t(`mcpServers.eventSubscriptionsTab.failure.${result.error}`)
          : t("mcpServers.eventSubscriptionsTab.test.httpStatus", { status: result.status ?? "" })}
      </AlertDescription>
    </Alert>
  );
}

interface MCPEventSubscriptionsTabProps {
  accessToken: string | null;
  canManage: boolean;
}

export function MCPEventSubscriptionsTab({ accessToken, canManage }: MCPEventSubscriptionsTabProps) {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const [pending, setPending] = useState<MCPEventSubscription | null>(null);
  const queryOptions = {
    queryKey: eventSubscriptionKeys.lists(),
    queryFn: () => fetchMCPEventSubscriptions(accessToken!),
    enabled: !!accessToken,
    refetchInterval: REFETCH_INTERVAL_MS,
  };
  const { data, error, isLoading, isFetching, refetch } = useQuery<MCPEventSubscriptionsResponse, Error>(queryOptions);
  const terminate = useMutation<MCPEventSubscription, Error, string>({
    mutationFn: (subscriptionId) => terminateMCPEventSubscription(accessToken!, subscriptionId),
    onSettled: () => queryClient.invalidateQueries({ queryKey: eventSubscriptionKeys.lists() }),
  });
  const test = useMutation<MCPEventTestDeliveryResult, Error, string>({
    mutationFn: (subscriptionId) => sendMCPEventTestDelivery(accessToken!, subscriptionId),
  });
  const confirmTerminate = () => {
    if (pending === null) return;
    terminate.mutate(pending.id);
    setPending(null);
  };

  return (
    <div className="mt-4 space-y-4" data-testid="mcp-event-subscriptions-tab">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-base font-semibold text-foreground">{t("mcpServers.eventSubscriptionsTab.title")}</h2>
          <p className="text-sm text-muted-foreground">{t("mcpServers.eventSubscriptionsTab.description")}</p>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={() => refetch()}
          disabled={isFetching}
          aria-label={t("mcpServers.eventSubscriptionsTab.refresh")}
        >
          <RefreshCw className={`size-4 ${isFetching ? "animate-spin" : ""}`} />
          {t("mcpServers.eventSubscriptionsTab.refresh")}
        </Button>
      </div>
      {terminate.isError ? (
        <Alert variant="destructive">
          <AlertTitle>{t("mcpServers.eventSubscriptionsTab.terminate.error")}</AlertTitle>
          <AlertDescription>{terminate.error.message}</AlertDescription>
        </Alert>
      ) : null}
      {test.isError ? (
        <Alert variant="destructive">
          <AlertTitle>{t("mcpServers.eventSubscriptionsTab.test.failed")}</AlertTitle>
          <AlertDescription>{test.error.message}</AlertDescription>
        </Alert>
      ) : null}
      {test.isSuccess ? <TestResultAlert result={test.data} /> : null}
      {isLoading ? (
        <div
          role="status"
          className="flex items-center justify-center gap-3 rounded-lg border border-dashed border-border bg-card p-12"
        >
          <UiLoadingSpinner className="size-6 text-muted-foreground" />
          <p className="text-sm text-muted-foreground">{t("mcpServers.eventSubscriptionsTab.loading")}</p>
        </div>
      ) : null}
      {error ? (
        <Alert variant="destructive">
          <AlertTitle>{t("mcpServers.eventSubscriptionsTab.loadError")}</AlertTitle>
          <AlertDescription>{error.message}</AlertDescription>
        </Alert>
      ) : null}
      {data ? (
        <SubscriptionsTable
          data={data}
          canManage={canManage}
          onTest={(subscription) => test.mutate(subscription.id)}
          onTerminate={setPending}
        />
      ) : null}
      <AlertDialog open={pending !== null} onOpenChange={(open) => !open && setPending(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{t("mcpServers.eventSubscriptionsTab.confirm.title")}</AlertDialogTitle>
            <AlertDialogDescription>
              {pending ? t("mcpServers.eventSubscriptionsTab.confirm.body", { event: pending.event_name }) : ""}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <Button variant="outline" onClick={() => setPending(null)}>
              {t("mcpServers.eventSubscriptionsTab.confirm.cancel")}
            </Button>
            <Button variant="destructive" onClick={confirmTerminate} disabled={terminate.isPending}>
              {t("mcpServers.eventSubscriptionsTab.confirm.confirm")}
            </Button>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}

export default MCPEventSubscriptionsTab;
