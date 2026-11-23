"use client";

import React, { useState } from "react";
import { useTranslation } from "react-i18next";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Send, XCircle } from "lucide-react";
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
import { sendMCPEventTestDelivery, terminateMCPEventSubscription } from "@/components/networking";
import type { MCPEventSubscription, MCPEventTestDeliveryResult } from "@/components/mcp_tools/types";
import { canSendTestEvent, canTerminate } from "./eventSubscriptionView";

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

export function SubscriptionActions({
  subscription,
  accessToken,
}: {
  subscription: MCPEventSubscription;
  accessToken: string;
}) {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const [confirm, setConfirm] = useState(false);
  const terminate = useMutation({
    mutationFn: () => terminateMCPEventSubscription(accessToken, subscription.id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["mcpEventSubscriptions"] }),
  });
  const test = useMutation({ mutationFn: () => sendMCPEventTestDelivery(accessToken, subscription.id) });
  return (
    <>
      <Button
        variant="outline"
        size="sm"
        disabled={test.isPending || !canSendTestEvent(subscription)}
        onClick={() => test.mutate()}
        aria-label={t("mcpServers.eventSubscriptionsTab.test.ariaLabel", { id: subscription.id })}
      >
        <Send />
        {t("mcpServers.eventSubscriptionsTab.test.label")}
      </Button>
      <Button
        variant="outline"
        size="sm"
        disabled={terminate.isPending || !canTerminate(subscription)}
        onClick={() => setConfirm(true)}
        aria-label={t("mcpServers.eventSubscriptionsTab.terminate.ariaLabel", { id: subscription.id })}
      >
        <XCircle />
        {t("mcpServers.eventSubscriptionsTab.terminate.label")}
      </Button>
      {test.data && <TestResultAlert result={test.data} />}
      {(terminate.error || test.error) && <p role="alert">{(terminate.error || test.error)?.message}</p>}
      <AlertDialog open={confirm} onOpenChange={setConfirm}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{t("mcpServers.eventSubscriptionsTab.confirm.title")}</AlertDialogTitle>
            <AlertDialogDescription>
              {t("mcpServers.eventSubscriptionsTab.confirm.body", { event: subscription.event_name })}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <Button variant="outline" onClick={() => setConfirm(false)}>
              {t("mcpServers.eventSubscriptionsTab.confirm.cancel")}
            </Button>
            <Button
              variant="destructive"
              onClick={() => {
                setConfirm(false);
                terminate.mutate();
              }}
            >
              {t("mcpServers.eventSubscriptionsTab.confirm.confirm")}
            </Button>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}
