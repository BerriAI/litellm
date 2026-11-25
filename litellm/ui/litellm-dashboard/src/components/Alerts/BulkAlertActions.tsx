import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import {
  AlertDialog,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogCancel,
} from "@/components/ui/alert-dialog";
import { backgroundFetchClient } from "@/lib/http/api";
import type { DeprecationAlert } from "@/components/ModelDeprecationAlerts/deprecationAlerts";
import { AlertStatus, effectiveAlertStatus } from "@/components/ModelDeprecationAlerts/migrationState";
import type { AlertFilter } from "./alertPresentation";
import { TYPE_LABEL_KEYS } from "./alertPresentation";

type BulkStatus = "resolved" | "dismissed";

export default function BulkAlertActions({
  filter,
  state,
  readOnly,
  deprecations,
  disabled,
}: {
  filter: AlertFilter;
  state: AlertStatus;
  readOnly: boolean;
  deprecations: DeprecationAlert[] | undefined;
  disabled: boolean;
}) {
  const { t } = useTranslation();
  const client = useQueryClient();
  const [status, setStatus] = useState<BulkStatus | null>(null);
  const mutation = useMutation({
    mutationFn: async (next: BulkStatus) => {
      if (filter !== "model-deprecations") {
        const { data } = await backgroundFetchClient.PATCH("/alerts", {
          body: { status: next, type: filter === "all" ? undefined : filter },
        });
        if (!data) throw new Error("Could not update alerts");
      }
      for (const alert of deprecations ?? []) {
        if (effectiveAlertStatus(alert) !== AlertStatus.Open) continue;
        const { data } = await backgroundFetchClient.PUT("/alerting/model-deprecations/state", {
          body: {
            model: alert.usageModel,
            deprecation_date: alert.model.deprecation_date,
            status: next,
            last_seen_date: alert.lastUsedDate,
            last_seen_requests: alert.lastDayRequests,
          },
        });
        if (!data) throw new Error("Could not update model deprecation alerts");
      }
    },
    onSuccess: () => setStatus(null),
    onSettled: () =>
      Promise.all(["alerts", "alert", "deprecationAlerts"].map((key) => client.invalidateQueries({ queryKey: [key] }))),
  });
  function choose(next: BulkStatus) {
    mutation.reset();
    setStatus(next);
  }
  if (readOnly || state !== AlertStatus.Open) return null;
  return (
    <>
      <Button variant="outline" size="sm" disabled={disabled || mutation.isPending} onClick={() => choose("resolved")}>
        {t("alerts.bulk.resolve")}
      </Button>
      <Button variant="outline" size="sm" disabled={disabled || mutation.isPending} onClick={() => choose("dismissed")}>
        {t("alerts.bulk.dismiss")}
      </Button>
      <AlertDialog open={status !== null} onOpenChange={(open) => !open && !mutation.isPending && setStatus(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>
              {t(status === "resolved" ? "alerts.bulk.resolve" : "alerts.bulk.dismiss")}
            </AlertDialogTitle>
            <AlertDialogDescription>
              {t("alerts.bulk.description", { type: t(TYPE_LABEL_KEYS[filter]) })}
            </AlertDialogDescription>
          </AlertDialogHeader>
          {mutation.isError && (
            <p role="alert" className="text-sm text-destructive">
              {t("alerts.bulk.error")}
            </p>
          )}
          <AlertDialogFooter>
            <AlertDialogCancel disabled={mutation.isPending}>{t("alerts.bulk.cancel")}</AlertDialogCancel>
            <Button disabled={mutation.isPending} onClick={() => status && mutation.mutate(status)}>
              {mutation.isPending
                ? t("alerts.bulk.saving")
                : t(status === "resolved" ? "alerts.bulk.resolve" : "alerts.bulk.dismiss")}
            </Button>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}
