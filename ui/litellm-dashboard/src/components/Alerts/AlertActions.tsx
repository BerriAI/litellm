import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Check, RotateCcw, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { backgroundFetchClient } from "@/lib/http/api";
import { AlertStatus } from "@/components/ModelDeprecationAlerts/migrationState";
import { AlertItem, presentAlert } from "./alertPresentation";

export default function AlertActions({ item, readOnly }: { item: AlertItem; readOnly: boolean }) {
  const client = useQueryClient();
  const mutation = useMutation({
    mutationFn: async (status: AlertStatus) => {
      if (item.source === "performance") {
        const result = await backgroundFetchClient.PATCH("/alerts/{alert_id}", {
          params: { path: { alert_id: item.alert.id } },
          body: { status },
        });
        if (!result.data) throw new Error("Could not save alert status");
        return;
      }
      const alert = item.alert;
      const result = await backgroundFetchClient.PUT("/alerting/model-deprecations/state", {
        body: {
          model: alert.usageModel,
          deprecation_date: alert.model.deprecation_date,
          status,
          last_seen_date: alert.lastUsedDate,
          last_seen_requests: alert.lastDayRequests,
        },
      });
      if (!result.data) throw new Error("Could not save alert status");
    },
    onSuccess: () =>
      Promise.all(
        (item.source === "performance" ? ["alerts", "alert"] : ["deprecationAlerts"]).map((key) =>
          client.invalidateQueries({ queryKey: [key] }),
        ),
      ),
  });
  const status = presentAlert(item).status;
  if (readOnly) return <span className="text-sm capitalize text-muted-foreground">{status}</span>;
  return (
    <div className="space-y-2">
      <div className="flex shrink-0 items-center gap-2">
        {status === "open" ? (
          <>
            <Button
              variant="outline"
              size="sm"
              disabled={mutation.isPending}
              onClick={() => mutation.mutate(AlertStatus.Resolved)}
            >
              <Check className="size-3.5" />
              Mark resolved
            </Button>
            <Button
              variant="ghost"
              size="sm"
              disabled={mutation.isPending}
              onClick={() => mutation.mutate(AlertStatus.Dismissed)}
            >
              <X className="size-3.5" />
              Dismiss
            </Button>
          </>
        ) : (
          <>
            <span className="text-sm capitalize text-muted-foreground">{status}</span>
            <Button
              variant="outline"
              size="sm"
              disabled={mutation.isPending}
              onClick={() => mutation.mutate(AlertStatus.Open)}
            >
              <RotateCcw className="size-3.5" />
              Reopen
            </Button>
          </>
        )}
      </div>
      {mutation.isError && (
        <p role="alert" className="text-xs text-destructive">
          Could not save status. Please try again.
        </p>
      )}
    </div>
  );
}
