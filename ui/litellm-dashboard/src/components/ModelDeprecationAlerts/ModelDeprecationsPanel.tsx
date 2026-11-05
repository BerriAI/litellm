"use client";

import { useQueryStates } from "nuqs";
import { Bell } from "lucide-react";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { Page } from "@/components/shared/Page";
import { Button, buttonVariants } from "@/components/ui/button";
import { useDeprecationAlerts } from "@/components/ModelDeprecationAlerts/useDeprecationAlerts";
import { hasCapability } from "@/utils/capabilities";
import { migrationUrlParsers, migrationHref } from "./migrationRoute";
import AlertsInbox from "./AlertsInbox";
import MigrationDetail from "./MigrationDetail";

export default function ModelAlertsPage() {
  const { accessToken, userRole, isViewOnly } = useAuthorized();
  const allowed = hasCapability(userRole, "viewGlobalSpend");
  const [params, setParams] = useQueryStates(migrationUrlParsers);
  const alerts = useDeprecationAlerts(accessToken);
  const selected = alerts.data?.find((alert) => alert.usageModel === params.model);
  const loadingDetail = !alerts.isError && Boolean(params.model) && alerts.isPending;
  const detailReady = !alerts.isError && Boolean(params.model) && !alerts.isPending;
  const missingDetail = detailReady && !selected;
  if (!allowed)
    return (
      <Page className="h-full gap-3 overflow-y-auto p-0 sm:p-0">
        <h1 className="text-xl font-semibold">Model deprecations</h1>
        <p>Gateway admin access is required to view affected keys and teams.</p>
      </Page>
    );
  return (
    <Page className="h-full gap-3 overflow-y-auto p-0 sm:p-0">
      {params.model && (
        <header className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h1 className="flex items-center gap-2 text-2xl font-semibold tracking-tight">
              <Bell size={20} />
              {params.model || "Model deprecations"}
            </h1>
            {!params.model && (
              <p className="mt-1 text-sm text-muted-foreground">
                Review impact, contact owners, and resolve each model migration.
              </p>
            )}
          </div>
          {params.model && (
            <a className={buttonVariants({ variant: "outline" })} href={migrationHref()}>
              ← Model deprecations
            </a>
          )}
        </header>
      )}
      {alerts.isError ? (
        <div role="alert">
          <p>Could not load alerts. No alert status has been changed.</p>
          <Button onClick={() => void alerts.refetch()}>Retry</Button>
        </div>
      ) : null}
      {!alerts.isError && !params.model && (
        <AlertsInbox
          alerts={alerts.data ?? []}
          pending={alerts.isPending}
          onRefresh={() => void alerts.refetch()}
          refreshing={alerts.isFetching}
          status={params.state}
          onStatusChange={(state) => void setParams({ state })}
        />
      )}
      {loadingDetail && <p role="status">Loading alert…</p>}
      {missingDetail && <p>No matching model alert was found in the last 30 days.</p>}
      {!alerts.isError && selected && (
        <MigrationDetail
          selected={selected}
          scope={alerts.scope}
          range={alerts.range}
          isViewOnly={Boolean(isViewOnly)}
        />
      )}
    </Page>
  );
}
