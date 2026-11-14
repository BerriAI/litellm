"use client";

import { useQueryStates } from "nuqs";
import { useTranslation } from "react-i18next";
import { ProviderLogo } from "@/components/molecules/models/ProviderLogo";
import { deprecationProvider } from "./DeprecationModel";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { Page } from "@/components/shared/Page";
import { Button, buttonVariants } from "@/components/ui/button";
import { useDeprecationAlerts } from "@/components/ModelDeprecationAlerts/useDeprecationAlerts";
import { hasCapability } from "@/utils/capabilities";
import { migrationUrlParsers, migrationHref } from "./migrationRoute";
import DeprecationEmailSetup from "./DeprecationEmailSetup";
import AlertsInbox from "./AlertsInbox";
import MigrationDetail from "./MigrationDetail";

export default function ModelDeprecationsPanel() {
  const { t } = useTranslation();
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
        <h1 className="text-xl font-semibold">{t("alerts.panel.title")}</h1>
        <p>{t("alerts.panel.adminRequired")}</p>
      </Page>
    );
  return (
    <Page className="h-full gap-3 overflow-y-auto p-0 sm:p-0">
      {params.model && (
        <header className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h1 className="flex items-center gap-2 text-2xl font-semibold tracking-tight">
              {selected && <ProviderLogo provider={deprecationProvider(selected)} className="size-6 shrink-0" />}
              <code className="rounded-md border bg-muted/50 px-2.5 py-1.5 font-mono text-base font-medium break-all">
                {params.model}
              </code>
            </h1>
          </div>
          <a className={buttonVariants({ variant: "outline" })} href={migrationHref()}>
            {t("alerts.panel.back")}
          </a>
        </header>
      )}
      <DeprecationEmailSetup />
      {alerts.isError ? (
        <div role="alert">
          <p>{t("alerts.panel.loadError")}</p>
          <Button onClick={() => void alerts.refetch()}>{t("alerts.panel.retry")}</Button>
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
      {loadingDetail && <p role="status">{t("alerts.panel.loadingAlert")}</p>}
      {missingDetail && <p>{t("alerts.panel.missingAlert")}</p>}
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
