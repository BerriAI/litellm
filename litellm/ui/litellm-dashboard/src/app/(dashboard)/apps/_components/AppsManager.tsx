"use client";

import { useMemo } from "react";
import { parseAsString, useQueryState } from "nuqs";
import { ErrorState } from "@/components/shared/ErrorState";
import { type UseQueryResult } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import useAuthorized from "../../hooks/useAuthorized";
import { extractProxyErrorMessage } from "@/lib/http/client";
import { AdminOnlyNotice } from "@/components/shared/AdminOnlyNotice";
import type { App, AppResource } from "./appForms";
import { useManagedApps } from "./useManagedApps";
import { applicationRows, grantedApplicationRows } from "./applications";
import { readsEveryApp, useGrantedApps } from "./useGrantedApps";
import ApplicationsTable from "./ApplicationsTable";
import AppSettingsDialog from "./AppSettingsDialog";
import { Card } from "@/components/ui/card";
import { Page } from "@/components/shared/Page";
import { PageHeader, PageHeaderTitle, PageHeaderDescription } from "@/components/shared/PageHeader";
import AppDefinitionSetting from "./AppDefinitionSetting";

export default function AppsManager() {
  const { t } = useTranslation();
  const { accessToken, userRole, isViewOnly, isLoading } = useAuthorized();
  if (isLoading) return <p role="status">{t("apps.loading")}</p>;
  return accessToken && readsEveryApp(userRole, isViewOnly) ? <ManagedApps /> : <GranteeApps />;
}

function GranteeApps() {
  const { t } = useTranslation();
  const granted = useGrantedApps();
  if (granted.apps.isLoading) return <p role="status">{t("apps.loading")}</p>;
  if (granted.hasGrant) return <GrantedApps apps={granted.apps.data ?? []} />;
  if (!granted.failed) return <AdminOnlyNotice pageTitle={t("apps.title")} />;
  return (
    <Page>
      <ErrorState message={extractProxyErrorMessage(granted.apps.error)} onRetry={() => void granted.apps.refetch()} />
    </Page>
  );
}

function ManagedApps() {
  const { t } = useTranslation();
  const { definition, resourceType, resources, apps } = useManagedApps();
  const [selected, setSelected] = useQueryState("app", parseAsString);
  const resource = resources.data?.find((row) => row.resource_id === selected);
  if (selected && [definition, resources].some((query) => query.isLoading))
    return <p role="status">{t("apps.loading")}</p>;
  return (
    <Page>
      <PageHeader>
        <PageHeaderTitle>{t("apps.title")}</PageHeaderTitle>
        <PageHeaderDescription>{t("apps.settingsHint")}</PageHeaderDescription>
      </PageHeader>
      {resourceType && (
        <ApplicationsList resources={resources} apps={apps} onOpen={(resourceId) => void setSelected(resourceId)} />
      )}
      {resource && (
        <AppSettingsDialog
          key={`${resource.resource_type}:${resource.resource_id}:${resource.app_id ?? "new"}`}
          resource={resource}
          onClose={() => void setSelected(null)}
        />
      )}
      <section className={resourceType ? "border-t pt-6" : undefined}>
        <AppDefinitionSetting />
      </section>
    </Page>
  );
}

function GrantedApps({ apps }: { apps: readonly App[] }) {
  const { t } = useTranslation();
  return (
    <Page>
      <PageHeader>
        <PageHeaderTitle>{t("apps.title")}</PageHeaderTitle>
        <PageHeaderDescription>{t("apps.grantedHint")}</PageHeaderDescription>
      </PageHeader>
      <Card className="gap-0 overflow-hidden py-0" aria-label={t("apps.list.title")}>
        <ApplicationsTable rows={grantedApplicationRows(apps)} loading={false} />
      </Card>
    </Page>
  );
}

function ApplicationsList({
  resources,
  apps,
  onOpen,
}: {
  resources: UseQueryResult<AppResource[]>;
  apps: UseQueryResult<App[]>;
  onOpen: (resourceId: string) => void;
}) {
  const { t } = useTranslation();
  const rows = useMemo(() => applicationRows(resources.data ?? [], apps.data ?? [], ""), [resources.data, apps.data]);
  return (
    <Card className="gap-0 overflow-hidden py-0" aria-label={t("apps.list.title")}>
      {resources.isError || apps.isError ? (
        <div className="p-3">
          <ErrorState
            message={extractProxyErrorMessage(resources.error ?? apps.error)}
            onRetry={() => void (resources.isError ? resources.refetch() : apps.refetch())}
          />
        </div>
      ) : (
        <ApplicationsTable rows={rows} loading={resources.isPending || apps.isPending} onOpen={onOpen} />
      )}
    </Card>
  );
}
