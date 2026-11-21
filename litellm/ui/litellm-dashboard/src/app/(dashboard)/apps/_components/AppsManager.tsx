"use client";

import { useMemo, useState } from "react";
import { parseAsString, useQueryState } from "nuqs";
import { ErrorState } from "@/components/shared/ErrorState";
import { useQuery, type UseQueryResult } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import useAuthorized from "../../hooks/useAuthorized";
import { fetchClient } from "@/lib/http/api";
import { extractProxyErrorMessage } from "@/lib/http/client";
import { AdminOnlyNotice } from "@/components/shared/AdminOnlyNotice";
import type { App, AppResource } from "./appForms";
import { useManagedApps } from "./useManagedApps";
import { applicationRows, grantedApplicationRows } from "./applications";
import { readsEveryApp, useGrantedApps } from "./useGrantedApps";
import ApplicationsTable from "./ApplicationsTable";
import AppSettingsForm from "./AppSettingsForm";
import AppDefinitionSetting from "./AppDefinitionSetting";
import { useOverallAppBudgetGroups } from "./useAppBudgetGroups";
import { CustomerBudgetDialog } from "../../customers/_components/CustomerBudgetDialog";

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
    <main className="mx-auto w-full max-w-[1920px] px-4 py-6 sm:px-6">
      <ErrorState message={extractProxyErrorMessage(granted.apps.error)} onRetry={() => void granted.apps.refetch()} />
    </main>
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
    <main className="mx-auto w-full max-w-[1920px] space-y-6 px-4 py-6 sm:px-6">
      <header>
        <h1 className="text-lg font-semibold">{t("apps.title")}</h1>
        <p className="mt-1 text-sm text-muted-foreground">{t("apps.settingsHint")}</p>
      </header>
      {resourceType && (
        <ApplicationsList resources={resources} apps={apps} onOpen={(resourceId) => void setSelected(resourceId)} />
      )}
      {resource && (
        <ResourceDefaults
          key={`${resource.resource_type}:${resource.resource_id}:${resource.app_id ?? "new"}`}
          resource={resource}
          onClose={() => void setSelected(null)}
        />
      )}
      <section className={resourceType ? "border-t pt-6" : undefined}>
        <AppDefinitionSetting />
      </section>
    </main>
  );
}

function GrantedApps({ apps }: { apps: readonly App[] }) {
  const { t } = useTranslation();
  return (
    <main className="mx-auto w-full max-w-[1920px] space-y-6 px-4 py-6 sm:px-6">
      <header>
        <h1 className="text-lg font-semibold">{t("apps.title")}</h1>
        <p className="mt-1 text-sm text-muted-foreground">{t("apps.grantedHint")}</p>
      </header>
      <section className="rounded-lg border bg-card" aria-label={t("apps.list.title")}>
        <ApplicationsTable rows={grantedApplicationRows(apps)} loading={false} />
      </section>
    </main>
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
    <section className="rounded-lg border bg-card" aria-label={t("apps.list.title")}>
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
    </section>
  );
}

function ResourceDefaults({ resource, onClose }: { resource: AppResource; onClose: () => void }) {
  const { t } = useTranslation();
  const [pending, setPending] = useState(false);
  const groups = useOverallAppBudgetGroups();
  const app = useQuery({
    queryKey: ["app", resource.app_id],
    queryFn: async () => {
      const response = await fetchClient.GET("/management/v1/apps/{app_id}", {
        params: { path: { app_id: resource.app_id! } },
      });
      return response.data;
    },
    enabled: resource.app_id !== null,
  });
  const loading = groups.isPending || (resource.app_id && app.isPending);
  const failed = groups.isError || app.isError;
  const ready = !loading && !failed;
  return (
    <CustomerBudgetDialog title={t("apps.editSettings", { alias: resource.alias })} pending={pending} onClose={onClose}>
      {loading && (
        <p role="status" className="p-6">
          {t("apps.loading")}
        </p>
      )}
      {!loading && failed && (
        <div className="p-6">
          <ErrorState
            message={extractProxyErrorMessage(groups.error ?? app.error)}
            onRetry={() => void (groups.isError ? groups.refetch() : app.refetch())}
          />
        </div>
      )}
      {ready && groups.data && (
        <AppSettingsForm
          resource={resource}
          app={app.data}
          profiles={groups.data.default_budgets}
          onClose={onClose}
          onPendingChange={setPending}
        />
      )}
    </CustomerBudgetDialog>
  );
}
