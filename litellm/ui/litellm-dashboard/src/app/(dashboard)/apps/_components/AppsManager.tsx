"use client";

import { budgetEditorFields, permanentBudgetEditorValue } from "../../customers/_components/customerBudgetForm";

import { useMemo, useState } from "react";
import { parseAsInteger, parseAsString, useQueryState } from "nuqs";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { DataTablePagination } from "@/components/shared/DataTable";
import { ErrorState } from "@/components/shared/ErrorState";
import { useQuery, useQueryClient, type UseQueryResult } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import useAuthorized from "../../hooks/useAuthorized";
import { fetchClient } from "@/lib/http/api";
import { extractProxyErrorMessage } from "@/lib/http/client";
import { useCustomerBudgetForm } from "../../customers/_components/useCustomerBudgetForm";
import { AdminOnlyNotice } from "@/components/shared/AdminOnlyNotice";
import { appSurchargeField, appSurchargeValue, type App, type AppResource } from "./appForms";
import { appDefinitionKey, appResourcesKey, loadAppDefinition, loadAppResources, loadApps } from "./appApi";
import { applicationRows, grantedApplicationRows } from "./applications";
import { readsEveryApp, useGrantedApps } from "./useGrantedApps";
import ApplicationsTable from "./ApplicationsTable";
import AppBudgetEditor from "./AppBudgetEditor";
import AppDefinitionSetting from "./AppDefinitionSetting";
import AppProfileSelect from "./AppProfileSelect";
import { useOverallAppBudgetGroups } from "./useAppBudgetGroups";
import type { BudgetGroups } from "../../customers/_components/customerBudgetGroupsPlan";
import { uiHref } from "@/utils/uiHref";

const APPLICATIONS_PAGE_SIZE = 25;

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
    <main className="mx-auto w-full max-w-[1600px] px-4 py-6 sm:px-6">
      <ErrorState message={extractProxyErrorMessage(granted.apps.error)} onRetry={() => void granted.apps.refetch()} />
    </main>
  );
}

function ManagedApps() {
  const { t } = useTranslation();
  const definition = useQuery({ queryKey: appDefinitionKey, queryFn: loadAppDefinition });
  const resourceType = definition.data?.resource_type ?? null;
  const resources = useQuery({
    queryKey: appResourcesKey(resourceType),
    queryFn: () => loadAppResources(resourceType!),
    enabled: resourceType !== null,
  });
  const apps = useQuery({
    queryKey: ["apps", resourceType],
    queryFn: () => loadApps(),
    enabled: resourceType !== null,
  });
  const [selected, setSelected] = useQueryState("app", parseAsString);
  const resource = resources.data?.find((row) => row.resource_id === selected);
  if (selected && [definition, resources].some((query) => query.isLoading))
    return <p role="status">{t("apps.loading")}</p>;
  if (selected && resource)
    return (
      <main className="mx-auto w-full max-w-[1200px] space-y-6 px-4 py-6 sm:px-6">
        <header className="space-y-1">
          <h1 className="text-lg font-semibold">{resource.alias}</h1>
          <p className="font-mono text-xs text-muted-foreground">{resource.resource_id}</p>
        </header>
        <ResourceDefaults
          key={`${resource.resource_type}:${resource.resource_id}:${resource.app_id ?? "new"}`}
          resource={resource}
        />
        <Button
          variant="link"
          size="sm"
          className="h-auto p-0 text-xs text-muted-foreground"
          onClick={() => void setSelected(null)}
        >
          {t("apps.list.back")}
        </Button>
      </main>
    );
  return (
    <main className="mx-auto w-full max-w-[1600px] space-y-6 px-4 py-6 sm:px-6">
      <header>
        <h1 className="text-lg font-semibold">{t("apps.title")}</h1>
        <p className="mt-1 text-sm text-muted-foreground">{t("apps.settingsHint")}</p>
      </header>
      {resourceType && (
        <ApplicationsList resources={resources} apps={apps} onOpen={(resourceId) => void setSelected(resourceId)} />
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
    <main className="mx-auto w-full max-w-[1600px] space-y-6 px-4 py-6 sm:px-6">
      <header>
        <h1 className="text-lg font-semibold">{t("apps.title")}</h1>
        <p className="mt-1 text-sm text-muted-foreground">{t("apps.grantedHint")}</p>
      </header>
      <section className="rounded-lg border bg-card" aria-label={t("apps.list.title")}>
        <ApplicationsTable rows={grantedApplicationRows(apps)} loading={false} searching={false} />
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
  const [search, setSearch] = useState("");
  const [page, setPage] = useQueryState("page", parseAsInteger.withDefault(1));
  const [requestedPageSize, setPageSize] = useQueryState(
    "page_size",
    parseAsInteger.withDefault(APPLICATIONS_PAGE_SIZE),
  );
  const pageSize = requestedPageSize > 0 ? requestedPageSize : APPLICATIONS_PAGE_SIZE;
  const rows = useMemo(
    () => applicationRows(resources.data ?? [], apps.data ?? [], search),
    [resources.data, apps.data, search],
  );
  const currentPage = Math.max(0, Math.min(page, Math.ceil(rows.length / pageSize)) - 1);
  return (
    <section className="rounded-lg border bg-card" aria-label={t("apps.list.title")}>
      <div className="flex flex-wrap items-center justify-between gap-3 border-b p-3">
        <Input
          className="max-w-xs"
          aria-label={t("apps.list.searchPlaceholder")}
          placeholder={t("apps.list.searchPlaceholder")}
          value={search}
          onChange={(event) => {
            setSearch(event.target.value);
            void setPage(1);
          }}
        />
        <span className="text-xs text-muted-foreground">{t("apps.list.count", { count: rows.length })}</span>
      </div>
      {resources.isError || apps.isError ? (
        <div className="p-3">
          <ErrorState
            message={extractProxyErrorMessage(resources.error ?? apps.error)}
            onRetry={() => void (resources.isError ? resources.refetch() : apps.refetch())}
          />
        </div>
      ) : (
        <ApplicationsTable
          rows={rows.slice(currentPage * pageSize, (currentPage + 1) * pageSize)}
          loading={resources.isPending || apps.isPending}
          searching={search.trim() !== ""}
          onOpen={onOpen}
        />
      )}
      <DataTablePagination
        page={currentPage}
        pageSize={pageSize}
        rowCount={rows.length}
        onPageChange={(pageIndex) => void setPage(pageIndex + 1)}
        onPageSizeChange={(size) => {
          void setPageSize(size);
          void setPage(1);
        }}
        isLoading={resources.isFetching || apps.isFetching}
      />
    </section>
  );
}

function ResourceDefaults({ resource }: { resource: AppResource }) {
  const { t } = useTranslation();
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
  if (groups.isPending || (resource.app_id && app.isPending)) return <p role="status">{t("apps.loading")}</p>;
  if (groups.isError)
    return <ErrorState message={extractProxyErrorMessage(groups.error)} onRetry={() => void groups.refetch()} />;
  if (app.isError)
    return <ErrorState message={extractProxyErrorMessage(app.error)} onRetry={() => void app.refetch()} />;
  return <AppDefaultsForm resource={resource} app={app.data} profiles={groups.data.default_budgets} />;
}

const appDefaultsFields = (app: App | undefined, profiles: BudgetGroups["default_budgets"]) => {
  const customSource = app?.budget ? "custom" : "inherit";
  return {
    budget: budgetEditorFields(app?.budget),
    budgetSource: profiles.some((profile) => profile.budget_id === app?.budget_id)
      ? `profile:${app?.budget_id}`
      : customSource,
    member: budgetEditorFields(app?.default_customer_member_budget),
    profile: app?.default_tier_budget_id ?? "",
    autoCreate: app?.auto_create_customers ?? true,
    surcharge: appSurchargeField(app?.customer_margin),
  };
};

function AppDefaultsForm({
  resource,
  app,
  profiles,
}: {
  resource: AppResource;
  app?: App;
  profiles: BudgetGroups["default_budgets"];
}) {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const [saved, setSaved] = useState(false);
  const { fields, setFields, update, pending, error, save } = useCustomerBudgetForm({
    initial: appDefaultsFields(app, profiles),
    parse: (fields) => {
      const budget =
        fields.budgetSource === "custom"
          ? permanentBudgetEditorValue(fields.budget, app?.budget)
          : ({ ok: true, value: null } as const);
      const member = permanentBudgetEditorValue(fields.member, app?.default_customer_member_budget);
      if (!budget.ok || !member.ok) return { ok: false, error: t("customerPortal.hierarchy.invalidNumber") };
      const margin = appSurchargeValue(fields.surcharge);
      if (!margin.ok) return { ok: false, error: t("apps.surchargeInvalid") };
      return {
        ok: true,
        value: {
          ...(fields.budgetSource === "custom"
            ? { budget: budget.value }
            : { budget_id: fields.budgetSource.startsWith("profile:") ? fields.budgetSource.slice(8) : null }),
          default_customer_member_budget: member.value,
          default_tier_budget_id: fields.profile || null,
          auto_create_customers: fields.autoCreate,
          customer_margin: margin.value,
        },
      };
    },
    write: async (body) => {
      setSaved(false);
      const response = app
        ? await fetchClient.PATCH("/management/v1/apps/{app_id}", { params: { path: { app_id: app.app_id } }, body })
        : await fetchClient.POST("/management/v1/apps", {
            body: { ...body, resource_id: resource.resource_id, resource_type: resource.resource_type },
          });
      if (response.data) setFields(appDefaultsFields(response.data, profiles));
      await Promise.all(
        ["app-resources", "apps", "app", "app-organization", "app-member"].map((key) =>
          queryClient.invalidateQueries({ queryKey: [key] }),
        ),
      );
      setSaved(true);
    },
  });
  return (
    <form onSubmit={save} className="space-y-6" aria-label={t("apps.defaultsFor", { alias: resource.alias })}>
      <AppBudgetEditor
        modelScope={resource}
        title={t("apps.appBudget")}
        description={t("apps.appBudgetHint")}
        id="app-budget"
        fields={fields.budget}
        onChange={(value) => update("budget", value)}
        disabled={pending}
        source={{
          profiles,
          value: fields.budgetSource,
          onChange: (source) => {
            if (source)
              setFields((current) => ({
                ...current,
                budgetSource: source,
                budget: { ...current.budget, budgetEnabled: source === "custom" },
              }));
          },
        }}
      />
      <a href={`${uiHref("customers/settings")}?tab=overall-app`} className="text-xs text-primary hover:underline">
        {t("apps.manageOverallProfiles")}
      </a>
      <section className="space-y-4 border-t pt-6" aria-labelledby="app-org-default-heading">
        <div className="space-y-1">
          <h3 id="app-org-default-heading" className="font-semibold">
            {t("apps.organizationDefault")}
          </h3>
          <p className="text-sm text-muted-foreground">{t("apps.defaultProfileHint")}</p>
        </div>
        <AppProfileSelect
          value={fields.profile}
          onChange={(value) => update("profile", value)}
          emptyLabel={t("customerPortal.createDialog.noLimit")}
        />
      </section>
      <div className="border-t pt-6">
        <AppBudgetEditor
          modelScope={resource}
          title={t("apps.memberDefault")}
          description={t("apps.memberDefaultHint")}
          id="app-member"
          fields={fields.member}
          onChange={(value) => update("member", value)}
          disabled={pending}
        />
      </div>
      <div className="space-y-1.5 border-t pt-4">
        <label className="flex items-center gap-2 text-sm font-medium">
          <Checkbox
            checked={fields.autoCreate}
            disabled={pending}
            onCheckedChange={(checked) => update("autoCreate", checked === true)}
          />
          {t("apps.autoCreateCustomers")}
        </label>
        <p className="text-xs text-muted-foreground">{t("apps.autoCreateCustomersHint")}</p>
      </div>
      <div className="max-w-md space-y-1.5 border-t pt-4">
        <label htmlFor="app-surcharge" className="block text-sm font-medium">
          {t("apps.surcharge")}
        </label>
        <Input
          id="app-surcharge"
          type="number"
          step="any"
          className="w-full"
          placeholder={t("apps.surchargePlaceholder")}
          disabled={pending}
          value={fields.surcharge}
          onChange={(event) => update("surcharge", event.target.value)}
        />
        <p className="text-xs text-muted-foreground">{t("apps.surchargeHint")}</p>
      </div>
      {error && (
        <p role="alert" className="text-sm text-destructive">
          {error}
        </p>
      )}
      {saved && (
        <p role="status" className="text-sm">
          {t("apps.saved")}
        </p>
      )}
      <Button type="submit" disabled={pending}>
        {t("apps.save")}
      </Button>
    </form>
  );
}
