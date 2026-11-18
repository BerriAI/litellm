"use client";

import { useMemo, useState, type FormEvent } from "react";
import { parseAsInteger, parseAsString, useQueryState } from "nuqs";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { DataTablePagination } from "@/components/shared/DataTable";
import { ErrorState } from "@/components/shared/ErrorState";
import { useQuery, useQueryClient, type UseQueryResult } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import useAuthorized from "../../hooks/useAuthorized";
import { isProxyAdminRole } from "@/utils/roles";
import { fetchClient } from "@/lib/http/api";
import { extractProxyErrorMessage } from "@/lib/http/client";
import { AdminOnlyNotice } from "@/components/shared/AdminOnlyNotice";
import {
  appBudgetFields,
  appBudgetValue,
  appSurchargeField,
  appSurchargeValue,
  type App,
  type AppResource,
} from "./appForms";
import { appDefinitionKey, appResourcesKey, loadAppDefinition, loadAppResources, loadApps } from "./appApi";
import { applicationRows } from "./applications";
import ApplicationsTable from "./ApplicationsTable";
import AppBudgetEditor from "./AppBudgetEditor";
import AppDefinitionSetting from "./AppDefinitionSetting";
import AppProfileSelect from "./AppProfileSelect";

const APPLICATIONS_PAGE_SIZE = 25;

export default function AppsManager() {
  const { t } = useTranslation();
  const { accessToken, userRole, isLoading } = useAuthorized();
  const canManage = Boolean(accessToken && isProxyAdminRole(userRole));
  const definition = useQuery({ queryKey: appDefinitionKey, queryFn: loadAppDefinition, enabled: canManage });
  const resourceType = definition.data?.resource_type ?? null;
  const resources = useQuery({
    queryKey: appResourcesKey(resourceType),
    queryFn: () => loadAppResources(resourceType!),
    enabled: canManage && resourceType !== null,
  });
  const apps = useQuery({
    queryKey: ["apps", resourceType],
    queryFn: () => loadApps(),
    enabled: canManage && resourceType !== null,
  });
  const [selected, setSelected] = useQueryState("app", parseAsString);
  const resource = resources.data?.find((row) => row.resource_id === selected);
  if (isLoading) return <p role="status">{t("apps.loading")}</p>;
  if (!canManage) return <AdminOnlyNotice pageTitle={t("apps.title")} />;
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
  if (resource.app_id && app.isPending) return <p role="status">{t("apps.loading")}</p>;
  if (app.isError)
    return <ErrorState message={extractProxyErrorMessage(app.error)} onRetry={() => void app.refetch()} />;
  return <AppDefaultsForm resource={resource} app={app.data} />;
}

function AppDefaultsForm({ resource, app }: { resource: AppResource; app?: App }) {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const [budget, setBudget] = useState(() => appBudgetFields(app?.budget));
  const [member, setMember] = useState(() => appBudgetFields(app?.default_customer_member_budget));
  const [profile, setProfile] = useState(app?.default_tier_budget_id ?? "");
  const [autoCreate, setAutoCreate] = useState(app?.auto_create_customers ?? true);
  const [surcharge, setSurcharge] = useState(() => appSurchargeField(app?.customer_margin));
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);
  const save = async (event: FormEvent) => {
    event.preventDefault();
    const appBudget = appBudgetValue(budget, app?.budget);
    const memberBudget = appBudgetValue(member, app?.default_customer_member_budget);
    if (!appBudget.ok || !memberBudget.ok) {
      setError(t("customerPortal.hierarchy.invalidNumber"));
      return;
    }
    const customerMargin = appSurchargeValue(surcharge);
    if (!customerMargin.ok) {
      setError(t("apps.surchargeInvalid"));
      return;
    }
    setPending(true);
    setError(null);
    setSaved(false);
    try {
      const body = {
        budget: appBudget.value,
        default_customer_member_budget: memberBudget.value,
        default_tier_budget_id: profile || null,
        auto_create_customers: autoCreate,
        customer_margin: customerMargin.value,
      };
      if (app)
        await fetchClient.PATCH("/management/v1/apps/{app_id}", { params: { path: { app_id: app.app_id } }, body });
      else
        await fetchClient.POST("/management/v1/apps", {
          body: { ...body, resource_id: resource.resource_id, resource_type: resource.resource_type },
        });
      await queryClient.invalidateQueries({ queryKey: ["app-resources"] });
      await queryClient.invalidateQueries({ queryKey: ["apps"] });
      await queryClient.invalidateQueries({ queryKey: ["app"] });
      await queryClient.invalidateQueries({ queryKey: ["app-organization"] });
      await queryClient.invalidateQueries({ queryKey: ["app-member"] });
      setSaved(true);
    } catch (failure) {
      setError(extractProxyErrorMessage(failure));
    } finally {
      setPending(false);
    }
  };
  return (
    <form onSubmit={save} className="space-y-5" aria-label={t("apps.defaultsFor", { alias: resource.alias })}>
      <h3 className="text-sm font-semibold">{t("apps.defaultsFor", { alias: resource.alias })}</h3>
      <div className="max-w-md">
        <AppProfileSelect
          id="app-default-profile"
          value={profile}
          onChange={setProfile}
          emptyLabel={t("apps.noDefaultProfile")}
        />
      </div>
      <p className="text-xs text-muted-foreground">{t("apps.defaultProfileHint")}</p>
      <div className="grid gap-6 lg:grid-cols-2">
        <AppBudgetEditor title={t("apps.appBudget")} id="app-budget" fields={budget} onChange={setBudget} />
        <AppBudgetEditor title={t("apps.memberDefault")} id="app-member" fields={member} onChange={setMember} />
      </div>
      <div className="space-y-1.5 border-t pt-4">
        <label className="flex items-center gap-2 text-sm font-medium">
          <Checkbox
            checked={autoCreate}
            disabled={pending}
            onCheckedChange={(checked) => setAutoCreate(checked === true)}
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
          value={surcharge}
          onChange={(event) => setSurcharge(event.target.value)}
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
