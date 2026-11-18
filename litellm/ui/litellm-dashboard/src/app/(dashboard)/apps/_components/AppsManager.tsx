"use client";

import { useState, type FormEvent } from "react";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { SearchSelect } from "@/components/shared/SearchSelect";
import { ErrorState } from "@/components/shared/ErrorState";
import { useQuery, useQueryClient } from "@tanstack/react-query";
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
import { appDefinitionKey, appResourcesKey, loadAppDefinition, loadAppResources } from "./appApi";
import AppBudgetEditor from "./AppBudgetEditor";
import AppDefinitionSetting from "./AppDefinitionSetting";
import AppProfileSelect from "./AppProfileSelect";

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
  const [selected, setSelected] = useState<string | null>(null);
  const resource = resources.data?.find((row) => row.resource_id === selected);
  if (isLoading) return <p role="status">{t("apps.loading")}</p>;
  if (!canManage) return <AdminOnlyNotice pageTitle={t("apps.settingsTitle")} />;
  return (
    <main className="mx-auto w-full max-w-[1200px] space-y-8 p-6">
      <header>
        <h1 className="text-2xl font-semibold">{t("apps.settingsTitle")}</h1>
        <p className="mt-2 text-sm text-muted-foreground">{t("apps.settingsHint")}</p>
      </header>
      <AppDefinitionSetting />
      {resourceType && (
        <section className="space-y-5 border-t pt-6" aria-label={t("apps.defaultsTitle")}>
          <div>
            <h2 className="text-base font-semibold">{t("apps.defaultsTitle")}</h2>
            <p className="mt-1 text-sm text-muted-foreground">{t("apps.defaultsHint")}</p>
          </div>
          {resources.isError && (
            <ErrorState message={extractProxyErrorMessage(resources.error)} onRetry={() => void resources.refetch()} />
          )}
          <div className="max-w-md space-y-2">
            <label htmlFor="app-defaults-resource" className="block text-sm">
              {t("apps.selectApp")}
            </label>
            <SearchSelect
              id="app-defaults-resource"
              className="w-full"
              placeholder={t("apps.selectApp")}
              value={selected}
              onValueChange={setSelected}
              disabled={!resources.data}
              options={resources.data?.map((row) => ({ value: row.resource_id, label: row.alias })) ?? []}
            />
          </div>
          {resources.data?.length === 0 && <p className="text-sm text-muted-foreground">{t("apps.noResources")}</p>}
          {resource && (
            <ResourceDefaults
              key={`${resource.resource_type}:${resource.resource_id}:${resource.app_id ?? "new"}`}
              resource={resource}
            />
          )}
        </section>
      )}
    </main>
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
