"use client";

import { useState, type FormEvent } from "react";
import { useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { fetchClient } from "@/lib/http/api";
import { ApiError, extractProxyErrorMessage } from "@/lib/http/client";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { loadApps, appSpend } from "./appApi";
import { appBudgetFields, appBudgetValue, type App, type AppOrganization } from "./appForms";
import AppBudgetEditor from "./AppBudgetEditor";

export default function CustomerOrganizationApps({ organizationId }: { organizationId: string }) {
  const { t } = useTranslation();
  const apps = useQuery({ queryKey: ["apps"], queryFn: () => loadApps() });
  const [appId, setAppId] = useState("");
  const app = apps.data?.find((row) => row.app_id === appId);
  return (
    <section className="space-y-4 rounded-lg border p-5" aria-label={t("apps.title")}>
      <h2 className="text-lg font-semibold">{t("apps.title")}</h2>
      <p className="text-sm text-muted-foreground">{t("apps.organizationHint")}</p>
      {apps.isPending && <p role="status">{t("apps.loading")}</p>}
      {apps.isError && <p role="alert">{extractProxyErrorMessage(apps.error)}</p>}
      {apps.data?.length === 0 && <p>{t("apps.empty")}</p>}
      <div className="max-w-sm space-y-2">
        <Label htmlFor="organization-app">{t("apps.selectApp")}</Label>
        <Select
          value={appId || "__none__"}
          onValueChange={(value) => setAppId(value === "__none__" ? "" : value ?? "")}
        >
          <SelectTrigger id="organization-app">
            <SelectValue>{app?.alias ?? t("apps.selectApp")}</SelectValue>
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="__none__">{t("apps.none")}</SelectItem>
            {apps.data?.map((row) => (
              <SelectItem key={row.app_id} value={row.app_id}>
                {row.alias}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      {app && <OrganizationApp key={app.app_id} app={app} organizationId={organizationId} />}
    </section>
  );
}

function OrganizationApp({ app, organizationId }: { app: App; organizationId: string }) {
  const { t } = useTranslation();
  const association = useQuery({
    queryKey: ["app-organization", app.app_id, organizationId],
    queryFn: async () => {
      try {
        const response = await fetchClient.GET(
          "/management/v1/apps/{app_id}/customer_organizations/{customer_org_id}",
          {
            params: { path: { app_id: app.app_id, customer_org_id: organizationId } },
          },
        );
        return response.data ?? null;
      } catch (error) {
        if (error instanceof ApiError && error.status === 404) return null;
        throw error;
      }
    },
    retry: false,
  });
  if (association.isPending) return <p role="status">{t("apps.loading")}</p>;
  if (association.isError)
    return (
      <div role="alert">
        {extractProxyErrorMessage(association.error)}
        <Button variant="outline" onClick={() => void association.refetch()}>
          {t("apps.retry")}
        </Button>
      </div>
    );
  return (
    <OrganizationAppForm
      app={app}
      organizationId={organizationId}
      row={association.data}
      onSaved={async () => {
        await association.refetch();
      }}
    />
  );
}

function OrganizationAppForm({
  app,
  organizationId,
  row,
  onSaved,
}: {
  app: App;
  organizationId: string;
  row: AppOrganization | null;
  onSaved: () => Promise<void>;
}) {
  const { t } = useTranslation();
  const [tier, setTier] = useState(row?.tier_budget_id ?? "");
  const [override, setOverride] = useState(() =>
    appBudgetFields(row?.budget_override_id ? row.effective_budget : null),
  );
  const [member, setMember] = useState(() =>
    appBudgetFields(row?.customer_member_budget_id ? row.effective_customer_member_budget : null),
  );
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const save = async (event: FormEvent) => {
    event.preventDefault();
    const budget = appBudgetValue(override, row?.budget_override_id ? row.effective_budget : null);
    const memberBudget = appBudgetValue(
      member,
      row?.customer_member_budget_id ? row.effective_customer_member_budget : null,
    );
    if (!budget.ok || !memberBudget.ok) {
      setError(t("customerPortal.hierarchy.invalidNumber"));
      return;
    }
    setPending(true);
    setError(null);
    try {
      await fetchClient.PUT("/management/v1/apps/{app_id}/customer_organizations/{customer_org_id}", {
        params: { path: { app_id: app.app_id, customer_org_id: organizationId } },
        body: {
          tier_budget_id: tier || null,
          budget_override: budget.value,
          customer_member_budget: memberBudget.value,
        },
      });
      await onSaved();
    } catch (failure) {
      setError(extractProxyErrorMessage(failure));
    } finally {
      setPending(false);
    }
  };
  return (
    <form onSubmit={save} className="space-y-5" aria-label={t("apps.organizationSettings")}>
      <p className="text-sm">{t("apps.organizationSpend", { spend: appSpend(row?.spend ?? 0) })}</p>
      <div className="max-w-sm space-y-2">
        <Label htmlFor="organization-app-tier">{t("apps.tier")}</Label>
        <Select
          value={tier || "__default__"}
          onValueChange={(value) => setTier(value === "__default__" ? "" : value ?? "")}
        >
          <SelectTrigger id="organization-app-tier">
            <SelectValue>
              {app.tiers?.find((item) => item.budget_id === tier)?.name ?? t("apps.inheritTier")}
            </SelectValue>
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="__default__">{t("apps.inheritTier")}</SelectItem>
            {app.tiers?.map((item) => (
              <SelectItem key={item.budget_id} value={item.budget_id}>
                {item.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      <div className="grid gap-5 lg:grid-cols-2">
        <AppBudgetEditor title={t("apps.override")} id="org-app-override" fields={override} onChange={setOverride} />
        <AppBudgetEditor
          title={t("apps.organizationMemberDefault")}
          id="org-app-member"
          fields={member}
          onChange={setMember}
        />
      </div>
      {error && (
        <p role="alert" className="text-destructive">
          {error}
        </p>
      )}
      <Button type="submit" disabled={pending}>
        {t("apps.save")}
      </Button>
    </form>
  );
}
