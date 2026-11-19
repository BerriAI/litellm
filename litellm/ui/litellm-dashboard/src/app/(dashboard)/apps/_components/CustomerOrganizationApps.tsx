"use client";

import { useState, type FormEvent } from "react";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { SearchSelect } from "@/components/shared/SearchSelect";
import { ErrorState } from "@/components/shared/ErrorState";
import { useQueries, useQuery, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { $api, fetchClient } from "@/lib/http/api";
import { ApiError, extractProxyErrorMessage } from "@/lib/http/client";
import { uiHref } from "@/utils/uiHref";
import { appDefinitionKey, appResourcesKey, loadAppDefinition, loadAppResources, loadApps } from "./appApi";
import { appBudgetFields, appBudgetValue, type App, type AppResource, type AppOrganization } from "./appForms";
import AppBudgetEditor from "./AppBudgetEditor";
import AppProfileSelect from "./AppProfileSelect";
import AppBudgetLimits from "./AppBudgetLimits";
import { appBudgetWindows } from "./appBudgetPresentation";
import { customerBudgetPeriod } from "../../customers/_components/customerBudgetUsage";

const organizationAppKey = (appId: string | null, organizationId: string) =>
  ["app-organization", appId, organizationId] as const;
const loadOrganizationApp = async (appId: string, organizationId: string) => {
  try {
    const response = await fetchClient.GET("/management/v1/apps/{app_id}/customer_organizations/{customer_org_id}", {
      params: { path: { app_id: appId, customer_org_id: organizationId } },
    });
    return response.data ?? null;
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) return null;
    throw error;
  }
};

export default function CustomerOrganizationApps({
  organizationId,
  organizationAlias,
  onAppSelect,
}: {
  organizationId: string;
  organizationAlias?: string | null;
  onAppSelect?: (appId: string) => void;
}) {
  const { t } = useTranslation();
  const [adding, setAdding] = useState(false);
  const [editing, setEditing] = useState<AppResource | null>(null);
  const [pending, setPending] = useState(false);
  const { definition, resourceType, groups, resources, rows, loading, error, noRows, ready, setupNeeded } =
    useOrganizationApps(organizationId);
  const closeDialog = () => {
    setAdding(false);
    setEditing(null);
  };
  return (
    <section id="app-budgets" className="space-y-4 border-t pt-6" aria-label={t("apps.organizationTitle")}>
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-base font-semibold">{t("apps.organizationTitle")}</h2>
          <p className="mt-1 text-sm text-muted-foreground">{t("apps.organizationHint")}</p>
        </div>
        <AddAppBudgetAction
          setupNeeded={setupNeeded}
          disabled={definition.isPending || definition.isError}
          onAdd={() => {
            setAdding(true);
            setEditing(null);
          }}
        />
      </div>
      {loading && (
        <p role="status" className="text-sm">
          {t("apps.loading")}
        </p>
      )}
      {error && <ErrorState message={extractProxyErrorMessage(error)} />}
      {setupNeeded && <p className="text-sm text-muted-foreground">{t("apps.defineFirst")}</p>}
      {noRows && ready && <p className="text-sm text-muted-foreground">{t("apps.noOrganizationBudgets")}</p>}
      {rows.length > 0 && (
        <OrganizationAppsTable
          rows={rows}
          groups={groups.data?.default_budgets ?? []}
          onAppSelect={onAppSelect}
          onEdit={(resource) => {
            setAdding(false);
            setEditing(resource);
          }}
        />
      )}
      <Dialog
        open={adding || editing !== null}
        onOpenChange={(open) => {
          if (!open && !pending) closeDialog();
        }}
      >
        <DialogContent className="max-h-[85vh] overflow-y-auto sm:max-w-2xl" showCloseButton={!pending}>
          <DialogHeader>
            <DialogTitle>{adding ? t("apps.addBudget") : t("apps.budgetFor", { alias: editing?.alias })}</DialogTitle>
            <DialogDescription>{t("apps.organizationHint")}</DialogDescription>
          </DialogHeader>
          {adding && resourceType && (
            <OrganizationAppPicker resources={resources} selected={editing} pending={pending} onChange={setEditing} />
          )}
          {editing && (
            <OrganizationApp
              key={`${editing.resource_type}:${editing.resource_id}`}
              resource={editing}
              organizationId={organizationId}
              organizationAlias={organizationAlias}
              pending={pending}
              onPendingChange={setPending}
              onClose={closeDialog}
            />
          )}
        </DialogContent>
      </Dialog>
    </section>
  );
}

function OrganizationAppPicker({
  resources,
  selected,
  pending,
  onChange,
}: {
  resources: ReturnType<typeof useOrganizationApps>["resources"];
  selected: AppResource | null;
  pending: boolean;
  onChange: (resource: AppResource | null) => void;
}) {
  const { t } = useTranslation();
  return (
    <div className="space-y-2">
      <label htmlFor="organization-app" className="block text-sm font-medium">
        {t("apps.selectApp")}
      </label>
      <SearchSelect
        id="organization-app"
        className="w-full"
        placeholder={t("apps.selectApp")}
        value={selected?.resource_id}
        onValueChange={(value) => onChange(resources.data?.find((resource) => resource.resource_id === value) ?? null)}
        disabled={!resources.data || pending}
        options={resources.data?.map((resource) => ({ value: resource.resource_id, label: resource.alias })) ?? []}
      />
      {resources.data?.length === 0 && <p className="text-sm text-muted-foreground">{t("apps.noResources")}</p>}
      {resources.isError && (
        <p role="alert" className="text-sm text-destructive">
          {extractProxyErrorMessage(resources.error)}
        </p>
      )}
    </div>
  );
}

function AddAppBudgetAction({
  setupNeeded,
  disabled,
  onAdd,
}: {
  setupNeeded: boolean;
  disabled: boolean;
  onAdd: () => void;
}) {
  const { t } = useTranslation();
  if (setupNeeded)
    return (
      <Button nativeButton={false} role="link" render={<a href={uiHref("apps")} />}>
        {t("apps.addBudget")}
      </Button>
    );
  return (
    <Button type="button" disabled={disabled} onClick={onAdd}>
      {t("apps.addBudget")}
    </Button>
  );
}

function useOrganizationApps(organizationId: string) {
  const definition = useQuery({ queryKey: appDefinitionKey, queryFn: loadAppDefinition });
  const resourceType = definition.data?.resource_type ?? null;
  const apps = useQuery({ queryKey: ["apps"], queryFn: () => loadApps(), enabled: resourceType !== null });
  const associations = useQueries({
    queries: (apps.data ?? []).map((app) => ({
      queryKey: organizationAppKey(app.app_id, organizationId),
      queryFn: () => loadOrganizationApp(app.app_id, organizationId),
      retry: false,
    })),
  });
  const groups = $api.useQuery(
    "get",
    "/management/v1/customer_organization_budget_groups",
    {},
    { enabled: resourceType !== null },
  );
  const resources = useQuery({
    queryKey: appResourcesKey(resourceType),
    queryFn: () => loadAppResources(resourceType!),
    enabled: resourceType !== null,
  });
  const rows = (apps.data ?? []).flatMap((app, index) => {
    const association = associations[index];
    return association?.data ? [{ app, row: association.data }] : [];
  });
  const loadingAssociations = apps.isPending || associations.some((query) => query.isPending);
  const loading = definition.isPending || (resourceType !== null && loadingAssociations);
  const failedAssociation = associations.find((query) => query.isError);
  const error = definition.error ?? apps.error ?? failedAssociation?.error;
  const noRows = resourceType !== null && rows.length === 0;
  const ready = !loading && !error;
  const setupNeeded = Boolean(definition.data) && resourceType === null;
  return { definition, resourceType, groups, resources, rows, loading, error, noRows, ready, setupNeeded };
}

function OrganizationAppsTable({
  rows,
  groups,
  onEdit,
  onAppSelect,
}: {
  rows: { app: App; row: AppOrganization }[];
  groups: { budget_id: string; name: string }[];
  onEdit: (resource: AppResource) => void;
  onAppSelect?: (appId: string) => void;
}) {
  const { t } = useTranslation();
  return (
    <div className="overflow-x-auto">
      <Table className="w-full text-sm" aria-label={t("apps.organizationTitle")}>
        <TableHeader>
          <TableRow className="border-b text-left text-xs text-muted-foreground">
            <TableHead className="py-3 font-medium">{t("apps.alias")}</TableHead>
            <TableHead className="py-3 font-medium">{t("apps.effectiveBudget")}</TableHead>
            <TableHead className="py-3 font-medium">{t("apps.actions")}</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map(({ app, row }) => {
            const profileId = row.budget_id ?? row.tier_budget_id ?? app.default_tier_budget_id;
            const profileName = groups.find((group) => group.budget_id === profileId)?.name;
            let sourceLabel = t("apps.appDefault");
            if (row.budget || row.budget_override_id) sourceLabel = t("apps.customBudget");
            else if (row.budget_id || row.tier_budget_id) sourceLabel = profileName ?? t("apps.profile");
            const { primary } = appBudgetWindows(row.effective_budget, row.budget_usage);
            const period = customerBudgetPeriod(primary?.budget_duration ?? null);
            const periodLabel = period === "other" ? primary?.budget_duration : t(`apps.${period}`);
            if (!row.budget_override_id && !row.budget_id && !row.tier_budget_id && profileName)
              sourceLabel = `${profileName} · ${sourceLabel}`;
            const resource: AppResource = {
              resource_type: app.resource_type,
              resource_id: app.resource_id,
              app_id: app.app_id,
              alias: app.alias,
            };
            return (
              <TableRow key={app.app_id} className="border-b">
                <TableCell className="py-3">
                  <span className="block font-medium">{app.alias}</span>
                  <span className="mt-1 block text-xs text-muted-foreground">
                    {[sourceLabel, periodLabel].filter(Boolean).join(" · ")}
                  </span>
                </TableCell>
                <TableCell className="py-3">
                  <AppBudgetLimits budget={row.effective_budget} usage={row.budget_usage} cumulativeSpend={row.spend} />
                </TableCell>
                <TableCell className="py-3 text-right">
                  {onAppSelect && (
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      onClick={() => onAppSelect(app.app_id)}
                      aria-label={t("apps.viewUsageFor", { alias: app.alias })}
                    >
                      {t("apps.viewUsage")}
                    </Button>
                  )}
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={() => onEdit(resource)}
                    aria-label={t("apps.editBudgetFor", { alias: app.alias })}
                  >
                    {t("apps.edit")}
                  </Button>
                </TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </div>
  );
}

function OrganizationApp({
  resource,
  organizationId,
  organizationAlias,
  pending,
  onPendingChange,
  onClose,
}: {
  resource: AppResource;
  organizationId: string;
  organizationAlias?: string | null;
  pending: boolean;
  onPendingChange: (pending: boolean) => void;
  onClose: () => void;
}) {
  const { t } = useTranslation();
  const associationOptions = {
    queryKey: organizationAppKey(resource.app_id ?? null, organizationId),
    queryFn: () => loadOrganizationApp(resource.app_id!, organizationId),
    enabled: Boolean(resource.app_id),
    retry: false,
  };
  const association = useQuery(associationOptions);
  if (resource.app_id && association.isPending) return <p role="status">{t("apps.loading")}</p>;
  if (association.isError)
    return (
      <ErrorState message={extractProxyErrorMessage(association.error)} onRetry={() => void association.refetch()} />
    );
  return (
    <OrganizationAppForm
      resource={resource}
      organizationId={organizationId}
      organizationAlias={organizationAlias}
      row={association.data ?? null}
      pending={pending}
      onPendingChange={onPendingChange}
      onClose={onClose}
    />
  );
}

function organizationBudgetSource(row: AppOrganization | null): "inherit" | "profile" | "custom" {
  if (row?.budget || (row?.budget_override != null && row.budget_override_expires_at == null)) return "custom";
  if (row?.budget_id || row?.tier_budget_id) return "profile";
  return "inherit";
}

function OrganizationAppForm({
  resource,
  organizationId,
  organizationAlias,
  row,
  pending,
  onPendingChange: setPending,
  onClose,
}: {
  resource: AppResource;
  organizationId: string;
  organizationAlias?: string | null;
  row: AppOrganization | null;
  pending: boolean;
  onPendingChange: (pending: boolean) => void;
  onClose: () => void;
}) {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const [source, setSource] = useState(() => organizationBudgetSource(row));
  const originalBudget = row?.budget ?? (row?.budget_override_expires_at == null ? row?.budget_override : null);
  const [profile, setProfile] = useState(row?.budget_id ?? row?.tier_budget_id ?? "");
  const [override, setOverride] = useState(() => appBudgetFields(originalBudget));
  const [member, setMember] = useState(() =>
    appBudgetFields(row?.customer_member_budget_id ? row.effective_customer_member_budget : null),
  );
  const [error, setError] = useState<string | null>(null);
  const changeSource = (value: string | null) => {
    if (value !== "inherit" && value !== "profile" && value !== "custom") return;
    setSource(value);
    if (value === "custom") setOverride((current) => ({ ...current, budgetEnabled: true }));
  };
  const save = async (event: FormEvent) => {
    event.preventDefault();
    const budget = appBudgetValue({ ...override, budgetEnabled: source === "custom" }, originalBudget);
    const memberBudget = appBudgetValue(
      member,
      row?.customer_member_budget_id ? row.effective_customer_member_budget : null,
    );
    const missingProfile = source === "profile" && !profile;
    if (!budget.ok || !memberBudget.ok || missingProfile) {
      setError(t(source === "profile" && !profile ? "apps.chooseProfile" : "customerPortal.hierarchy.invalidNumber"));
      return;
    }
    setPending(true);
    setError(null);
    try {
      let appId = resource.app_id;
      if (!appId) {
        const created = await fetchClient.POST("/management/v1/apps", {
          body: { resource_id: resource.resource_id, resource_type: resource.resource_type },
        });
        if (!created.data) throw new Error(t("apps.saveFailed"));
        appId = created.data.app_id;
      }
      await fetchClient.PUT("/management/v1/apps/{app_id}/customer_organizations/{customer_org_id}", {
        params: { path: { app_id: appId, customer_org_id: organizationId } },
        body: {
          ...(source === "custom" ? { budget: budget.value } : { budget_id: source === "profile" ? profile : null }),
          ...(row?.budget_override != null && row.budget_override_expires_at == null ? { budget_override: null } : {}),
          customer_member_budget: memberBudget.value,
        },
      });
      await queryClient.invalidateQueries({ queryKey: ["apps"] });
      await queryClient.invalidateQueries({ queryKey: ["app-resources"] });
      await queryClient.invalidateQueries({ queryKey: ["app-organization"] });
      await queryClient.invalidateQueries({ queryKey: ["app-member"] });
      onClose();
    } catch (failure) {
      setError(extractProxyErrorMessage(failure));
    } finally {
      setPending(false);
    }
  };
  return (
    <form onSubmit={save} className="space-y-5" aria-label={t("apps.organizationSettings")}>
      <h3 className="font-semibold">{t("apps.budgetFor", { alias: resource.alias })}</h3>
      <div className="grid gap-4 sm:grid-cols-2">
        <div className="space-y-1.5">
          <label htmlFor="app-budget-customer" className="block text-sm">
            {t("apps.customer")}
          </label>
          <Input id="app-budget-customer" value={organizationAlias || organizationId} readOnly />
        </div>
        <div className="space-y-1.5">
          <label htmlFor="app-budget-resource" className="block text-sm">
            {t("apps.alias")}
          </label>
          <Input id="app-budget-resource" value={resource.alias} readOnly />
        </div>
      </div>
      <div className="max-w-md space-y-1.5">
        <label htmlFor="app-budget-source" className="block text-sm">
          {t("apps.source")}
        </label>
        <SearchSelect
          id="app-budget-source"
          className="w-full"
          value={source}
          allowClear={false}
          onValueChange={changeSource}
          options={[
            { value: "inherit", label: t("apps.appDefault") },
            { value: "profile", label: t("apps.profile") },
            { value: "custom", label: t("apps.customBudget") },
          ]}
        />
      </div>
      {source === "profile" && (
        <div className="max-w-md">
          <AppProfileSelect
            id="organization-app-profile"
            value={profile}
            onChange={setProfile}
            emptyLabel={t("apps.chooseProfile")}
          />
        </div>
      )}
      {source === "custom" && (
        <AppBudgetEditor
          title={t("apps.customBudget")}
          optional={false}
          id="org-app-override"
          fields={{ ...override, budgetEnabled: true }}
          onChange={setOverride}
        />
      )}
      <AppBudgetEditor
        title={t("apps.organizationMemberDefault")}
        id="org-app-member"
        fields={member}
        onChange={setMember}
      />
      <p className="text-xs text-muted-foreground">{t("apps.memberDefaultHint")}</p>
      <p className="text-sm text-muted-foreground">
        {t("apps.onlyThisPair", { customer: organizationAlias || organizationId, alias: resource.alias })}
      </p>
      {error && (
        <p role="alert" className="text-sm text-destructive">
          {error}
        </p>
      )}
      <div className="flex gap-2">
        <Button type="submit" disabled={pending}>
          {t("apps.saveBudget")}
        </Button>
        <Button type="button" variant="outline" onClick={onClose} disabled={pending}>
          {t("apps.cancel")}
        </Button>
      </div>
    </form>
  );
}
