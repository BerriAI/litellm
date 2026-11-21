"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { CustomerBudgetDialog } from "../../customers/_components/CustomerBudgetDialog";
import { SearchSelect } from "@/components/shared/SearchSelect";
import { ErrorState } from "@/components/shared/ErrorState";
import { useQueries, useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { extractProxyErrorMessage } from "@/lib/http/client";
import { cn } from "@/lib/cva.config";
import { uiHref } from "@/utils/uiHref";
import {
  appDefinitionKey,
  appResourcesKey,
  loadAppDefinition,
  loadAppResources,
  loadApps,
  organizationAppKey,
  loadOrganizationApp,
} from "./appApi";
import { appOrganizationPresetId, type App, type AppResource, type AppOrganization } from "./appForms";
import {
  AppOrganizationBudgetDialog,
  OrganizationApp,
  appOrganizationSourceLabel,
} from "./AppOrganizationBudgetDialog";
import { useAppBudgetGroups } from "./useAppBudgetGroups";
import { AppBudgetRow, AppBudgetTable, type AppUsageContext } from "./AppBudgetCells";
import { calendarDate, customerPresetRange } from "../../customers/_components/customerUsageRange";

export default function CustomerOrganizationApps({
  organizationId,
  organizationAlias,
  onAppSelect,
  usageContext,
}: {
  organizationId: string;
  organizationAlias?: string | null;
  onAppSelect?: (appId: string) => void;
  usageContext?: AppUsageContext;
}) {
  const { t } = useTranslation();
  const [adding, setAdding] = useState(false);
  const [editing, setEditing] = useState<App | null>(null);
  const [selected, setSelected] = useState<AppResource | null>(null);
  const [pending, setPending] = useState(false);
  const { definition, resourceType, groups, resources, rows, loading, error, noRows, ready, setupNeeded } =
    useOrganizationApps(organizationId);
  const month = customerPresetRange("month");
  const closeDialog = () => {
    setAdding(false);
    setEditing(null);
    setSelected(null);
  };
  const openAddDialog = () => {
    setAdding(true);
    setEditing(null);
    setSelected(null);
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
          onAdd={openAddDialog}
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
          groups={groups.displayBudgets}
          onAppSelect={onAppSelect}
          usageLayout={Boolean(usageContext)}
          usageContext={
            usageContext ?? {
              scope: { org: organizationId, team: null, member: null },
              startDate: calendarDate(month.from),
              endDate: calendarDate(month.to),
              organizationSpend: null,
            }
          }
          onEdit={(resource) => {
            setAdding(false);
            setEditing(resource);
          }}
        />
      )}
      <CustomerBudgetDialog
        open={adding}
        onClose={closeDialog}
        pending={pending}
        title={t("apps.customerAppBudget", { customer: organizationAlias || organizationId })}
        description={t("apps.customerAppBudgetDescription")}
        className="sm:max-w-2xl"
      >
        <div className={cn("shrink-0 px-6 pt-6", !selected && "pb-6")}>
          {resourceType && (
            <OrganizationAppPicker resources={resources} selected={selected} pending={pending} onChange={setSelected} />
          )}
        </div>
        {selected && (
          <OrganizationApp
            key={`${selected.resource_type}:${selected.resource_id}`}
            resource={selected}
            organizationId={organizationId}
            organizationAlias={organizationAlias}
            pending={pending}
            onPendingChange={setPending}
            onClose={closeDialog}
          />
        )}
      </CustomerBudgetDialog>
      {editing && (
        <AppOrganizationBudgetDialog
          app={editing}
          organizationId={organizationId}
          organizationAlias={organizationAlias || organizationId}
          onClose={closeDialog}
        />
      )}
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
      <label htmlFor="organization-app" className="block text-sm text-muted-foreground">
        {t("apps.selectApp")}
      </label>
      <SearchSelect
        id="organization-app"
        className="h-9 w-full"
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
  const groups = useAppBudgetGroups(
    (apps.data ?? []).map((app, index) => appOrganizationPresetId(associations[index]?.data, app)),
    resourceType !== null,
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
  usageContext,
  usageLayout,
}: {
  rows: { app: App; row: AppOrganization }[];
  groups: { budget_id: string; name: string }[];
  onEdit: (app: App) => void;
  onAppSelect?: (appId: string) => void;
  usageContext: AppUsageContext;
  usageLayout: boolean;
}) {
  const { t } = useTranslation();
  return (
    <AppBudgetTable scope="organization" usageLayout={usageLayout}>
      {rows.map(({ app, row }) => (
        <AppBudgetRow
          key={app.app_id}
          scope="organization"
          app={app}
          row={row}
          sourceLabel={appOrganizationSourceLabel(app, row, groups, t)}
          metrics={{ layout: usageLayout ? "full" : "compact", context: usageContext }}
          onAppSelect={onAppSelect}
          onEdit={() => onEdit(app)}
        />
      ))}
    </AppBudgetTable>
  );
}
