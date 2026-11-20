"use client";

import { compactBudgetEditorFields, permanentBudgetEditorValue } from "../../customers/_components/customerBudgetForm";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { CustomerBudgetDialog, CustomerBudgetForm } from "../../customers/_components/CustomerBudgetDialog";
import { useCustomerBudgetForm } from "../../customers/_components/useCustomerBudgetForm";
import { Switch } from "@/components/ui/switch";
import { Input } from "@/components/ui/input";
import { SearchSelect } from "@/components/shared/SearchSelect";
import { ErrorState } from "@/components/shared/ErrorState";
import { useQueries, useQuery, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { fetchClient } from "@/lib/http/api";
import { extractProxyErrorMessage } from "@/lib/http/client";
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
import { appBaseBudget, appOrganizationPresetId, type App, type AppResource, type AppOrganization } from "./appForms";
import CustomerBudgetCompactFields from "../../customers/_components/CustomerBudgetCompactFields";
import { CustomerBudgetEditor } from "../../customers/_components/CustomerLimitsFields";
import {
  budgetChanges,
  customerBudgetPatch,
  hasLegacyCustomerBudget,
} from "../../customers/_components/customerBudgetForm";
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
  const [editing, setEditing] = useState<AppResource | null>(null);
  const [pending, setPending] = useState(false);
  const { definition, resourceType, groups, resources, rows, loading, error, noRows, ready, setupNeeded } =
    useOrganizationApps(organizationId);
  const month = customerPresetRange("month");
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
        open={adding || editing !== null}
        onClose={closeDialog}
        pending={pending}
        title={adding ? t("apps.addBudget") : t("apps.budgetFor", { alias: editing?.alias })}
        description={t("apps.organizationHint")}
        className="sm:max-w-2xl"
      >
        <div className="grid shrink-0 gap-4 px-6 pt-6 last:pb-6 sm:grid-cols-2">
          <div className="space-y-2">
            <label htmlFor="app-budget-customer" className="text-sm text-muted-foreground">
              {t("apps.customer")}
            </label>
            <Input id="app-budget-customer" value={organizationAlias || organizationId} readOnly />
          </div>
          {adding && resourceType ? (
            <OrganizationAppPicker resources={resources} selected={editing} pending={pending} onChange={setEditing} />
          ) : (
            <div className="space-y-2">
              <label htmlFor="app-budget-resource" className="text-sm text-muted-foreground">
                {t("apps.alias")}
              </label>
              <Input id="app-budget-resource" value={editing?.alias ?? ""} readOnly />
            </div>
          )}
        </div>
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
      </CustomerBudgetDialog>
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
  onEdit: (resource: AppResource) => void;
  onAppSelect?: (appId: string) => void;
  usageContext: AppUsageContext;
  usageLayout: boolean;
}) {
  const { t } = useTranslation();
  return (
    <AppBudgetTable scope="organization" usageLayout={usageLayout}>
      {rows.map(({ app, row }) => {
        const profileId = appOrganizationPresetId(row, app);
        const profileName = groups.find((group) => group.budget_id === profileId)?.name;
        let sourceLabel = t("apps.appDefault");
        if (row.budget || hasLegacyCustomerBudget(row))
          sourceLabel = t("customerPortal.organizationGroups.customBadge");
        else if (row.budget_id || row.tier_budget_id) sourceLabel = profileName ?? t("apps.profile");
        const inheritedProfile = !row.budget_id && !row.tier_budget_id && !hasLegacyCustomerBudget(row);
        if (inheritedProfile && profileName) sourceLabel = `${profileName} · ${sourceLabel}`;
        return (
          <AppBudgetRow
            key={app.app_id}
            scope="organization"
            app={app}
            row={row}
            sourceLabel={sourceLabel}
            metrics={{ layout: usageLayout ? "full" : "compact", context: usageContext }}
            onAppSelect={onAppSelect}
            onEdit={() => onEdit(app)}
          />
        );
      })}
    </AppBudgetTable>
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
  const row = association.data ?? null;
  const apps = useQuery({ queryKey: ["apps"], queryFn: () => loadApps() });
  const app = apps.data?.find((item) => item.app_id === resource.app_id);
  const presetId = appOrganizationPresetId(association.data, app);
  const groups = useAppBudgetGroups(association.data?.budget ? [] : [presetId]);
  const ownsPolicy = organizationBudgetSource(row) === "custom";
  const catalogError = (groups.isError || apps.isError) && (
    <ErrorState
      message={extractProxyErrorMessage(groups.error ?? apps.error)}
      onRetry={() => void Promise.all([groups.refetch(), apps.refetch()])}
    />
  );
  if (!ownsPolicy) {
    if (groups.isPending || apps.isPending) return <p role="status">{t("apps.loading")}</p>;
    if (catalogError) return catalogError;
  }
  if (resource.app_id && association.isPending) return <p role="status">{t("apps.loading")}</p>;
  if (association.isError)
    return (
      <ErrorState message={extractProxyErrorMessage(association.error)} onRetry={() => void association.refetch()} />
    );
  return (
    <>
      {catalogError}
      <OrganizationAppForm
        resource={resource}
        organizationId={organizationId}
        organizationAlias={organizationAlias}
        row={row}
        groups={groups.data?.default_budgets ?? []}
        inherited={groups.data?.default_budgets.find((group) => group.budget_id === presetId) ?? null}
        pending={pending}
        onPendingChange={onPendingChange}
        onClose={onClose}
      />
    </>
  );
}

function organizationBudgetSource(row: AppOrganization | null): "inherit" | "profile" | "custom" {
  if (row?.budget || hasLegacyCustomerBudget(row)) return "custom";
  if (row?.budget_id || row?.tier_budget_id) return "profile";
  return "inherit";
}

function OrganizationAppForm({
  resource,
  organizationId,
  organizationAlias,
  row,
  groups,
  inherited,
  pending,
  onPendingChange,
  onClose,
}: {
  resource: AppResource;
  organizationId: string;
  organizationAlias?: string | null;
  row: AppOrganization | null;
  groups: NonNullable<ReturnType<typeof useAppBudgetGroups>["data"]>["default_budgets"];
  inherited: import("../../customers/_components/customerBudgetForm").Budget | null;
  pending: boolean;
  onPendingChange: (pending: boolean) => void;
  onClose: () => void;
}) {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const [baseline] = useState(() => ({
    budget: appBaseBudget(row, inherited),
    member: row?.customer_member_budget_id ? row.effective_customer_member_budget : null,
  }));
  const original = baseline.budget;
  const memberOriginal = baseline.member;
  const form = useCustomerBudgetForm(
    {
      initial: {
        source:
          organizationBudgetSource(row) === "profile"
            ? `profile:${row?.budget_id ?? row?.tier_budget_id}`
            : organizationBudgetSource(row),
        budget: compactBudgetEditorFields(original),
        member: compactBudgetEditorFields(memberOriginal),
        policy: original,
      },
      parse: ({ source, budget, member, policy }) => {
        const parsed = customerBudgetPatch({ ...budget, budgetEnabled: source === "custom" }, original, row);
        const memberBudget = permanentBudgetEditorValue(member, memberOriginal);
        if (!parsed.ok || !memberBudget.ok) return { ok: false, error: t("customerPortal.hierarchy.invalidNumber") };
        const policyChanges = budgetChanges(
          {
            model_tpm_limit: policy?.model_tpm_limit ?? null,
            model_rpm_limit: policy?.model_rpm_limit ?? null,
            cap_policy: policy?.cap_policy ?? null,
          },
          original,
        );
        return {
          ok: true,
          value: {
            ...(source === "custom"
              ? { ...parsed.value, budget: { ...parsed.value.budget, ...policyChanges } }
              : { budget_id: source.startsWith("profile:") ? source.slice(8) : null }),
            ...(hasLegacyCustomerBudget(row) ? { budget_override: null } : {}),
            customer_member_budget: memberBudget.value,
          },
        };
      },
      write: async (body) => {
        const appId =
          resource.app_id ??
          (
            await fetchClient.POST("/management/v1/apps", {
              body: { resource_id: resource.resource_id, resource_type: resource.resource_type },
            })
          ).data?.app_id;
        if (!appId) throw new Error(t("apps.saveFailed"));
        await fetchClient.PUT("/management/v1/apps/{app_id}/customer_organizations/{customer_org_id}", {
          params: { path: { app_id: appId, customer_org_id: organizationId } },
          body,
        });
        await Promise.all(
          ["apps", "app-resources", "app-organization", "app-member"].map((key) =>
            queryClient.invalidateQueries({ queryKey: [key] }),
          ),
        );
        onClose();
      },
    },
    onPendingChange,
  );
  const { fields, update, error, save } = form;
  const changeSource = (source: string | null) => {
    if (!source) return;
    const selected = groups.find((group) => `profile:${group.budget_id}` === fields.source);
    if (source === "custom") {
      update("policy", selected ?? original);
      update("budget", compactBudgetEditorFields(selected ?? original));
    }
    update("source", source);
  };
  return (
    <CustomerBudgetForm
      onSubmit={save}
      pending={pending}
      error={error}
      onClose={onClose}
      saveLabel={t("apps.saveBudget")}
      label={t("apps.organizationSettings")}
    >
      <CustomerBudgetEditor
        id="app-budget-source"
        title={t("apps.source")}
        description={t("apps.customerAppLimitHint")}
        source={{
          profiles: groups,
          value: fields.source,
          onChange: changeSource,
          title: t("apps.source"),
          disabled: pending,
          inheritedOption: { label: t("apps.appDefault"), summary: t("apps.appDefaultHint") },
        }}
      >
        <CustomerBudgetCompactFields
          fields={fields.budget}
          original={original}
          onChange={(value) => update("budget", value)}
          disabled={pending}
          id="org-app-override"
        />
      </CustomerBudgetEditor>
      <section aria-labelledby="app-member-limits-heading" className="space-y-4 border-t pt-5">
        <div className="flex items-start justify-between gap-4">
          <div className="space-y-1">
            <h3 id="app-member-limits-heading" className="font-semibold">
              {t("apps.optionalMemberLimits")}
            </h3>
            <p className="text-sm text-muted-foreground">{t("apps.optionalMemberLimitsHint")}</p>
          </div>
          <Switch
            aria-label={t("apps.enableMemberLimits")}
            checked={fields.member.budgetEnabled}
            disabled={pending}
            onCheckedChange={(budgetEnabled) => update("member", { ...fields.member, budgetEnabled })}
          />
        </div>
        {fields.member.budgetEnabled && (
          <CustomerBudgetCompactFields
            fields={fields.member}
            original={row?.customer_member_budget_id ? row.effective_customer_member_budget : null}
            onChange={(value) => update("member", value)}
            disabled={pending}
            id="org-app-member"
          />
        )}
      </section>
      <p className="text-xs text-muted-foreground">
        {t("apps.onlyThisPair", { customer: organizationAlias || organizationId, alias: resource.alias })}
      </p>
    </CustomerBudgetForm>
  );
}
