"use client";

import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import type { TFunction } from "i18next";
import { Input } from "@/components/ui/input";
import { ErrorState } from "@/components/shared/ErrorState";
import { fetchClient } from "@/lib/http/api";
import { extractProxyErrorMessage } from "@/lib/http/client";
import { CustomerBudgetDialog, CustomerBudgetForm } from "../../customers/_components/CustomerBudgetDialog";
import { useCustomerBudgetForm } from "../../customers/_components/useCustomerBudgetForm";
import { CustomerBudgetEditor } from "../../customers/_components/CustomerLimitsFields";
import {
  compactBudgetEditorFields,
  budgetPolicyChanges,
  customerBudgetPatch,
  hasLegacyCustomerBudget,
} from "../../customers/_components/customerBudgetForm";
import { appBaseBudget, appOrganizationPresetId, type App, type AppResource, type AppOrganization } from "./appForms";
import { organizationAppKey, loadOrganizationApp } from "./appApi";
import { useAppBudgetGroups } from "./useAppBudgetGroups";

export function AppOrganizationBudgetDialog({
  app,
  organizationId,
  organizationAlias,
  onClose,
}: {
  app: App;
  organizationId: string;
  organizationAlias: string;
  onClose: () => void;
}) {
  const { t } = useTranslation();
  const [pending, setPending] = useState(false);
  return (
    <CustomerBudgetDialog
      onClose={onClose}
      pending={pending}
      title={t("apps.customerAppBudget", { customer: organizationAlias || organizationId })}
      description={t("apps.customerAppBudgetDescription")}
      className="sm:max-w-2xl"
    >
      <div className="shrink-0 space-y-2 px-6 pt-6">
        <label htmlFor="app-budget-resource" className="block text-sm text-muted-foreground">
          {t("apps.alias")}
        </label>
        <Input id="app-budget-resource" value={app.alias} readOnly />
      </div>
      <OrganizationApp
        key={`${app.app_id}:${organizationId}`}
        resource={app}
        organizationId={organizationId}
        organizationAlias={organizationAlias}
        pending={pending}
        onPendingChange={setPending}
        onClose={onClose}
      />
    </CustomerBudgetDialog>
  );
}

export function appOrganizationSourceLabel(
  app: App,
  row: AppOrganization | null,
  groups: readonly { budget_id: string; name: string }[],
  t: TFunction,
) {
  const profileId = appOrganizationPresetId(row, app);
  const profileName = groups.find((group) => group.budget_id === profileId)?.name;
  const labels = {
    inherit: t("apps.appDefault"),
    profile: profileName ?? t("apps.profile"),
    custom: t("customerPortal.organizationGroups.customBadge"),
  };
  const sourceLabel = labels[organizationBudgetSource(row)];
  const inheritedProfile = !row?.budget_id && !row?.tier_budget_id && !hasLegacyCustomerBudget(row);
  return inheritedProfile && profileName ? `${profileName} · ${sourceLabel}` : sourceLabel;
}

export function OrganizationApp({
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
  const app = useQuery({
    queryKey: ["app", resource.app_id],
    queryFn: async () =>
      (
        await fetchClient.GET("/management/v1/apps/{app_id}", {
          params: { path: { app_id: resource.app_id! } },
        })
      ).data,
    enabled: Boolean(resource.app_id),
  });
  const presetId = appOrganizationPresetId(association.data, app.data);
  const groups = useAppBudgetGroups(association.data?.budget ? [] : [presetId]);
  const catalog = groups.data ?? { default_budgets: [], legacy_budgets: [] };
  const legacyGroup = catalog.legacy_budgets.find((group) => group.budget_id === presetId);
  const ownsPolicy = organizationBudgetSource(row) === "custom";
  const catalogError = (groups.isError || app.isError) && (
    <ErrorState
      message={extractProxyErrorMessage(groups.error ?? app.error)}
      onRetry={() => void Promise.all([groups.refetch(), ...(resource.app_id ? [app.refetch()] : [])])}
    />
  );
  if (!ownsPolicy) {
    if (groups.isPending || (resource.app_id && app.isPending)) return <p role="status">{t("apps.loading")}</p>;
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
        groups={catalog.default_budgets}
        inherited={groups.displayBudgets.find((group) => group.budget_id === presetId) ?? null}
        legacyGroup={legacyGroup}
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
  legacyGroup,
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
  legacyGroup?: NonNullable<ReturnType<typeof useAppBudgetGroups>["data"]>["legacy_budgets"][number];
  pending: boolean;
  onPendingChange: (pending: boolean) => void;
  onClose: () => void;
}) {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const [original] = useState(() => appBaseBudget(row, inherited));
  const form = useCustomerBudgetForm(
    {
      initial: {
        source:
          organizationBudgetSource(row) === "profile"
            ? `profile:${row?.budget_id ?? row?.tier_budget_id}`
            : organizationBudgetSource(row),
        budget: compactBudgetEditorFields(original),
        policy: original,
      },
      parse: ({ source, budget, policy }) => {
        const parsed = customerBudgetPatch({ ...budget, budgetEnabled: source === "custom" }, original, row);
        if (!parsed.ok) return { ok: false, error: t(parsed.error) };
        const policyChanges = budgetPolicyChanges(policy, original);
        return {
          ok: true,
          value: {
            ...(source === "custom"
              ? { ...parsed.value, budget: { ...parsed.value.budget, ...policyChanges } }
              : { budget_id: source.startsWith("profile:") ? source.slice(8) : null }),
            ...(hasLegacyCustomerBudget(row) ? { budget_override: null } : {}),
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
        budget={{
          modelScope: resource,
          fields: fields.budget,
          onChange: (value) => update("budget", value),
          disabled: pending,
          id: "org-app-override",
        }}
        source={{
          profiles: groups,
          value: fields.source,
          onChange: changeSource,
          title: t("apps.source"),
          disabled: pending,
          inheritedOption: { label: t("apps.appDefault"), summary: t("apps.appDefaultHint") },
          unavailableSelection: legacyGroup && {
            value: `profile:${legacyGroup.budget_id}`,
            label: t("apps.legacyOrganizationGroup", { name: legacyGroup.name }),
            summary: t("apps.legacyOrganizationGroupHint"),
          },
        }}
      />
      <p className="text-xs text-muted-foreground">
        {t("apps.onlyThisPair", { customer: organizationAlias || organizationId, alias: resource.alias })}
      </p>
    </CustomerBudgetForm>
  );
}
