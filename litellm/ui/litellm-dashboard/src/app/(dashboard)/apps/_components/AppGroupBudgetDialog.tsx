"use client";

import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { uiHref } from "@/utils/uiHref";
import { Input } from "@/components/ui/input";
import { Field, FieldLabel } from "@/components/ui/field";
import { MultiSelect } from "@/components/shared/MultiSelect";
import { SearchSelect } from "@/components/shared/SearchSelect";
import { ErrorState } from "@/components/shared/ErrorState";
import { fetchClient } from "@/lib/http/api";
import { extractProxyErrorMessage } from "@/lib/http/client";
import { CustomerBudgetDialog, CustomerBudgetForm } from "../../customers/_components/CustomerBudgetDialog";
import { CustomerBudgetEditor } from "../../customers/_components/CustomerLimitsFields";
import { useCustomerBudgetForm } from "../../customers/_components/useCustomerBudgetForm";
import {
  budgetSelectionFields,
  budgetSelectionValue,
  changeBudgetSelection,
} from "../../customers/_components/customerBudgetForm";
import type { BudgetGroups } from "../../customers/_components/customerBudgetGroupsPlan";
import { customerOrganizationsCall } from "../../customers/_components/customerOrganizations";
import useAuthorized from "../../hooks/useAuthorized";
import { useAppGroupBudgetGroups, useAppBudgetGroups } from "./useAppBudgetGroups";
import { useManagedApps } from "./useManagedApps";
import { invalidateAppBudgets } from "./appApi";
import type { App, AppResource } from "./appForms";
import type { AppGroup, AppGroupOrganization } from "./appGroupApi";

export type GroupBudgetTarget =
  | { kind: "group"; group?: AppGroup }
  | { kind: "organization"; group: AppGroup; organization?: AppGroupOrganization };

export default function AppGroupBudgetDialog({ target, onClose }: { target: GroupBudgetTarget; onClose: () => void }) {
  const { t } = useTranslation();
  const [pending, setPending] = useState(false);
  const overall = useAppGroupBudgetGroups(target.kind === "group");
  const organization = useAppBudgetGroups([], target.kind === "organization");
  const catalog = target.kind === "group" ? overall : organization;
  const managed = useManagedApps(target.kind === "group");
  const membershipQueries =
    target.kind === "group"
      ? [managed.definition, ...(managed.resourceType ? [managed.resources, managed.apps] : [])]
      : [];
  const queries = [catalog, ...membershipQueries];
  const loading = queries.some((query) => query.isPending);
  const failed = queries.find((query) => query.isError);
  const groupTitle = target.group ? "apps.groups.edit" : "apps.groups.create";
  return (
    <CustomerBudgetDialog
      title={t(target.kind === "organization" ? "apps.groups.customerBudget" : groupTitle)}
      description={t("apps.groups.capHint")}
      pending={pending}
      onClose={onClose}
    >
      {loading && (
        <p role="status" className="p-6">
          {t("apps.loading")}
        </p>
      )}
      {failed && (
        <div className="p-6">
          <ErrorState message={extractProxyErrorMessage(failed.error)} onRetry={() => void failed.refetch()} />
        </div>
      )}
      {catalog.data && !loading && !failed && (
        <GroupBudgetForm
          target={target}
          resources={managed.resources.data ?? []}
          apps={managed.apps.data ?? []}
          profiles={catalog.data.default_budgets}
          defaultBudgetId={target.kind === "group" ? catalog.data.default_budget_template_id : null}
          onClose={onClose}
          onPendingChange={setPending}
        />
      )}
    </CustomerBudgetDialog>
  );
}

function GroupBudgetForm({
  target,
  resources,
  apps,
  profiles,
  defaultBudgetId,
  onClose,
  onPendingChange,
}: {
  target: GroupBudgetTarget;
  resources: readonly AppResource[];
  apps: readonly App[];
  profiles: BudgetGroups["default_budgets"];
  defaultBudgetId?: string | null;
  onClose: () => void;
  onPendingChange: (pending: boolean) => void;
}) {
  const { t } = useTranslation();
  const { accessToken, userId } = useAuthorized();
  const queryClient = useQueryClient();
  const entity = target.kind === "group" ? target.group : target.organization;
  const organizations = useQuery({
    queryKey: ["customer-organizations", userId],
    queryFn: ({ signal }) => customerOrganizationsCall(accessToken!, signal),
    enabled: target.kind === "organization" && Boolean(accessToken),
  });
  const form = useCustomerBudgetForm(
    {
      initial: {
        name: target.group?.name ?? "",
        appResources: apps
          .filter((app) => target.group && app.app_group_id === target.group.app_group_id)
          .map((app) => app.resource_id),
        description: target.group?.description ?? "",
        organization: target.kind === "organization" ? target.organization?.customer_org_id ?? "" : "",
        ...budgetSelectionFields(entity, profiles, defaultBudgetId),
      },
      parse: (fields) => {
        if (target.kind === "group" && !fields.name.trim()) return { ok: false, error: t("apps.groups.nameRequired") };
        if (target.kind === "organization" && !fields.organization)
          return { ok: false, error: t("apps.groups.customerRequired") };
        const budget = budgetSelectionValue(fields, entity?.budget);
        if (!budget.ok) return { ok: false, error: t(budget.error) };
        return {
          ok: true,
          value: {
            name: fields.name.trim(),
            selectedApps: resources
              .filter((resource) => fields.appResources.includes(resource.resource_id))
              .map(({ resource_type, resource_id }) => ({ resource_type, resource_id })),
            description: fields.description || null,
            organization: fields.organization,
            selection: budget.value,
          },
        };
      },
      write: async ({ name, description, organization, selection, selectedApps }) => {
        if (target.kind === "organization") {
          await fetchClient.PUT("/management/v1/app_groups/{app_group_id}/customer_organizations/{customer_org_id}", {
            params: { path: { app_group_id: target.group.app_group_id, customer_org_id: organization } },
            body: selection,
          });
        } else if (target.group) {
          await fetchClient.PATCH("/management/v1/app_groups/{app_group_id}", {
            params: { path: { app_group_id: target.group.app_group_id } },
            body: { name, description, apps: selectedApps, ...selection },
          });
        } else {
          await fetchClient.POST("/management/v1/app_groups", {
            body: { name, description, apps: selectedApps, ...selection },
          });
        }
        await invalidateAppBudgets(queryClient);
        onClose();
      },
    },
    onPendingChange,
  );
  const changeSource = (source: string | null) => {
    if (!source) return;
    form.setFields((current) => ({ ...current, ...changeBudgetSelection(current, source, profiles) }));
  };
  return (
    <CustomerBudgetForm
      pending={form.pending}
      error={form.error}
      onSubmit={form.save}
      onClose={onClose}
      label={t("apps.groups.budgetSettings")}
    >
      {target.kind === "group" ? (
        <>
          <Field>
            <FieldLabel htmlFor="app-group-name">{t("apps.groups.name")}</FieldLabel>
            <Input
              id="app-group-name"
              required
              maxLength={255}
              value={form.fields.name}
              onChange={(event) => form.update("name", event.target.value)}
            />
          </Field>
          <Field>
            <FieldLabel htmlFor="app-group-apps">{t("apps.groups.apps")}</FieldLabel>
            <MultiSelect
              id="app-group-apps"
              placeholder={t("apps.groups.selectApps")}
              value={form.fields.appResources}
              onValueChange={(value) => form.update("appResources", value)}
              disabled={form.pending}
              options={resources.map((resource) => ({
                value: resource.resource_id,
                label: resource.alias,
                description: resource.resource_id,
              }))}
            />
            <p className="text-xs text-muted-foreground">{t("apps.groups.selectionHint")}</p>
          </Field>
          <Field>
            <FieldLabel htmlFor="app-group-description">{t("apps.groups.description")}</FieldLabel>
            <Input
              id="app-group-description"
              value={form.fields.description}
              onChange={(event) => form.update("description", event.target.value)}
            />
          </Field>
        </>
      ) : (
        <Field>
          <FieldLabel htmlFor="app-group-customer">{t("apps.groups.customer")}</FieldLabel>
          <SearchSelect
            id="app-group-customer"
            className="w-full"
            value={form.fields.organization || undefined}
            onValueChange={(value) => form.update("organization", value ?? "")}
            disabled={Boolean(target.organization) || organizations.isPending}
            options={(organizations.data ?? []).map((org) => ({
              value: org.customer_org_id,
              label: org.alias || org.customer_org_id,
            }))}
          />
          {organizations.isError && (
            <ErrorState
              message={extractProxyErrorMessage(organizations.error)}
              onRetry={() => void organizations.refetch()}
            />
          )}
        </Field>
      )}
      <CustomerBudgetEditor
        id="app-group-budget"
        title={t("apps.groups.sharedBudget")}
        description={t("apps.groups.capHint")}
        budget={{
          fields: form.fields.budget,
          onChange: (value) => form.update("budget", value),
          disabled: form.pending,
        }}
        source={{
          profiles,
          value: form.fields.budgetSource,
          onChange: changeSource,
          title: t("apps.source"),
          disabled: form.pending,
          inheritedOption: { label: t("customerPortal.createDialog.noLimit"), summary: t("apps.noAdditionalLimits") },
        }}
      />
      {target.kind === "group" && (
        <a className="text-sm text-primary hover:underline" href={`${uiHref("customers/settings")}?tab=app-group`}>
          {t("apps.groups.manageBudgetGroups")}
        </a>
      )}
    </CustomerBudgetForm>
  );
}
