"use client";

import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { Input } from "@/components/ui/input";
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
import { useOverallAppBudgetGroups, useAppBudgetGroups } from "./useAppBudgetGroups";
import type { AppGroup, AppGroupOrganization } from "./appGroupApi";

export type GroupBudgetTarget =
  | { kind: "group"; group?: AppGroup }
  | { kind: "organization"; group: AppGroup; organization?: AppGroupOrganization };

export default function AppGroupBudgetDialog({ target, onClose }: { target: GroupBudgetTarget; onClose: () => void }) {
  const { t } = useTranslation();
  const [pending, setPending] = useState(false);
  const overall = useOverallAppBudgetGroups(target.kind === "group");
  const organization = useAppBudgetGroups([], target.kind === "organization");
  const catalog = target.kind === "group" ? overall : organization;
  const groupTitle = target.group ? "apps.groups.edit" : "apps.groups.create";
  return (
    <CustomerBudgetDialog
      title={t(target.kind === "organization" ? "apps.groups.customerBudget" : groupTitle)}
      description={t("apps.groups.capHint")}
      pending={pending}
      onClose={onClose}
    >
      {catalog.isPending && (
        <p role="status" className="p-6">
          {t("apps.loading")}
        </p>
      )}
      {catalog.isError && (
        <div className="p-6">
          <ErrorState message={extractProxyErrorMessage(catalog.error)} onRetry={() => void catalog.refetch()} />
        </div>
      )}
      {catalog.data && (
        <GroupBudgetForm
          target={target}
          profiles={catalog.data.default_budgets}
          onClose={onClose}
          onPendingChange={setPending}
        />
      )}
    </CustomerBudgetDialog>
  );
}

function GroupBudgetForm({
  target,
  profiles,
  onClose,
  onPendingChange,
}: {
  target: GroupBudgetTarget;
  profiles: BudgetGroups["default_budgets"];
  onClose: () => void;
  onPendingChange: (pending: boolean) => void;
}) {
  const { t } = useTranslation();
  const { accessToken } = useAuthorized();
  const queryClient = useQueryClient();
  const entity = target.kind === "group" ? target.group : target.organization;
  const organizations = useQuery({
    queryKey: ["app-group-customer-options"],
    queryFn: ({ signal }) => customerOrganizationsCall(accessToken!, signal),
    enabled: target.kind === "organization" && Boolean(accessToken),
  });
  const form = useCustomerBudgetForm(
    {
      initial: {
        name: target.group?.name ?? "",
        description: target.group?.description ?? "",
        organization: target.kind === "organization" ? target.organization?.customer_org_id ?? "" : "",
        ...budgetSelectionFields(entity, profiles),
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
            description: fields.description || null,
            organization: fields.organization,
            selection: budget.value,
          },
        };
      },
      write: async ({ name, description, organization, selection }) => {
        if (target.kind === "organization") {
          await fetchClient.PUT("/management/v1/app_groups/{app_group_id}/customer_organizations/{customer_org_id}", {
            params: { path: { app_group_id: target.group.app_group_id, customer_org_id: organization } },
            body: selection,
          });
        } else if (target.group) {
          await fetchClient.PATCH("/management/v1/app_groups/{app_group_id}", {
            params: { path: { app_group_id: target.group.app_group_id } },
            body: { name, description, ...selection },
          });
        } else {
          await fetchClient.POST("/management/v1/app_groups", { body: { name, description, ...selection } });
        }
        await Promise.all(
          ["app-groups", "app-group-organizations"].map((key) => queryClient.invalidateQueries({ queryKey: [key] })),
        );
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
          <div className="space-y-2">
            <label htmlFor="app-group-name" className="text-sm font-medium">
              {t("apps.groups.name")}
            </label>
            <Input
              id="app-group-name"
              required
              maxLength={255}
              value={form.fields.name}
              onChange={(event) => form.update("name", event.target.value)}
            />
          </div>
          <div className="space-y-2">
            <label htmlFor="app-group-description" className="text-sm font-medium">
              {t("apps.groups.description")}
            </label>
            <Input
              id="app-group-description"
              value={form.fields.description}
              onChange={(event) => form.update("description", event.target.value)}
            />
          </div>
        </>
      ) : (
        <div className="space-y-2">
          <label htmlFor="app-group-customer" className="text-sm font-medium">
            {t("apps.groups.customer")}
          </label>
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
        </div>
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
    </CustomerBudgetForm>
  );
}
