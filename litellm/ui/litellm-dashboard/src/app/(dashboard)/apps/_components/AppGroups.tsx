"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { parseAsString, useQueryState } from "nuqs";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { SearchSelect } from "@/components/shared/SearchSelect";
import { ErrorState } from "@/components/shared/ErrorState";
import { extractProxyErrorMessage } from "@/lib/http/client";
import { customerOrganizationsCall } from "../../customers/_components/customerOrganizations";
import useAuthorized from "../../hooks/useAuthorized";
import AppBudgetLimits from "./AppBudgetLimits";
import AppGroupBudgetDialog, { type GroupBudgetTarget } from "./AppGroupBudgetDialog";
import { useAppGroups, loadAppGroupOrganizations, type AppGroup } from "./appGroupApi";
import type { App } from "./appForms";

export default function AppGroups({
  apps,
  onOpenApp,
}: {
  apps: readonly App[];
  onOpenApp: (resourceId: string) => void;
}) {
  const { t } = useTranslation();
  const groups = useAppGroups();
  const [selected, setSelected] = useQueryState("app_group", parseAsString);
  const [editing, setEditing] = useState<GroupBudgetTarget | null>(null);
  const group = groups.data?.find((item) => item.app_group_id === selected);
  const missingGroup = Boolean(selected) && groups.isSuccess && !group;
  return (
    <section className="space-y-4 rounded-lg border bg-card p-5" aria-label={t("apps.groups.title")}>
      <header className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h2 className="text-base font-semibold">{t("apps.groups.title")}</h2>
          <p className="mt-1 text-sm text-muted-foreground">{t("apps.groups.hint")}</p>
        </div>
        <Button variant="outline" onClick={() => setEditing({ kind: "group" })}>
          {t("apps.groups.create")}
        </Button>
      </header>
      {groups.isPending && <p role="status">{t("apps.loading")}</p>}
      {groups.isError && (
        <ErrorState message={extractProxyErrorMessage(groups.error)} onRetry={() => void groups.refetch()} />
      )}
      {groups.data && groups.data.length === 0 && (
        <p className="text-sm text-muted-foreground">{t("apps.groups.empty")}</p>
      )}
      {Boolean(groups.data?.length) && (
        <div className="max-w-md space-y-2">
          <label htmlFor="selected-app-group" className="text-sm font-medium">
            {t("apps.groups.choose")}
          </label>
          <SearchSelect
            id="selected-app-group"
            className="w-full"
            value={selected ?? undefined}
            onValueChange={(value) => void setSelected(value)}
            options={(groups.data ?? []).map((row) => ({ value: row.app_group_id, label: row.name }))}
          />
        </div>
      )}
      {missingGroup && <p role="alert">{t("apps.groups.notFound")}</p>}
      {group && (
        <GroupDetails
          group={group}
          apps={apps.filter((app) => app.app_group_id === group.app_group_id)}
          onOpenApp={onOpenApp}
          onEdit={setEditing}
        />
      )}
      {editing && <AppGroupBudgetDialog target={editing} onClose={() => setEditing(null)} />}
    </section>
  );
}

function GroupDetails({
  group,
  apps,
  onOpenApp,
  onEdit,
}: {
  group: AppGroup;
  apps: readonly App[];
  onOpenApp: (resourceId: string) => void;
  onEdit: (target: GroupBudgetTarget) => void;
}) {
  const { t } = useTranslation();
  const { accessToken } = useAuthorized();
  const associations = useQuery({
    queryKey: ["app-group-organizations", group.app_group_id],
    queryFn: () => loadAppGroupOrganizations(group.app_group_id),
  });
  const organizations = useQuery({
    queryKey: ["app-group-customer-options"],
    queryFn: ({ signal }) => customerOrganizationsCall(accessToken!, signal),
    enabled: Boolean(accessToken),
  });
  const labels = new Map(organizations.data?.map((org) => [org.customer_org_id, org.alias || org.customer_org_id]));
  return (
    <div className="space-y-6 border-t pt-5">
      <div className="flex flex-wrap justify-between gap-4">
        <div className="space-y-3">
          <h3 className="font-semibold">{group.name}</h3>
          {group.description && <p className="text-sm text-muted-foreground">{group.description}</p>}
          <AppBudgetLimits budget={group.budget} usage={group.budget_usage} cumulativeSpend={group.spend} />
        </div>
        <Button variant="outline" onClick={() => onEdit({ kind: "group", group })}>
          {t("apps.groups.edit")}
        </Button>
      </div>
      <div className="space-y-3">
        <h3 className="text-sm font-semibold">{t("apps.groups.apps")}</h3>
        <p className="text-xs text-muted-foreground">{t("apps.groups.appLimitsHint")}</p>
        {apps.length === 0 && <p className="text-sm text-muted-foreground">{t("apps.groups.noApps")}</p>}
        {apps.map((app) => (
          <div key={app.app_id} className="flex flex-wrap items-center justify-between gap-4 rounded-md border p-4">
            <span className="text-sm font-medium">{app.alias}</span>
            <AppBudgetLimits budget={app.budget} usage={app.budget_usage} cumulativeSpend={app.spend} />
            <Button variant="outline" onClick={() => onOpenApp(app.resource_id)}>
              {t("apps.groups.editApp")}
            </Button>
          </div>
        ))}
      </div>
      <div className="space-y-3 border-t pt-4">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <h3 className="text-sm font-semibold">{t("apps.groups.customers")}</h3>
          <Button variant="outline" onClick={() => onEdit({ kind: "organization", group })}>
            {t("apps.groups.setCustomerBudget")}
          </Button>
        </div>
        {associations.isPending && <p role="status">{t("apps.loading")}</p>}
        {associations.isError && (
          <ErrorState
            message={extractProxyErrorMessage(associations.error)}
            onRetry={() => void associations.refetch()}
          />
        )}
        {associations.data?.length === 0 && (
          <p className="text-sm text-muted-foreground">{t("apps.groups.noCustomers")}</p>
        )}
        {associations.data?.map((organization) => (
          <div
            key={organization.customer_org_id}
            className="flex flex-wrap items-center justify-between gap-4 rounded-md border p-4"
          >
            <span className="text-sm font-medium">
              {labels.get(organization.customer_org_id) ?? organization.customer_org_id}
            </span>
            <AppBudgetLimits
              budget={organization.budget}
              usage={organization.budget_usage}
              cumulativeSpend={organization.spend}
            />
            <Button variant="outline" onClick={() => onEdit({ kind: "organization", group, organization })}>
              {t("apps.groups.editBudget")}
            </Button>
          </div>
        ))}
      </div>
    </div>
  );
}
