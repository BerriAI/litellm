"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { parseAsString, useQueryState } from "nuqs";
import { useTranslation } from "react-i18next";
import { ArrowLeft, Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardDescription, CardAction, CardContent } from "@/components/ui/card";
import { TableCell, TableHead, TableRow } from "@/components/ui/table";
import { Page } from "@/components/shared/Page";
import { PageHeader, PageHeaderTitle, PageHeaderDescription } from "@/components/shared/PageHeader";
import { AdminOnlyNotice } from "@/components/shared/AdminOnlyNotice";
import { ErrorState } from "@/components/shared/ErrorState";
import { extractProxyErrorMessage } from "@/lib/http/client";
import { customerOrganizationsCall } from "../../customers/_components/customerOrganizations";
import { CustomerUsageTable, CustomerUsageEditButton } from "../../customers/_components/CustomerUsageTable";
import useAuthorized from "../../hooks/useAuthorized";
import AppBudgetLimits from "./AppBudgetLimits";
import AppGroupBudgetDialog, { type GroupBudgetTarget } from "./AppGroupBudgetDialog";
import AppSettingsDialog from "./AppSettingsDialog";
import ApplicationsTable from "./ApplicationsTable";
import AppGroupsTable from "./AppGroupsTable";
import { grantedApplicationRows } from "./applications";
import { loadApps } from "./appApi";
import { readsEveryApp } from "./useGrantedApps";
import { useAppGroups, loadAppGroupOrganizations, type AppGroup } from "./appGroupApi";

export default function AppGroups() {
  const { t } = useTranslation();
  const { accessToken, userRole, isViewOnly, isLoading } = useAuthorized();
  if (isLoading) return <p role="status">{t("apps.loading")}</p>;
  if (!accessToken || !readsEveryApp(userRole, isViewOnly))
    return <AdminOnlyNotice pageTitle={t("apps.groups.title")} />;
  return <ManagedAppGroups />;
}

function ManagedAppGroups() {
  const { t } = useTranslation();
  const groups = useAppGroups();
  const [selected, setSelected] = useQueryState("app_group", parseAsString);
  const [editing, setEditing] = useState<GroupBudgetTarget | null>(null);
  const group = groups.data?.find((item) => item.app_group_id === selected);
  return (
    <Page>
      <PageHeader>
        {selected && (
          <Button variant="ghost" className="mb-3" onClick={() => void setSelected(null)}>
            <ArrowLeft /> {t("apps.groups.title")}
          </Button>
        )}
        <PageHeaderTitle>{group?.name ?? t("apps.groups.title")}</PageHeaderTitle>
        <PageHeaderDescription>{group?.description || t("apps.groups.hint")}</PageHeaderDescription>
      </PageHeader>
      {groups.isError && (
        <ErrorState message={extractProxyErrorMessage(groups.error)} onRetry={() => void groups.refetch()} />
      )}
      {group && <GroupDetails key={group.app_group_id} group={group} onEdit={setEditing} />}
      {selected && !group && !groups.isError && (
        <p role={groups.isPending ? "status" : "alert"}>
          {t(groups.isPending ? "apps.loading" : "apps.groups.notFound")}
        </p>
      )}
      {!selected && (
        <AppGroupsTable
          groups={groups.data ?? []}
          loading={groups.isPending}
          onOpen={(id) => void setSelected(id)}
          onCreate={() => setEditing({ kind: "group" })}
          onEdit={(group) => setEditing({ kind: "group", group })}
        />
      )}
      {editing && <AppGroupBudgetDialog target={editing} onClose={() => setEditing(null)} />}
    </Page>
  );
}

function GroupDetails({ group, onEdit }: { group: AppGroup; onEdit: (target: GroupBudgetTarget) => void }) {
  const { t } = useTranslation();
  const { accessToken, userId } = useAuthorized();
  const [selectedApp, setSelectedApp] = useQueryState("app", parseAsString);
  const apps = useQuery({ queryKey: ["apps"], queryFn: () => loadApps() });
  const rows = grantedApplicationRows((apps.data ?? []).filter((app) => app.app_group_id === group.app_group_id));
  const resource = rows.find((row) => row.resource.resource_id === selectedApp)?.resource;
  const associations = useQuery({
    queryKey: ["app-group-organizations", group.app_group_id],
    queryFn: () => loadAppGroupOrganizations(group.app_group_id),
  });
  const organizations = useQuery({
    queryKey: ["customer-organizations", userId],
    queryFn: ({ signal }) => customerOrganizationsCall(accessToken!, signal),
    enabled: Boolean(accessToken),
  });
  const emptyMessage = associations.data?.length === 0 ? t("apps.groups.noCustomers") : undefined;
  const labels = new Map(organizations.data?.map((org) => [org.customer_org_id, org.alias || org.customer_org_id]));
  return (
    <>
      <Card>
        <CardHeader>
          <CardTitle>{t("apps.groups.sharedBudget")}</CardTitle>
          <CardDescription>{t("apps.groups.capHint")}</CardDescription>
          <CardAction>
            <CustomerUsageEditButton onClick={() => onEdit({ kind: "group", group })}>
              {t("apps.groups.edit")}
            </CustomerUsageEditButton>
          </CardAction>
        </CardHeader>
        <CardContent>
          <AppBudgetLimits budget={group.budget} usage={group.budget_usage} cumulativeSpend={group.spend} />
        </CardContent>
      </Card>
      <Card className="overflow-hidden pb-0">
        <CardHeader>
          <CardTitle>{t("apps.groups.apps")}</CardTitle>
          <CardDescription>{t("apps.groups.appLimitsHint")}</CardDescription>
        </CardHeader>
        {apps.isError ? (
          <ErrorState message={extractProxyErrorMessage(apps.error)} onRetry={() => void apps.refetch()} />
        ) : (
          <ApplicationsTable rows={rows} loading={apps.isPending} onOpen={(id) => void setSelectedApp(id)} />
        )}
      </Card>
      <Card className="overflow-hidden pb-0">
        <CardHeader>
          <CardTitle>{t("apps.groups.customers")}</CardTitle>
          <CardAction>
            <Button variant="outline" size="sm" onClick={() => onEdit({ kind: "organization", group })}>
              <Plus /> {t("apps.groups.setCustomerBudget")}
            </Button>
          </CardAction>
        </CardHeader>
        {associations.isError ? (
          <ErrorState
            message={extractProxyErrorMessage(associations.error)}
            onRetry={() => void associations.refetch()}
          />
        ) : (
          <CustomerUsageTable
            label={t("apps.groups.customers")}
            columnCount={3}
            message={associations.isPending ? t("apps.loading") : emptyMessage}
            headers={
              <>
                <TableHead className="px-4">{t("apps.groups.customer")}</TableHead>
                <TableHead className="px-4">{t("apps.groups.sharedBudget")}</TableHead>
                <TableHead className="px-4 text-right">{t("apps.actions")}</TableHead>
              </>
            }
          >
            {associations.data?.map((organization) => (
              <TableRow key={organization.customer_org_id}>
                <TableCell className="px-4 py-5">
                  {labels.get(organization.customer_org_id) ?? organization.customer_org_id}
                </TableCell>
                <TableCell className="px-4">
                  <AppBudgetLimits
                    budget={organization.budget}
                    usage={organization.budget_usage}
                    cumulativeSpend={organization.spend}
                  />
                </TableCell>
                <TableCell className="px-4 text-right">
                  <CustomerUsageEditButton onClick={() => onEdit({ kind: "organization", group, organization })}>
                    {t("apps.groups.editBudget")}
                  </CustomerUsageEditButton>
                </TableCell>
              </TableRow>
            ))}
          </CustomerUsageTable>
        )}
      </Card>
      {resource && <AppSettingsDialog resource={resource} onClose={() => void setSelectedApp(null)} />}
    </>
  );
}
