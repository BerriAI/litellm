"use client";

import { useState } from "react";
import { useQueryStates } from "nuqs";
import { useTranslation } from "react-i18next";
import { PageHeader, PageHeaderTitle, PageHeaderDescription } from "@/components/shared/PageHeader";
import { AdminOnlyNotice } from "@/components/shared/AdminOnlyNotice";
import { ErrorState } from "@/components/shared/ErrorState";
import { extractProxyErrorMessage } from "@/lib/http/client";
import useAuthorized from "../../hooks/useAuthorized";
import AppGroupBudgetDialog, { type GroupBudgetTarget } from "./AppGroupBudgetDialog";
import AppGroupsTable from "./AppGroupsTable";
import { readsEveryApp } from "./useGrantedApps";
import { useAppGroups } from "./appGroupApi";
import AppGroupDetails, { appGroupDetailParsers } from "./AppGroupDetails";

export default function AppGroups() {
  const { t } = useTranslation();
  const { accessToken, userRole, isViewOnly, isLoading } = useAuthorized();
  if (isLoading) return <p role="status">{t("apps.loading")}</p>;
  if (!accessToken || !readsEveryApp(userRole, isViewOnly))
    return <AdminOnlyNotice framed pageTitle={t("apps.groups.title")} />;
  return <ManagedAppGroups />;
}

function ManagedAppGroups() {
  const { t } = useTranslation();
  const groups = useAppGroups();
  const [params, setParams] = useQueryStates(appGroupDetailParsers);
  const selected = params.app_group;
  const setSelected = (app_group: string | null) => {
    const selection = { app_group, group_tab: null, app_id: null, app: null };
    void setParams(selection, { history: "push" });
  };
  const [editing, setEditing] = useState<GroupBudgetTarget | null>(null);
  const group = groups.data?.find((item) => item.app_group_id === selected);
  if (group)
    return (
      <>
        <AppGroupDetails group={group} onEdit={setEditing} onBack={() => setSelected(null)} />
        {editing && <AppGroupBudgetDialog target={editing} onClose={() => setEditing(null)} />}
      </>
    );
  return (
    <>
      <PageHeader>
        <PageHeaderTitle>{t("apps.groups.title")}</PageHeaderTitle>
        <PageHeaderDescription>{t("apps.groups.hint")}</PageHeaderDescription>
      </PageHeader>
      {groups.isError && (
        <ErrorState message={extractProxyErrorMessage(groups.error)} onRetry={() => void groups.refetch()} />
      )}
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
    </>
  );
}
