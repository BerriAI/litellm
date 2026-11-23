"use client";

import { useTranslation } from "react-i18next";
import { ErrorState } from "@/components/shared/ErrorState";
import { extractProxyErrorMessage } from "@/lib/http/client";
import AppDetailPage from "../_components/AppDetailPage";
import AppSettingsForm from "../_components/AppSettingsForm";
import { grantedApplicationRows } from "../_components/applications";
import { useOverallAppBudgetGroups } from "../_components/useAppBudgetGroups";
import type { App } from "../_components/appForms";

export default function AppSettingsPage() {
  return <AppDetailPage active="settings">{(app) => <AppSettings key={app.app_id} app={app} />}</AppDetailPage>;
}

function AppSettings({ app }: { app: App }) {
  const { t } = useTranslation();
  const groups = useOverallAppBudgetGroups();
  const row = grantedApplicationRows([app])[0];
  if (groups.isPending) return <p role="status">{t("apps.loading")}</p>;
  if (groups.isError)
    return <ErrorState message={extractProxyErrorMessage(groups.error)} onRetry={() => void groups.refetch()} />;
  if (!row || !groups.data) return null;
  return <AppSettingsForm app={app} resource={row.resource} profiles={groups.data.default_budgets} />;
}
