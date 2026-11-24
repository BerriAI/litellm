"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { ErrorState } from "@/components/shared/ErrorState";
import { fetchClient } from "@/lib/http/api";
import { extractProxyErrorMessage } from "@/lib/http/client";
import { CustomerBudgetDialog } from "../../customers/_components/CustomerBudgetDialog";
import AppSettingsForm from "./AppSettingsForm";
import { useOverallAppBudgetGroups } from "./useAppBudgetGroups";
import type { AppResource } from "./appForms";

export default function AppSettingsDialog({ resource, onClose }: { resource: AppResource; onClose: () => void }) {
  const { t } = useTranslation();
  const [pending, setPending] = useState(false);
  const groups = useOverallAppBudgetGroups();
  const app = useQuery({
    queryKey: ["app", resource.app_id],
    queryFn: async () => {
      const response = await fetchClient.GET("/management/v1/apps/{app_id}", {
        params: { path: { app_id: resource.app_id! } },
      });
      return response.data;
    },
    enabled: resource.app_id !== null,
  });
  const loading = groups.isPending || (resource.app_id && app.isPending);
  const failed = groups.isError || app.isError;
  const ready = !loading && !failed;
  return (
    <CustomerBudgetDialog title={t("apps.editSettings", { alias: resource.alias })} pending={pending} onClose={onClose}>
      {loading && (
        <p role="status" className="p-6">
          {t("apps.loading")}
        </p>
      )}
      {!loading && failed && (
        <div className="p-6">
          <ErrorState
            message={extractProxyErrorMessage(groups.error ?? app.error)}
            onRetry={() => void (groups.isError ? groups.refetch() : app.refetch())}
          />
        </div>
      )}
      {ready && groups.data && (
        <AppSettingsForm
          resource={resource}
          app={app.data}
          profiles={groups.data.default_budgets}
          defaultBudgetId={groups.data.default_budget_template_id}
          onClose={onClose}
          onPendingChange={setPending}
        />
      )}
    </CustomerBudgetDialog>
  );
}
