"use client";

import { useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { SearchSelect } from "@/components/shared/SearchSelect";
import { ErrorState } from "@/components/shared/ErrorState";
import { extractProxyErrorMessage } from "@/lib/http/client";
import { loadApps } from "../../apps/_components/appApi";

export default function CustomerUsageAppFilter({
  value,
  onChange,
  appGroupId,
}: {
  value: string | null;
  onChange: (value: string | null) => void;
  appGroupId?: string;
}) {
  const { t } = useTranslation();
  const apps = useQuery({ queryKey: ["apps"], queryFn: () => loadApps() });
  const options = [
    { value: "__all__", label: t("apps.allApps") },
    ...(apps.data ?? [])
      .filter((app) => !appGroupId || app.app_group_id === appGroupId)
      .map((app) => ({ value: app.app_id, label: app.alias })),
  ];
  return (
    <div className="w-full space-y-1.5 sm:w-56">
      <label className="block text-xs text-muted-foreground" htmlFor="customer-usage-app">
        {t("apps.filterApp")}
      </label>
      <SearchSelect
        id="customer-usage-app"
        options={options}
        value={value ?? "__all__"}
        onValueChange={(next) => onChange(next === "__all__" ? null : next)}
        allowClear={false}
        disabled={!apps.data}
      />
      {apps.error && <ErrorState message={extractProxyErrorMessage(apps.error)} onRetry={() => void apps.refetch()} />}
    </div>
  );
}
