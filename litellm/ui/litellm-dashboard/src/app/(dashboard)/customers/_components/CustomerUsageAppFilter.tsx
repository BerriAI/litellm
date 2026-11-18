"use client";

import { useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { SearchSelect } from "@/components/shared/SearchSelect";
import { ErrorState } from "@/components/shared/ErrorState";
import { extractProxyErrorMessage } from "@/lib/http/client";
import { appDefinitionKey, appResourcesKey, loadAppDefinition, loadAppResources } from "../../apps/_components/appApi";

export default function CustomerUsageAppFilter({
  value,
  onChange,
}: {
  value: string | null;
  onChange: (value: string | null) => void;
}) {
  const { t } = useTranslation();
  const definition = useQuery({ queryKey: appDefinitionKey, queryFn: loadAppDefinition });
  const resourceType = definition.data?.resource_type ?? null;
  const resources = useQuery({
    queryKey: appResourcesKey(resourceType),
    queryFn: () => loadAppResources(resourceType!),
    enabled: resourceType !== null,
  });
  const error = definition.error ?? resources.error;
  const ready = Boolean(definition.data) && (resourceType === null || Boolean(resources.data));
  const options = [
    { value: "__all__", label: t("apps.allApps") },
    ...(resources.data ?? []).flatMap((resource) =>
      resource.app_id ? [{ value: resource.app_id, label: resource.alias }] : [],
    ),
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
        disabled={!ready}
      />
      {error && (
        <ErrorState
          message={extractProxyErrorMessage(error)}
          onRetry={() => {
            void definition.refetch();
            void resources.refetch();
          }}
        />
      )}
    </div>
  );
}
