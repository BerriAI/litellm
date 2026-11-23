"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import { SearchSelect } from "@/components/shared/SearchSelect";
import { ErrorState } from "@/components/shared/ErrorState";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { fetchClient } from "@/lib/http/api";
import { extractProxyErrorMessage } from "@/lib/http/client";
import { appDefinitionKey, appResourcesKey, loadAppDefinition, loadAppResources } from "./appApi";
import { APP_RESOURCE_LABELS, type AppResourceType } from "./appForms";

export default function AppDefinitionSetting() {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const definition = useQuery({ queryKey: appDefinitionKey, queryFn: loadAppDefinition });
  const [draft, setDraft] = useState<AppResourceType | null>(null);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const selected = draft ?? definition.data?.resource_type ?? null;
  const resources = useQuery({
    queryKey: appResourcesKey(selected),
    queryFn: () => loadAppResources(selected!),
    enabled: selected !== null,
  });
  const unchanged = selected === definition.data?.resource_type;
  const resourcesUnavailable = resources.isPending || resources.isError;
  const saveUnavailable = pending || !selected || unchanged;
  const selectResourceType = (value: string | null) => {
    const resourceType = (Object.keys(APP_RESOURCE_LABELS) as AppResourceType[]).find((type) => type === value);
    if (resourceType) setDraft(resourceType);
    setError(null);
  };
  const save = async () => {
    if (!selected) return;
    setPending(true);
    setError(null);
    try {
      const response = await fetchClient.PUT("/management/v1/apps/definition", {
        body: {
          resource_type: selected,
          auto_create_customers_without_app: definition.data?.auto_create_customers_without_app ?? true,
        },
      });
      if (!response.data) throw new Error(t("apps.saveFailed"));
      queryClient.setQueryData(appDefinitionKey, response.data);
      await queryClient.invalidateQueries({ queryKey: ["app-resources"] });
      await queryClient.invalidateQueries({ queryKey: ["apps"] });
      setDraft(null);
    } catch (failure) {
      setError(extractProxyErrorMessage(failure));
    } finally {
      setPending(false);
    }
  };
  const saveWithoutApp = async (allowed: boolean) => {
    setPending(true);
    setError(null);
    try {
      const response = await fetchClient.PUT("/management/v1/apps/definition", {
        body: { resource_type: definition.data?.resource_type ?? null, auto_create_customers_without_app: allowed },
      });
      if (!response.data) throw new Error(t("apps.saveFailed"));
      queryClient.setQueryData(appDefinitionKey, response.data);
    } catch (failure) {
      setError(extractProxyErrorMessage(failure));
    } finally {
      setPending(false);
    }
  };
  return (
    <section className="space-y-5" aria-label={t("apps.definition.title")}>
      <div>
        <h2 className="text-base font-semibold">{t("apps.definition.title")}</h2>
        <p className="mt-1 text-sm text-muted-foreground">{t("apps.definition.hint")}</p>
      </div>
      {definition.isPending && <p role="status">{t("apps.loading")}</p>}
      {definition.isError && (
        <ErrorState message={extractProxyErrorMessage(definition.error)} onRetry={() => void definition.refetch()} />
      )}
      {definition.data && (
        <>
          <div className="max-w-md space-y-2">
            <label htmlFor="gateway-app-resource" className="block text-sm font-medium">
              {t("apps.definition.label")}
            </label>
            <SearchSelect
              id="gateway-app-resource"
              className="w-full"
              placeholder={t("apps.definition.choose")}
              value={selected ?? undefined}
              allowClear={false}
              onValueChange={selectResourceType}
              options={(Object.keys(APP_RESOURCE_LABELS) as AppResourceType[]).map((type) => ({
                value: type,
                label: t(APP_RESOURCE_LABELS[type]),
              }))}
              disabled={pending}
            />
          </div>
          {selected && (
            <div className="space-y-3 border-l-2 pl-4">
              {resources.isPending && <p role="status">{t("apps.definition.discovering")}</p>}
              {resources.isError && (
                <p role="alert" className="text-destructive">
                  {extractProxyErrorMessage(resources.error)}
                </p>
              )}
              {resources.data && (
                <>
                  <p className="text-sm">
                    {t("apps.definition.discovered", {
                      count: resources.data.length,
                      type: t(APP_RESOURCE_LABELS[selected]),
                    })}
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {resources.data.slice(0, 8).map((resource) => (
                      <Badge key={resource.resource_id} variant="secondary">
                        {resource.alias}
                      </Badge>
                    ))}
                    {resources.data.length > 8 && (
                      <span className="text-xs text-muted-foreground">
                        {t("apps.definition.more", { count: resources.data.length - 8 })}
                      </span>
                    )}
                  </div>
                </>
              )}
            </div>
          )}
          <div className="flex items-center gap-3">
            <Button disabled={saveUnavailable || resourcesUnavailable} onClick={() => void save()}>
              {t("apps.definition.save")}
            </Button>
            {definition.data.resource_type && (
              <span className="text-xs text-muted-foreground">
                {t("apps.definition.configured", { type: t(APP_RESOURCE_LABELS[definition.data.resource_type]) })}
              </span>
            )}
          </div>
          <p className="text-xs text-muted-foreground">{t("apps.definition.lockHint")}</p>
          <CustomersWithoutAppSwitch
            checked={definition.data.auto_create_customers_without_app}
            disabled={pending}
            onChange={(allowed) => void saveWithoutApp(allowed)}
          />
        </>
      )}
      {error && (
        <p role="alert" className="text-sm text-destructive">
          {error}
        </p>
      )}
    </section>
  );
}

function CustomersWithoutAppSwitch({
  checked,
  disabled,
  onChange,
}: {
  checked?: boolean;
  disabled: boolean;
  onChange: (allowed: boolean) => void;
}) {
  const { t } = useTranslation();
  return (
    <div className="space-y-1.5 border-t pt-4">
      <label className="flex items-center gap-2 text-sm font-medium">
        <Checkbox checked={checked ?? true} disabled={disabled} onCheckedChange={(value) => onChange(value === true)} />
        {t("apps.definition.autoCreateWithoutApp")}
      </label>
      <p className="text-xs text-muted-foreground">{t("apps.definition.autoCreateWithoutAppHint")}</p>
    </div>
  );
}
