"use client";

import { useMemo, useState, type FormEvent } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";
import { ErrorState } from "@/components/shared/ErrorState";
import { SearchSelect } from "@/components/shared/SearchSelect";
import { Button } from "@/components/ui/button";
import { fetchClient } from "@/lib/http/api";
import { extractProxyErrorMessage } from "@/lib/http/client";
import { CustomerBudgetSourceSelect } from "../../customers/_components/CustomerLimitsFields";
import { useCustomerBudgetSave } from "../../customers/_components/useCustomerBudgetSave";
import { useAppBudgetGroups } from "./useAppBudgetGroups";
import { useManagedApps } from "./useManagedApps";
import { applicationRows, type ApplicationRow } from "./applications";
import { invalidateAppBudgets } from "./appApi";

const defaultRowClass = "grid items-start gap-4 px-4 py-3 sm:grid-cols-[minmax(12rem,1fr)_minmax(0,2fr)]";

type DefaultDraft = { appId: string | null; profile: string; savedProfile: string; error?: string };

export default function AppOrganizationDefaults() {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const { definition, resourceType, resources, apps } = useManagedApps();
  const [drafts, setDrafts] = useState<Record<string, DefaultDraft>>({});
  const [adding, setAdding] = useState(false);
  const { pending, error, run } = useCustomerBudgetSave();
  const rows = useMemo(() => applicationRows(resources.data ?? [], apps.data ?? [], ""), [resources.data, apps.data]);
  const [previousRows, setPreviousRows] = useState(rows);
  if (previousRows !== rows) {
    setPreviousRows(rows);
    const remaining = Object.entries(drafts).filter(([id, draft]) => {
      const app = rows.find(({ resource }) => resource.resource_id === id)?.app;
      return !app || draft.profile !== draft.savedProfile || (app.default_tier_budget_id ?? "") !== draft.savedProfile;
    });
    if (remaining.length !== Object.keys(drafts).length) setDrafts(Object.fromEntries(remaining));
  }
  const groups = useAppBudgetGroups(rows.map(({ app }) => app?.default_tier_budget_id));
  const available = rows.filter(({ app, resource }) => !app && !drafts[resource.resource_id]);
  const dirty = rows.filter(({ resource }) => {
    const draft = drafts[resource.resource_id];
    return draft && (!draft.appId || draft.profile !== draft.savedProfile);
  });
  const selectGroup = ({ app, resource }: ApplicationRow, profile: string) =>
    setDrafts((current) => ({
      ...current,
      [resource.resource_id]: {
        ...(current[resource.resource_id] ?? {
          appId: resource.app_id ?? null,
          savedProfile: app?.default_tier_budget_id ?? "",
        }),
        profile,
        error: undefined,
      },
    }));
  const save = async (event: FormEvent) => {
    event.preventDefault();
    if (!dirty.length || !groups.isSuccess) return;
    await run(async () => {
      const results = await Promise.all(
        dirty.map(async ({ resource }) => {
          const id = resource.resource_id;
          const draft = drafts[id];
          const body = { default_tier_budget_id: draft.profile || null };
          try {
            const response = draft.appId
              ? await fetchClient.PATCH("/management/v1/apps/{app_id}", {
                  params: { path: { app_id: draft.appId } },
                  body,
                })
              : await fetchClient.POST("/management/v1/apps", {
                  body: { ...body, resource_id: id, resource_type: resource.resource_type },
                });
            return [
              id,
              response.data
                ? { appId: response.data.app_id, profile: draft.profile, savedProfile: draft.profile }
                : { ...draft, error: t("apps.saveFailed") },
            ] as const;
          } catch (cause) {
            return [id, { ...draft, error: extractProxyErrorMessage(cause) }] as const;
          }
        }),
      );
      setDrafts((current) => ({ ...current, ...Object.fromEntries(results) }));
      await invalidateAppBudgets(queryClient);
      if (results.every(([, result]) => !("error" in result))) toast.success(t("apps.saved"));
    });
  };
  const failed = [definition, resources, apps].find((query) => query.isError);
  const resourcesLoading = resourceType && (resources.isPending || apps.isPending);
  if (failed)
    return <ErrorState message={extractProxyErrorMessage(failed.error)} onRetry={() => void failed.refetch()} />;
  if (definition.isPending || resourcesLoading) return <p role="status">{t("apps.loading")}</p>;
  return (
    <section className="mt-6 space-y-4 border-t pt-6" aria-labelledby="app-org-default-heading">
      <div className="space-y-1">
        <h3 id="app-org-default-heading" className="font-semibold">
          {t("apps.organizationDefault")}
        </h3>
        <p className="text-sm text-muted-foreground">{t("apps.defaultProfileHint")}</p>
      </div>
      <form onSubmit={save} className="space-y-4">
        <div className="divide-y rounded-lg border">
          <div className={`${defaultRowClass} bg-muted/40 text-xs text-muted-foreground`}>
            <span>{t("apps.alias")}</span>
            <span>{t("apps.profile")}</span>
          </div>
          {rows
            .filter(({ app, resource }) => app || drafts[resource.resource_id])
            .map((row) => {
              const draft = drafts[row.resource.resource_id];
              const profile = draft?.profile ?? row.app?.default_tier_budget_id ?? "";
              const legacyGroup = groups.data?.legacy_budgets.find((group) => group.budget_id === profile);
              return (
                <div
                  key={row.resource.resource_id}
                  role="group"
                  aria-label={t("apps.defaultsFor", { alias: row.resource.alias })}
                  className={defaultRowClass}
                >
                  <span className="py-2 text-sm">{row.resource.alias}</span>
                  <div className="space-y-2">
                    <CustomerBudgetSourceSelect
                      profiles={groups.data?.default_budgets ?? []}
                      title={t("apps.profile")}
                      value={profile ? `profile:${profile}` : "inherit"}
                      onChange={(value) => selectGroup(row, value?.startsWith("profile:") ? value.slice(8) : "")}
                      disabled={pending || !groups.isSuccess}
                      inheritedOption={{
                        label: t("customerPortal.createDialog.noLimit"),
                        summary: t("apps.noAdditionalLimits"),
                      }}
                      custom={false}
                      unavailableSelection={
                        legacyGroup && {
                          value: `profile:${legacyGroup.budget_id}`,
                          label: t("apps.legacyOrganizationGroup", { name: legacyGroup.name }),
                          summary: t("apps.legacyOrganizationGroupHint"),
                        }
                      }
                    />
                    {draft?.error && (
                      <p role="alert" className="text-sm text-destructive">
                        {draft.error}
                      </p>
                    )}
                  </div>
                </div>
              );
            })}
          {adding && (
            <div className={defaultRowClass}>
              <SearchSelect
                aria-label={t("apps.selectApp")}
                className="w-full"
                placeholder={t("apps.selectApp")}
                value={null}
                disabled={pending}
                onValueChange={(value) => {
                  const row = available.find(({ resource }) => resource.resource_id === value);
                  if (row) {
                    selectGroup(row, "");
                    setAdding(false);
                  }
                }}
                options={available.map(({ resource }) => ({ value: resource.resource_id, label: resource.alias }))}
              />
            </div>
          )}
          <div className="p-2">
            <Button
              type="button"
              variant="ghost"
              size="sm"
              disabled={pending || adding || !available.length}
              onClick={() => setAdding(true)}
            >
              {t("apps.addAppDefault")}
            </Button>
          </div>
        </div>
        {groups.isError && (
          <ErrorState message={extractProxyErrorMessage(groups.error)} onRetry={() => void groups.refetch()} />
        )}
        {error && (
          <p role="alert" className="text-sm text-destructive">
            {error}
          </p>
        )}
        <div className="flex justify-end border-t pt-4">
          <Button type="submit" disabled={pending || !dirty.length || !groups.isSuccess}>
            {t("customerPortal.hierarchy.save")}
          </Button>
        </div>
      </form>
      {!rows.length && <p className="text-sm text-muted-foreground">{t("apps.noResources")}</p>}
    </section>
  );
}
