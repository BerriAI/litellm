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
import { CustomerBudgetFormDialog } from "../../customers/_components/CustomerBudgetDialog";
import CustomerBudgetCompactFields from "../../customers/_components/CustomerBudgetCompactFields";
import {
  budgetChanges,
  budgetPolicyChanges,
  compactBudgetEditorFields,
  compactBudgetEditorValue,
  type Budget,
} from "../../customers/_components/customerBudgetForm";
import { useCustomerBudgetSave } from "../../customers/_components/useCustomerBudgetSave";
import { useAppBudgetGroups, useAppMemberBudgetGroups } from "./useAppBudgetGroups";
import { useManagedApps } from "./useManagedApps";
import { applicationRows, grantedApplicationRows, type ApplicationRow } from "./applications";
import { invalidateAppBudgets } from "./appApi";
import type { App } from "./appForms";

const defaultRowClass = "grid items-start gap-4 px-4 py-3 sm:grid-cols-[minmax(12rem,1fr)_minmax(0,2fr)]";
const defaultCopy = {
  organization: ["apps.organizationDefault", "apps.defaultProfileHint"],
  member: ["apps.memberDefault", "apps.memberDefaultsHint"],
} as const;

type DefaultScope = "organization" | "member";
type DefaultValue = { source: string; budget: Budget | null };
type DefaultDraft = { appId: string | null; value: DefaultValue; saved: DefaultValue; error?: string };

function defaultValue(app: App | null, scope: DefaultScope, groups: readonly { budget_id: string }[]): DefaultValue {
  const id = scope === "member" ? app?.default_customer_member_budget_id : app?.default_tier_budget_id;
  const budget = scope === "member" ? app?.default_customer_member_budget ?? null : null;
  const custom = budget !== null && !groups.some((group) => group.budget_id === id);
  if (custom) return { source: "custom", budget };
  return { source: id ? `profile:${id}` : "inherit", budget };
}

function sameDefault(left: DefaultValue, right: DefaultValue) {
  if (left.source !== right.source) return false;
  if (left.source !== "custom") return true;
  return (
    Object.keys(budgetChanges(left.budget ?? {}, right.budget)).length === 0 &&
    Object.keys(budgetChanges(right.budget ?? {}, left.budget)).length === 0
  );
}

function defaultPatch(draft: DefaultDraft, scope: DefaultScope) {
  const id = draft.value.source.startsWith("profile:") ? draft.value.source.slice(8) : null;
  if (scope === "organization") return { default_tier_budget_id: id };
  if (draft.value.source !== "custom") return { default_customer_member_budget_id: id };
  return {
    default_customer_member_budget: {
      ...budgetChanges(draft.value.budget ?? {}, draft.saved.budget),
      ...budgetPolicyChanges(draft.value.budget, draft.saved.budget),
    },
  };
}

function MemberDefaultDialog({
  row,
  budget,
  onApply,
  onClose,
}: {
  row: ApplicationRow;
  budget: Budget | null;
  onApply: (budget: Budget) => void;
  onClose: () => void;
}) {
  const { t } = useTranslation();
  const [fields, setFields] = useState(() => ({ ...compactBudgetEditorFields(budget), budgetEnabled: true }));
  const [error, setError] = useState<string | null>(null);
  const apply = (event: FormEvent) => {
    event.preventDefault();
    const parsed = compactBudgetEditorValue(fields, budget);
    if (!parsed.ok) return setError(t(parsed.error));
    if (parsed.value) onApply(parsed.value);
  };
  return (
    <CustomerBudgetFormDialog
      title={t("apps.memberDefaultFor", { alias: row.resource.alias })}
      description={t("apps.memberDefaultsHint")}
      pending={false}
      onClose={onClose}
      form={{ error, onSubmit: apply, saveLabel: t("apps.applyDefault") }}
    >
      <CustomerBudgetCompactFields
        fields={fields}
        onChange={setFields}
        id="app-member-default"
        modelScope={row.resource}
        disabled={false}
      />
    </CustomerBudgetFormDialog>
  );
}

export default function AppBudgetDefaults({ scope = "organization", app }: { scope?: DefaultScope; app?: App }) {
  const rows = useMemo(() => grantedApplicationRows(app ? [app] : []), [app]);
  return app ? (
    <AppBudgetDefaultRows key={app.app_id} scope={scope} rows={rows} singleApp />
  ) : (
    <ManagedAppBudgetDefaults scope={scope} />
  );
}

function ManagedAppBudgetDefaults({ scope }: { scope: DefaultScope }) {
  const { t } = useTranslation();
  const { definition, resourceType, resources, apps } = useManagedApps();
  const rows = useMemo(() => applicationRows(resources.data ?? [], apps.data ?? [], ""), [resources.data, apps.data]);
  const failed = [definition, resources, apps].find((query) => query.isError);
  if (failed)
    return <ErrorState message={extractProxyErrorMessage(failed.error)} onRetry={() => void failed.refetch()} />;
  if (definition.isPending || (resourceType && (resources.isPending || apps.isPending)))
    return <p role="status">{t("apps.loading")}</p>;
  return <AppBudgetDefaultRows scope={scope} rows={rows} />;
}

function AppBudgetDefaultRows({
  scope,
  rows,
  singleApp = false,
}: {
  scope: DefaultScope;
  rows: ApplicationRow[];
  singleApp?: boolean;
}) {
  const { t } = useTranslation();
  const [title, description] = defaultCopy[scope];
  const queryClient = useQueryClient();
  const [drafts, setDrafts] = useState<Record<string, DefaultDraft>>({});
  const [adding, setAdding] = useState(false);
  const [editing, setEditing] = useState<{ row: ApplicationRow; draft: DefaultDraft } | null>(null);
  const { pending, error, run } = useCustomerBudgetSave();
  const organizationGroups = useAppBudgetGroups(
    rows.map(({ app }) => app?.default_tier_budget_id),
    scope === "organization",
  );
  const memberGroups = useAppMemberBudgetGroups(scope === "member");
  const groups = scope === "member" ? memberGroups : organizationGroups;
  const valueFor = (app: App | null) => defaultValue(app, scope, groups.data?.default_budgets ?? []);
  const [previousRows, setPreviousRows] = useState(rows);
  if (previousRows !== rows) {
    setPreviousRows(rows);
    const remaining = Object.entries(drafts).filter(([id, draft]) => {
      const app = rows.find(({ resource }) => resource.resource_id === id)?.app;
      return !app || !sameDefault(draft.value, draft.saved) || !sameDefault(valueFor(app), draft.saved);
    });
    if (remaining.length !== Object.keys(drafts).length) setDrafts(Object.fromEntries(remaining));
  }
  const available = rows.filter(({ app, resource }) => !app && !drafts[resource.resource_id]);
  const dirty = rows.filter(({ resource }) => {
    const draft = drafts[resource.resource_id];
    return draft && (!draft.appId || !sameDefault(draft.value, draft.saved));
  });
  const draftFor = ({ app, resource }: ApplicationRow): DefaultDraft =>
    drafts[resource.resource_id] ?? {
      appId: resource.app_id ?? null,
      value: valueFor(app),
      saved: valueFor(app),
    };
  const selectGroup = (row: ApplicationRow, source = "inherit") => {
    const draft = draftFor(row);
    if (source === "custom") {
      const selected = groups.data?.default_budgets.find(
        (group) => `profile:${group.budget_id}` === draft.value.source,
      );
      setEditing({
        row,
        draft: {
          ...draft,
          value: {
            source,
            budget:
              draft.value.source === draft.saved.source && draft.value.budget
                ? draft.value.budget
                : selected ?? draft.value.budget,
          },
        },
      });
      return;
    }
    setDrafts((current) => ({
      ...current,
      [row.resource.resource_id]: { ...draft, value: { ...draft.value, source }, error: undefined },
    }));
  };
  const save = async (event: FormEvent) => {
    event.preventDefault();
    if (!dirty.length || !groups.isSuccess) return;
    await run(async () => {
      const results = await Promise.all(
        dirty.map(async ({ resource }) => {
          const id = resource.resource_id;
          const draft = drafts[id];
          const body = defaultPatch(draft, scope);
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
                ? { appId: response.data.app_id, value: valueFor(response.data), saved: valueFor(response.data) }
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
  return (
    <section className="mt-6 space-y-4 border-t pt-6" aria-labelledby={`app-${scope}-default-heading`}>
      <div className="space-y-1">
        <h3 id={`app-${scope}-default-heading`} className="font-semibold">
          {t(title)}
        </h3>
        <p className="text-sm text-muted-foreground">{t(description)}</p>
      </div>
      <form onSubmit={save} className="space-y-4">
        <div className="divide-y rounded-lg border">
          {!singleApp && (
            <div className={`${defaultRowClass} bg-muted/40 text-xs text-muted-foreground`}>
              <span>{t("apps.alias")}</span>
              <span>{t("apps.profile")}</span>
            </div>
          )}
          {rows
            .filter(({ app, resource }) => app || drafts[resource.resource_id])
            .map((row) => {
              const draft = drafts[row.resource.resource_id];
              const value = draft?.value ?? valueFor(row.app);
              const legacyGroup = groups.data?.legacy_budgets.find(
                (group) => `profile:${group.budget_id}` === value.source,
              );
              return (
                <div
                  key={row.resource.resource_id}
                  role="group"
                  aria-label={t("apps.defaultsFor", { alias: row.resource.alias })}
                  className={singleApp ? "p-5" : defaultRowClass}
                >
                  {!singleApp && <span className="py-2 text-sm">{row.resource.alias}</span>}
                  <div className="space-y-2">
                    <CustomerBudgetSourceSelect
                      profiles={groups.data?.default_budgets ?? []}
                      title={t("apps.profile")}
                      value={value.source}
                      onChange={(value) => value && selectGroup(row, value)}
                      disabled={pending || !groups.isSuccess}
                      inheritedOption={{
                        label: t("customerPortal.createDialog.noLimit"),
                        summary: t("apps.noAdditionalLimits"),
                      }}
                      custom={scope === "member"}
                      unavailableSelection={
                        legacyGroup && {
                          value: `profile:${legacyGroup.budget_id}`,
                          label: t("apps.legacyOrganizationGroup", { name: legacyGroup.name }),
                          summary: t("apps.legacyOrganizationGroupHint"),
                        }
                      }
                    />
                    {scope === "member" && value.source === "custom" && (
                      <Button
                        type="button"
                        variant="outline"
                        size="sm"
                        disabled={pending || !groups.isSuccess}
                        onClick={() => selectGroup(row, value.source)}
                      >
                        {t("apps.editMemberDefault")}
                      </Button>
                    )}
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
                    selectGroup(row);
                    setAdding(false);
                  }
                }}
                options={available.map(({ resource }) => ({ value: resource.resource_id, label: resource.alias }))}
              />
            </div>
          )}
          {!singleApp && (
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
          )}
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
      {editing && (
        <MemberDefaultDialog
          row={editing.row}
          budget={editing.draft.value.budget}
          onClose={() => setEditing(null)}
          onApply={(budget) => {
            setDrafts((current) => ({
              ...current,
              [editing.row.resource.resource_id]: {
                ...editing.draft,
                value: { source: "custom", budget },
                error: undefined,
              },
            }));
            setEditing(null);
          }}
        />
      )}
      {!rows.length && <p className="text-sm text-muted-foreground">{t("apps.noResources")}</p>}
    </section>
  );
}
