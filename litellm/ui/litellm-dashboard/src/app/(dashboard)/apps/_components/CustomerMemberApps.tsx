"use client";

import { compactBudgetEditorFields, type Budget } from "../../customers/_components/customerBudgetForm";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { fetchClient } from "@/lib/http/api";
import { extractProxyErrorMessage } from "@/lib/http/client";
import { Button } from "@/components/ui/button";
import { TableCell, TableRow } from "@/components/ui/table";
import { loadApps, loadOrganizationApp, organizationAppKey } from "./appApi";
import { appBaseBudget, type App, type AppMember } from "./appForms";
import CustomerBudgetCompactFields from "../../customers/_components/CustomerBudgetCompactFields";
import { CustomerBudgetEditor } from "../../customers/_components/CustomerLimitsFields";
import { CustomerBudgetDialog, CustomerBudgetForm } from "../../customers/_components/CustomerBudgetDialog";
import { useCustomerBudgetForm } from "../../customers/_components/useCustomerBudgetForm";
import { customerBudgetPatch, hasLegacyCustomerBudget } from "../../customers/_components/customerBudgetForm";
import { ErrorState } from "@/components/shared/ErrorState";
import { AppBudgetRow, AppBudgetTable, type AppUsageContext } from "./AppBudgetCells";

const BUDGET_SOURCES = {
  member_budget: "apps.customBudget",
  member_override: "apps.memberOverride",
  customer_default: "apps.inheritedCustomer",
  app_default: "apps.inheritedApp",
  none: "apps.noInheritedBudget",
} as const;

export default function CustomerMemberApps({
  memberId,
  organizationId,
  usageContext,
  onAppSelect,
}: {
  memberId: string;
  organizationId: string;
  usageContext?: AppUsageContext;
  onAppSelect?: (appId: string) => void;
}) {
  const { t } = useTranslation();
  const apps = useQuery({ queryKey: ["apps"], queryFn: () => loadApps() });
  return (
    <section className="space-y-4 border-t pt-6" aria-label={t("apps.memberTitle")}>
      <h2 className="text-base font-semibold">{t("apps.memberTitle")}</h2>
      <p className="text-sm text-muted-foreground">{t("apps.memberHint")}</p>
      {apps.isPending && <p role="status">{t("apps.loading")}</p>}
      {apps.isError && <p role="alert">{extractProxyErrorMessage(apps.error)}</p>}
      {apps.data?.length === 0 && <p>{t("apps.noMemberApps")}</p>}
      {apps.data && apps.data.length > 0 && (
        <AppBudgetTable scope="member" usageLayout={Boolean(usageContext)}>
          {apps.data.map((app) => (
            <MemberAppRow
              key={app.app_id}
              app={app}
              memberId={memberId}
              organizationId={organizationId}
              usageContext={usageContext}
              onAppSelect={onAppSelect}
            />
          ))}
        </AppBudgetTable>
      )}
    </section>
  );
}

function MemberAppRow({
  app,
  memberId,
  organizationId,
  usageContext,
  onAppSelect,
}: {
  app: App;
  memberId: string;
  organizationId: string;
  usageContext?: AppUsageContext;
  onAppSelect?: (appId: string) => void;
}) {
  const { t } = useTranslation();
  const [editing, setEditing] = useState(false);
  const member = useQuery({
    queryKey: ["app-member", app.app_id, memberId],
    queryFn: async () => {
      const response = await fetchClient.GET("/management/v1/apps/{app_id}/customer_members/{customer_member_id}", {
        params: { path: { app_id: app.app_id, customer_member_id: memberId } },
      });
      return response.data;
    },
  });
  if (member.isPending || member.isError)
    return (
      <TableRow>
        <TableCell className="py-4 font-medium">{app.alias}</TableCell>
        <TableCell colSpan={usageContext ? 8 : 2}>
          {member.isPending ? (
            <p role="status">{t("apps.loading")}</p>
          ) : (
            <>
              <p role="alert">{extractProxyErrorMessage(member.error)}</p>
              <Button variant="ghost" size="xs" onClick={() => void member.refetch()}>
                {t("apps.retry")}
              </Button>
            </>
          )}
        </TableCell>
      </TableRow>
    );
  const sourceLabel = t(BUDGET_SOURCES[member.data?.budget_source ?? "none"]);
  return (
    <>
      <AppBudgetRow
        scope="member"
        app={app}
        row={member.data}
        sourceLabel={sourceLabel}
        metrics={usageContext ? { layout: "full", context: usageContext } : undefined}
        onAppSelect={onAppSelect}
        onEdit={() => setEditing(!editing)}
      />
      {editing && member.data && (
        <MemberBudgetLoader
          organizationId={organizationId}
          app={app}
          row={member.data}
          onCancel={() => setEditing(false)}
          onSaved={async () => {
            await member.refetch();
            setEditing(false);
          }}
        />
      )}
    </>
  );
}

function MemberBudgetLoader(props: {
  app: App;
  row: AppMember;
  organizationId: string;
  onSaved: () => Promise<void>;
  onCancel: () => void;
}) {
  const { t } = useTranslation();
  const needsInherited = !props.row.budget && !hasLegacyCustomerBudget(props.row);
  const inherited = useQuery({
    queryKey: organizationAppKey(props.app.app_id, props.organizationId),
    queryFn: () => loadOrganizationApp(props.app.app_id, props.organizationId),
    enabled: needsInherited,
  });
  if (needsInherited && inherited.isPending) return <p role="status">{t("apps.loading")}</p>;
  if (needsInherited && inherited.isError) return <ErrorState onRetry={() => void inherited.refetch()} />;
  const original = appBaseBudget(
    props.row,
    inherited.data?.effective_customer_member_budget ?? props.app.default_customer_member_budget,
  );
  return <MemberBudgetForm {...props} original={original} />;
}

function MemberBudgetForm({
  app,
  row,
  original: loadedOriginal,
  onSaved,
  onCancel,
}: {
  app: App;
  row: AppMember;
  original: Budget | null;
  onSaved: () => Promise<void>;
  onCancel: () => void;
}) {
  const { t } = useTranslation();
  const [original] = useState(loadedOriginal);
  const form = useCustomerBudgetForm({
    initial: {
      ...compactBudgetEditorFields(original),
      budgetEnabled: Boolean(row.budget) || hasLegacyCustomerBudget(row),
    },
    parse: (fields) => {
      const result = customerBudgetPatch(fields, original, row);
      return result.ok ? result : { ok: false, error: t(result.error) };
    },
    write: async (body) => {
      await fetchClient.PUT("/management/v1/apps/{app_id}/customer_members/{customer_member_id}", {
        params: { path: { app_id: app.app_id, customer_member_id: row.customer_member_id } },
        body,
      });
      await onSaved();
    },
  });
  const { fields, setFields, update, pending, error, save } = form;
  return (
    <CustomerBudgetDialog
      title={t("apps.editMemberBudget", { alias: app.alias })}
      description={t("apps.memberOverrideHint")}
      pending={pending}
      onClose={onCancel}
    >
      <CustomerBudgetForm
        pending={pending}
        error={error}
        onSubmit={save}
        onClose={onCancel}
        saveLabel={t("apps.saveMemberBudget")}
      >
        <CustomerBudgetEditor
          id={`app-member-${app.app_id}`}
          source={{
            profiles: [],
            value: fields.budgetEnabled ? "custom" : "inherit",
            title: t("apps.source"),
            disabled: pending,
            inheritedOption: { label: t("apps.inheritedBudget"), summary: t("apps.inheritMemberHint") },
            onChange: (value) => update("budgetEnabled", value === "custom"),
          }}
        >
          <CustomerBudgetCompactFields
            id={`app-member-${app.app_id}`}
            fields={fields}
            original={original}
            disabled={pending}
            onChange={setFields}
          />
        </CustomerBudgetEditor>
      </CustomerBudgetForm>
    </CustomerBudgetDialog>
  );
}
