"use client";

import { useState, type FormEvent } from "react";
import { useWatch } from "react-hook-form";
import { useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { fetchClient } from "@/lib/http/api";
import { extractProxyErrorMessage } from "@/lib/http/client";
import { useZodForm } from "@/lib/forms/useZodForm";
import { Button } from "@/components/ui/button";
import { TableCell, TableRow } from "@/components/ui/table";
import { loadApps, loadOrganizationApp, organizationAppKey } from "./appApi";
import { appBaseBudget, type App, type AppMember } from "./appForms";
import { CustomerBudgetEditor } from "../../customers/_components/CustomerLimitsFields";
import { CustomerBudgetDialog, CustomerBudgetFormDialog } from "../../customers/_components/CustomerBudgetDialog";
import { useCustomerBudgetSave } from "../../customers/_components/useCustomerBudgetSave";
import { hasLegacyCustomerBudget, type Budget } from "../../customers/_components/customerBudgetForm";
import { ErrorState } from "@/components/shared/ErrorState";
import { DataTablePagination, useUrlTableState } from "@/components/shared/DataTable";
import { AppBudgetRow, AppBudgetTable, type AppUsageContext } from "./AppBudgetCells";
import { memberAppDraft } from "./memberAppMapper";
import { createMemberAppSchema } from "./memberAppSchema";

const MEMBER_APPS_TABLE = {
  keyPrefix: "member_apps_",
  sortFields: [],
  defaultSort: { id: "", desc: false },
  defaultPageSize: 5,
  filterColumns: [],
};

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
  const { pagination, onPaginationChange } = useUrlTableState(MEMBER_APPS_TABLE);
  const { pageSize, pageIndex } = pagination;
  const page = Math.min(pageIndex, Math.max(0, Math.ceil((apps.data?.length ?? 0) / pageSize) - 1));
  return (
    <section className="space-y-4 border-t pt-6" aria-label={t("apps.memberTitle")}>
      <h2 className="text-base font-semibold">{t("apps.memberTitle")}</h2>
      <p className="text-sm text-muted-foreground">{t("apps.memberHint")}</p>
      {apps.isPending && <p role="status">{t("apps.loading")}</p>}
      {apps.isError && <p role="alert">{extractProxyErrorMessage(apps.error)}</p>}
      {apps.data?.length === 0 && <p>{t("apps.noMemberApps")}</p>}
      {apps.data && apps.data.length > 0 && (
        <AppBudgetTable
          scope="member"
          usageLayout={Boolean(usageContext)}
          footer={
            <DataTablePagination
              className="justify-end border-t px-4 py-3"
              page={page}
              pageSize={pageSize}
              pageSizeOptions={[5, 10, 25]}
              rowCount={apps.data.length}
              onPageChange={(pageIndex) => onPaginationChange({ pageIndex, pageSize })}
              onPageSizeChange={(pageSize) => onPaginationChange({ pageIndex: 0, pageSize })}
            />
          }
        >
          {apps.data.slice(page * pageSize, (page + 1) * pageSize).map((app) => (
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
  if (needsInherited && (inherited.isPending || inherited.isError))
    return (
      <CustomerBudgetDialog
        title={t("apps.editMemberBudget", { alias: props.app.alias })}
        description={t("apps.memberOverrideHint")}
        pending={false}
        onClose={props.onCancel}
      >
        {inherited.isPending ? (
          <p role="status">{t("apps.loading")}</p>
        ) : (
          <ErrorState onRetry={() => void inherited.refetch()} />
        )}
      </CustomerBudgetDialog>
    );
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
  const schema = createMemberAppSchema(original, row, t);
  const form = useZodForm(schema, { defaultValues: { draft: memberAppDraft(row, original) } });
  const fields = useWatch({ control: form.control, name: "draft" });
  const { pending, error, setError, run } = useCustomerBudgetSave();
  const save = async (event: FormEvent) => {
    event.preventDefault();
    if (pending) return;
    const result = schema.safeParse({ draft: fields });
    if (!result.success) return setError(result.error.issues[0].message);
    await run(async () => {
      await fetchClient.PUT("/management/v1/apps/{app_id}/customer_members/{customer_member_id}", {
        params: { path: { app_id: app.app_id, customer_member_id: row.customer_member_id } },
        body: result.data,
      });
      await onSaved();
    });
  };
  const changeSource = (value: string | null) => {
    form.setValue("draft", { ...form.getValues("draft"), budgetEnabled: value === "custom" });
    setError(null);
  };
  return (
    <CustomerBudgetFormDialog
      title={t("apps.editMemberBudget", { alias: app.alias })}
      description={t("apps.memberOverrideHint")}
      pending={pending}
      onClose={onCancel}
      form={{ error, onSubmit: save, saveLabel: t("apps.saveMemberBudget") }}
    >
      <CustomerBudgetEditor
        id={`app-member-${app.app_id}`}
        budget={{ modelScope: app, fields, disabled: pending, onChange: (value) => form.setValue("draft", value) }}
        source={{
          profiles: [],
          value: fields.budgetEnabled ? "custom" : "inherit",
          title: t("apps.source"),
          disabled: pending,
          inheritedOption: { label: t("apps.inheritedBudget"), summary: t("apps.inheritMemberHint") },
          onChange: changeSource,
        }}
      />
    </CustomerBudgetFormDialog>
  );
}
