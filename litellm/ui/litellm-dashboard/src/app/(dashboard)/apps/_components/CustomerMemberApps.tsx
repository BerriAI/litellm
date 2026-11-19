"use client";

import { useState, type FormEvent } from "react";
import { useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { fetchClient } from "@/lib/http/api";
import { extractProxyErrorMessage } from "@/lib/http/client";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { loadApps } from "./appApi";
import { appBudgetFields, appBudgetValue, type App, type AppMember } from "./appForms";
import AppBudgetEditor from "./AppBudgetEditor";
import AppBudgetLimits from "./AppBudgetLimits";
import { appBudgetWindows } from "./appBudgetPresentation";
import { customerBudgetPeriod } from "../../customers/_components/customerBudgetUsage";

const BUDGET_SOURCES = {
  member_override: "apps.memberOverride",
  customer_default: "apps.inheritedCustomer",
  app_default: "apps.inheritedApp",
  none: "apps.noInheritedBudget",
} as const;

export default function CustomerMemberApps({ memberId }: { memberId: string }) {
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
        <Table className="w-full text-left text-sm" aria-label={t("apps.memberTitle")}>
          <TableHeader>
            <TableRow>
              <TableHead className="border-b py-3 text-xs font-medium text-muted-foreground">
                {t("apps.alias")}
              </TableHead>
              <TableHead className="border-b py-3 text-xs font-medium text-muted-foreground">
                {t("apps.memberBudget")}
              </TableHead>
              <TableHead className="border-b py-3 text-xs font-medium text-muted-foreground">
                {t("apps.actions")}
              </TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {apps.data.map((app) => (
              <MemberAppRow key={app.app_id} app={app} memberId={memberId} />
            ))}
          </TableBody>
        </Table>
      )}
    </section>
  );
}

function MemberAppRow({ app, memberId }: { app: App; memberId: string }) {
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
  if (member.isPending)
    return (
      <TableRow>
        <TableCell className="py-4 font-medium">{app.alias}</TableCell>
        <TableCell colSpan={2}>
          <p role="status">{t("apps.loading")}</p>
        </TableCell>
      </TableRow>
    );
  if (member.isError)
    return (
      <TableRow>
        <TableCell className="py-4 font-medium">{app.alias}</TableCell>
        <TableCell colSpan={2}>
          <p role="alert">{extractProxyErrorMessage(member.error)}</p>
          <Button variant="ghost" size="xs" onClick={() => void member.refetch()}>
            {t("apps.retry")}
          </Button>
        </TableCell>
      </TableRow>
    );
  const { primary } = appBudgetWindows(member.data?.effective_budget, member.data?.budget_usage);
  const period = customerBudgetPeriod(primary?.budget_duration ?? null);
  const periodLabel = period === "other" ? primary?.budget_duration : t(`apps.${period}`);
  const sourceLabel = t(BUDGET_SOURCES[member.data?.budget_source ?? "none"]);
  return (
    <>
      <TableRow>
        <TableCell className="py-4">
          <span className="font-medium">{app.alias}</span>
          <span className="mt-1 block text-xs text-muted-foreground">
            {[sourceLabel, periodLabel].filter(Boolean).join(" · ")}
          </span>
        </TableCell>
        <TableCell className="py-4">
          <AppBudgetLimits
            budget={member.data?.effective_budget}
            usage={member.data?.budget_usage}
            cumulativeSpend={member.data?.spend}
          />
        </TableCell>
        <TableCell>
          <Button
            variant="ghost"
            size="xs"
            aria-label={t("apps.editMemberBudget", { alias: app.alias })}
            onClick={() => setEditing(!editing)}
          >
            {t("apps.edit")}
          </Button>
        </TableCell>
      </TableRow>
      {editing && member.data && (
        <TableRow>
          <TableCell colSpan={3}>
            <MemberBudgetForm
              app={app}
              row={member.data}
              onCancel={() => setEditing(false)}
              onSaved={async () => {
                await member.refetch();
                setEditing(false);
              }}
            />
          </TableCell>
        </TableRow>
      )}
    </>
  );
}

function MemberBudgetForm({
  app,
  row,
  onSaved,
  onCancel,
}: {
  app: App;
  row: AppMember;
  onSaved: () => Promise<void>;
  onCancel: () => void;
}) {
  const { t } = useTranslation();
  const originalBudget = row.budget ?? row.budget_override;
  const [fields, setFields] = useState(() => appBudgetFields(originalBudget));
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const save = async (event: FormEvent) => {
    event.preventDefault();
    const budget = appBudgetValue(fields, originalBudget);
    if (!budget.ok) {
      setError(t(budget.error));
      return;
    }
    setPending(true);
    setError(null);
    try {
      await fetchClient.PUT("/management/v1/apps/{app_id}/customer_members/{customer_member_id}", {
        params: { path: { app_id: app.app_id, customer_member_id: row.customer_member_id } },
        body: { budget: budget.value, ...(row.budget_override != null ? { budget_override: null } : {}) },
      });
      await onSaved();
    } catch (failure) {
      setError(extractProxyErrorMessage(failure));
    } finally {
      setPending(false);
    }
  };
  return (
    <form onSubmit={save} className="space-y-4 py-4" aria-label={t("apps.editMemberBudget", { alias: app.alias })}>
      <AppBudgetEditor
        title={t("apps.memberOverride")}
        id={`app-member-${app.app_id}`}
        fields={fields}
        onChange={setFields}
      />
      <p className="text-sm text-muted-foreground">{t("apps.memberOverrideHint")}</p>
      {error && (
        <p role="alert" className="text-destructive">
          {error}
        </p>
      )}
      <div className="flex gap-2">
        <Button type="submit" disabled={pending}>
          {t("apps.saveMemberBudget")}
        </Button>
        <Button type="button" variant="outline" disabled={pending} onClick={onCancel}>
          {t("apps.cancel")}
        </Button>
      </div>
    </form>
  );
}
