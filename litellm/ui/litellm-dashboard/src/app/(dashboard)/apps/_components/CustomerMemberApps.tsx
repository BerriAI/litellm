"use client";

import { useState, type FormEvent } from "react";
import { useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { fetchClient } from "@/lib/http/api";
import { extractProxyErrorMessage } from "@/lib/http/client";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { loadApps, appSpend } from "./appApi";
import { appBudgetFields, appBudgetValue, type App, type AppMember } from "./appForms";
import AppBudgetEditor from "./AppBudgetEditor";

export default function CustomerMemberApps({ memberId }: { memberId: string }) {
  const { t } = useTranslation();
  const apps = useQuery({ queryKey: ["apps"], queryFn: () => loadApps() });
  return (
    <section className="space-y-4 rounded-lg border p-5" aria-label={t("apps.title")}>
      <h2 className="text-lg font-semibold">{t("apps.title")}</h2>
      <p className="text-sm text-muted-foreground">{t("apps.memberHint")}</p>
      {apps.isPending && <p role="status">{t("apps.loading")}</p>}
      {apps.isError && <p role="alert">{extractProxyErrorMessage(apps.error)}</p>}
      {apps.data?.length === 0 && <p>{t("apps.empty")}</p>}
      {apps.data && apps.data.length > 0 && (
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>{t("apps.alias")}</TableHead>
              <TableHead>{t("apps.spend")}</TableHead>
              <TableHead>{t("apps.memberBudget")}</TableHead>
              <TableHead>{t("apps.actions")}</TableHead>
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
        <TableCell>{app.alias}</TableCell>
        <TableCell colSpan={3}>
          <p role="status">{t("apps.loading")}</p>
        </TableCell>
      </TableRow>
    );
  if (member.isError)
    return (
      <TableRow>
        <TableCell>{app.alias}</TableCell>
        <TableCell colSpan={3}>
          <p role="alert">{extractProxyErrorMessage(member.error)}</p>
          <Button variant="outline" onClick={() => void member.refetch()}>
            {t("apps.retry")}
          </Button>
        </TableCell>
      </TableRow>
    );
  return (
    <>
      <TableRow>
        <TableCell>{app.alias}</TableCell>
        <TableCell>{appSpend(member.data?.spend ?? 0)}</TableCell>
        <TableCell>{member.data?.budget_override_id ? t("apps.memberOverride") : t("apps.inheritedBudget")}</TableCell>
        <TableCell>
          <Button
            variant="outline"
            aria-label={t("apps.editMemberBudget", { alias: app.alias })}
            onClick={() => setEditing(!editing)}
          >
            {t("apps.edit")}
          </Button>
        </TableCell>
      </TableRow>
      {editing && member.data && (
        <TableRow>
          <TableCell colSpan={4}>
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
  const [fields, setFields] = useState(() => appBudgetFields(row.budget_override_id ? row.effective_budget : null));
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const save = async (event: FormEvent) => {
    event.preventDefault();
    const budget = appBudgetValue(fields, row.budget_override_id ? row.effective_budget : null);
    if (!budget.ok) {
      setError(t(budget.error));
      return;
    }
    setPending(true);
    setError(null);
    try {
      await fetchClient.PUT("/management/v1/apps/{app_id}/customer_members/{customer_member_id}", {
        params: { path: { app_id: app.app_id, customer_member_id: row.customer_member_id } },
        body: { budget_override: budget.value },
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
