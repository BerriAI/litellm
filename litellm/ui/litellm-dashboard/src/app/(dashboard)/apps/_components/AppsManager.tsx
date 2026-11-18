"use client";

import { useState, type FormEvent } from "react";
import { useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import useAuthorized from "../../hooks/useAuthorized";
import { isProxyAdminRole } from "@/utils/roles";
import { fetchClient } from "@/lib/http/api";
import { extractProxyErrorMessage } from "@/lib/http/client";
import { AdminOnlyNotice } from "@/components/shared/AdminOnlyNotice";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { appBudgetFields, appBudgetValue, type App } from "./appForms";
import { loadApps, appSpend } from "./appApi";
import AppBudgetEditor from "./AppBudgetEditor";
import AppTiers from "./AppTiers";

export default function AppsManager() {
  const { t } = useTranslation();
  const { accessToken, userRole, isLoading } = useAuthorized();
  const canManage = isProxyAdminRole(userRole);
  const apps = useQuery({ queryKey: ["apps"], queryFn: () => loadApps(), enabled: Boolean(accessToken && canManage) });
  const [selected, setSelected] = useState<string | null | undefined>(undefined);
  if (isLoading) return <p role="status">{t("apps.loading")}</p>;
  if (!canManage) return <AdminOnlyNotice pageTitle={t("apps.title")} />;
  const app = apps.data?.find((row) => row.app_id === selected);
  return (
    <main className="mx-auto w-full max-w-[1400px] space-y-6 p-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold">{t("apps.title")}</h1>
        <Button onClick={() => setSelected(null)}>{t("apps.create")}</Button>
      </div>
      {apps.isPending && <p role="status">{t("apps.loading")}</p>}
      {apps.isError && (
        <div role="alert">
          {extractProxyErrorMessage(apps.error)}
          <Button variant="outline" onClick={() => void apps.refetch()}>
            {t("apps.retry")}
          </Button>
        </div>
      )}
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>{t("apps.alias")}</TableHead>
            <TableHead>{t("apps.spend")}</TableHead>
            <TableHead>{t("apps.tiers")}</TableHead>
            <TableHead>{t("apps.actions")}</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {apps.data?.map((row) => (
            <TableRow key={row.app_id}>
              <TableCell>{row.alias}</TableCell>
              <TableCell>{appSpend(row.spend ?? 0)}</TableCell>
              <TableCell>{row.tier_budget_ids?.length ?? 0}</TableCell>
              <TableCell>
                <Button
                  variant="outline"
                  onClick={() => setSelected(row.app_id)}
                  aria-label={t("apps.editApp", { alias: row.alias })}
                >
                  {t("apps.edit")}
                </Button>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
      {apps.data?.length === 0 && <p>{t("apps.empty")}</p>}
      {(selected === null || app) && (
        <AppEditor
          key={app?.app_id ?? "new"}
          app={app}
          onCancel={() => setSelected(undefined)}
          onSaved={async (saved) => {
            await apps.refetch();
            setSelected(saved.app_id);
          }}
          onReload={async () => {
            await apps.refetch();
          }}
        />
      )}
    </main>
  );
}

function AppEditor({
  app,
  onSaved,
  onCancel,
  onReload,
}: {
  app?: App;
  onSaved: (app: App) => Promise<void>;
  onCancel: () => void;
  onReload: () => Promise<void>;
}) {
  const { t } = useTranslation();
  const [alias, setAlias] = useState(app?.alias ?? "");
  const [budget, setBudget] = useState(() => appBudgetFields(app?.budget));
  const [member, setMember] = useState(() => appBudgetFields(app?.default_customer_member_budget));
  const [tier, setTier] = useState(app?.default_tier_budget_id ?? "");
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const defaultTier = app?.tier_budget_ids?.includes(tier) ? tier : "";
  const save = async (event: FormEvent) => {
    event.preventDefault();
    const appBudget = appBudgetValue(budget, app?.budget);
    const memberBudget = appBudgetValue(member, app?.default_customer_member_budget);
    if (!appBudget.ok || !memberBudget.ok) {
      setError(t("customerPortal.hierarchy.invalidNumber"));
      return;
    }
    setPending(true);
    setError(null);
    try {
      const body = {
        alias: alias.trim(),
        budget: appBudget.value,
        default_customer_member_budget: memberBudget.value,
        default_tier_budget_id: defaultTier || null,
      };
      const response = app
        ? await fetchClient.PATCH("/management/v1/apps/{app_id}", { params: { path: { app_id: app.app_id } }, body })
        : await fetchClient.POST("/management/v1/apps", { body });
      if (response.data) await onSaved(response.data);
    } catch (failure) {
      setError(extractProxyErrorMessage(failure));
    } finally {
      setPending(false);
    }
  };
  return (
    <section className="space-y-6 rounded-lg border p-5" aria-label={t(app ? "apps.edit" : "apps.create")}>
      <form onSubmit={save} className="space-y-5">
        <div className="space-y-2">
          <Label htmlFor="app-alias">{t("apps.alias")}</Label>
          <Input id="app-alias" required value={alias} onChange={(event) => setAlias(event.target.value)} />
        </div>
        <div className="grid gap-5 lg:grid-cols-2">
          <AppBudgetEditor title={t("apps.appBudget")} id="app-budget" fields={budget} onChange={setBudget} />
          <AppBudgetEditor title={t("apps.memberDefault")} id="app-member" fields={member} onChange={setMember} />
        </div>
        {app && (
          <div className="space-y-2">
            <Label htmlFor="app-default-tier">{t("apps.defaultTier")}</Label>
            <Select
              value={defaultTier || "__none__"}
              onValueChange={(value) => setTier(value === "__none__" ? "" : value ?? "")}
            >
              <SelectTrigger id="app-default-tier">
                <SelectValue>
                  {app.tiers?.find((row) => row.budget_id === defaultTier)?.name ?? t("apps.none")}
                </SelectValue>
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="__none__">{t("apps.none")}</SelectItem>
                {app.tiers?.map((row) => (
                  <SelectItem key={row.budget_id} value={row.budget_id}>
                    {row.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        )}
        {error && (
          <p role="alert" className="text-destructive">
            {error}
          </p>
        )}
        <div className="flex gap-2">
          <Button type="submit" disabled={pending || !alias.trim()}>
            {t("apps.save")}
          </Button>
          <Button type="button" variant="outline" disabled={pending} onClick={onCancel}>
            {t("apps.close")}
          </Button>
        </div>
      </form>
      {app && <AppTiers app={app} onSaved={onReload} />}
    </section>
  );
}
