"use client";

import { useState, type FormEvent } from "react";
import { useTranslation } from "react-i18next";
import { fetchClient } from "@/lib/http/api";
import { extractProxyErrorMessage } from "@/lib/http/client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Table, TableHeader, TableHead, TableBody, TableRow, TableCell } from "@/components/ui/table";
import type { components } from "@/lib/http/schema";
import { appBudgetFields, appTierBudgetValue, type App } from "./appForms";
import AppBudgetEditor from "./AppBudgetEditor";

type Tier = components["schemas"]["AppTier"];

export default function AppTiers({ app, onSaved }: { app: App; onSaved: () => Promise<void> }) {
  const { t } = useTranslation();
  const [draft, setDraft] = useState<Tier | null | undefined>(undefined);
  const [deleting, setDeleting] = useState<Tier | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  const remove = async () => {
    if (!deleting) return;
    setPending(true);
    setError(null);
    try {
      await fetchClient.DELETE("/management/v1/apps/{app_id}/tiers", {
        params: { path: { app_id: app.app_id } },
        body: { budget_id: deleting.budget_id },
      });
      await onSaved();
      setDeleting(null);
    } catch (failure) {
      setError(extractProxyErrorMessage(failure));
    } finally {
      setPending(false);
    }
  };
  return (
    <section className="space-y-4" aria-label={t("apps.tiers")}>
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">{t("apps.tiers")}</h2>
        <Button
          type="button"
          onClick={() => {
            setDraft(null);
            setError(null);
          }}
        >
          {t("apps.addTier")}
        </Button>
      </div>
      {error && (
        <p role="alert" className="text-destructive">
          {error}
        </p>
      )}
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>{t("apps.name")}</TableHead>
            <TableHead>{t("apps.maxBudget")}</TableHead>
            <TableHead>{t("apps.tpm")}</TableHead>
            <TableHead>{t("apps.rpm")}</TableHead>
            <TableHead>{t("apps.duration")}</TableHead>
            <TableHead>{t("apps.actions")}</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {app.tiers?.map((tier) => (
            <TableRow key={tier.budget_id}>
              <TableCell>{tier.name}</TableCell>
              <TableCell>{tier.max_budget ?? t("apps.unlimited")}</TableCell>
              <TableCell>{tier.tpm_limit ?? t("apps.unlimited")}</TableCell>
              <TableCell>{tier.rpm_limit ?? t("apps.unlimited")}</TableCell>
              <TableCell>{tier.budget_duration ?? t("apps.none")}</TableCell>
              <TableCell className="space-x-2">
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => {
                    setDraft(tier);
                    setError(null);
                  }}
                >
                  {t("apps.edit")}
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => {
                    setDeleting(tier);
                    setError(null);
                  }}
                >
                  {t("apps.delete")}
                </Button>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
      {!app.tiers?.length && <p>{t("apps.noTiers")}</p>}
      {deleting && (
        <div role="group" aria-label={t("apps.confirmDelete")} className="flex items-center gap-3 rounded border p-3">
          <p>{t("apps.deleteTierPrompt", { name: deleting.name })}</p>
          <Button type="button" disabled={pending} onClick={() => void remove()}>
            {t("apps.confirmDelete")}
          </Button>
          <Button type="button" variant="outline" disabled={pending} onClick={() => setDeleting(null)}>
            {t("apps.cancel")}
          </Button>
        </div>
      )}
      {draft !== undefined && (
        <TierForm
          key={draft?.budget_id ?? "new"}
          app={app}
          tier={draft}
          onCancel={() => setDraft(undefined)}
          onSaved={async () => {
            await onSaved();
            setDraft(undefined);
          }}
        />
      )}
    </section>
  );
}

function TierForm({
  app,
  tier,
  onSaved,
  onCancel,
}: {
  app: App;
  tier: Tier | null;
  onSaved: () => Promise<void>;
  onCancel: () => void;
}) {
  const { t } = useTranslation();
  const [name, setName] = useState(tier?.name ?? "");
  const [fields, setFields] = useState(() => appBudgetFields(tier ?? {}));
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  const save = async (event: FormEvent) => {
    event.preventDefault();
    const budget = appTierBudgetValue(fields, tier);
    if (!budget.ok) {
      setError(t(budget.error));
      return;
    }
    setPending(true);
    setError(null);
    try {
      const body = { ...budget.value, name: name.trim() };
      const params = { path: { app_id: app.app_id } };
      if (tier)
        await fetchClient.PATCH("/management/v1/apps/{app_id}/tiers", {
          params,
          body: { ...body, budget_id: tier.budget_id },
        });
      else await fetchClient.POST("/management/v1/apps/{app_id}/tiers", { params, body });
      await onSaved();
    } catch (failure) {
      setError(extractProxyErrorMessage(failure));
    } finally {
      setPending(false);
    }
  };
  return (
    <form onSubmit={save} className="space-y-4 rounded-lg border p-4" aria-label={t("apps.tierForm")}>
      <div className="space-y-2">
        <Label htmlFor="tier-name">{t("apps.name")}</Label>
        <Input id="tier-name" required value={name} onChange={(event) => setName(event.target.value)} />
      </div>
      <AppBudgetEditor title={t("apps.tierBudget")} id="tier-budget" fields={fields} onChange={setFields} />
      {error && (
        <p role="alert" className="text-destructive">
          {error}
        </p>
      )}
      <div className="flex gap-2">
        <Button type="submit" disabled={pending || !name.trim()}>
          {t("apps.saveTier")}
        </Button>
        <Button type="button" variant="outline" disabled={pending} onClick={onCancel}>
          {t("apps.cancel")}
        </Button>
      </div>
    </form>
  );
}
