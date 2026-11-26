"use client";

import { useState, type FormEvent } from "react";
import { useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { useControlPlane } from "@/contexts/ControlPlaneContext";
import type { ControlPlaneApi } from "@/lib/http/controlPlane";
import { Page, PageContent } from "@/components/shared/Page";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ControlPlaneSectionHeader, ErrorNotice, ResultsTable } from "./ControlPlaneShared";
import { ControlPlaneAdmin } from "./ControlPlaneAdmin";

function Coverage({
  result,
}: {
  result: { partial: boolean; unreachable: { id: string; reason: string }[]; truncated_dataplanes?: string[] };
}) {
  const { t } = useTranslation("translation", { keyPrefix: "controlPlane" });
  if (!result.partial && !result.unreachable.length && !result.truncated_dataplanes?.length) return null;
  return (
    <div role="status" className="space-y-2 rounded border border-amber-500 p-3">
      {result.partial && <p>{t("partialResultsTotalsDoNotCoverEveryDataplane")}</p>}
      {result.unreachable.length > 0 && (
        <ul>
          {result.unreachable.map(({ id, reason }) => (
            <li key={id}>
              {t("unreachableLabel")} {id} ({reason})
            </li>
          ))}
        </ul>
      )}
      {!!result.truncated_dataplanes?.length && (
        <p>
          {t("truncatedLabel")} {result.truncated_dataplanes.join(", ")}
        </p>
      )}
    </div>
  );
}

function Fleet({ api, section }: { api: ControlPlaneApi; section: "fleet-models" | "fleet-teams" }) {
  const { t } = useTranslation("translation", { keyPrefix: "controlPlane" });
  const models = useQuery({ queryKey: ["fleet", "models"], queryFn: api.models, enabled: section === "fleet-models" });
  const teams = useQuery({ queryKey: ["fleet", "teams"], queryFn: api.teams, enabled: section === "fleet-teams" });
  const result = section === "fleet-models" ? models : teams;
  return (
    <>
      <ControlPlaneSectionHeader section={section} />
      <ErrorNotice error={result.error} />
      {result.isPending && <p role="status">{t("loadingFleet")}</p>}
      {result.data && <Coverage result={result.data} />}
      {section === "fleet-models" && models.data && (
        <ResultsTable
          onRefresh={() => void models.refetch()}
          isFetching={models.isFetching}
          columns={[t("model"), t("dataplane"), t("provider"), t("mode")]}
          rows={models.data.items.map((row) => [row.model_name, row.dataplane_id, row.provider, row.mode])}
        />
      )}
      {section === "fleet-teams" && teams.data && (
        <ResultsTable
          onRefresh={() => void teams.refetch()}
          isFetching={teams.isFetching}
          columns={[t("team"), t("dataplane"), t("members"), t("spend")]}
          rows={teams.data.items.map((row) => [
            row.team_alias ?? row.team_id,
            row.dataplane_id,
            row.members,
            row.spend,
          ])}
        />
      )}
    </>
  );
}

const usageViews = ["summary", "daily", "top"] as const;

function Usage({ api }: { api: ControlPlaneApi }) {
  const { t } = useTranslation("translation", { keyPrefix: "controlPlane" });
  const [view, setView] = useState<"summary" | "daily" | "top">("summary");
  const [dates, setDates] = useState({ start: "", end: "" });
  const search = new URLSearchParams();
  if (dates.start) search.set("start_date", dates.start);
  if (dates.end) search.set("end_date", dates.end);
  const summary = useQuery({
    queryKey: ["usage", "summary", dates],
    queryFn: () => api.usageSummary(search),
    enabled: view === "summary",
  });
  const daily = useQuery({
    queryKey: ["usage", "daily", dates],
    queryFn: () => api.usageDaily(search),
    enabled: view === "daily",
  });
  const top = useQuery({
    queryKey: ["usage", "top", dates],
    queryFn: () => api.usageTop(search),
    enabled: view === "top",
  });
  const result = { summary, daily, top }[view];
  const tableRefresh = { onRefresh: () => void result.refetch(), isFetching: result.isFetching };
  const metrics = (row: { spend: number; api_requests: number; total_tokens: number }) => [
    row.spend,
    row.api_requests,
    row.total_tokens,
  ];
  function applyDates(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const fields = new FormData(event.currentTarget);
    setDates({ start: String(fields.get("start") ?? ""), end: String(fields.get("end") ?? "") });
  }
  return (
    <>
      <ControlPlaneSectionHeader section="fleet-usage" />
      <div className="flex gap-2" aria-label={t("usageViews")}>
        {usageViews.map((item) => (
          <Button
            key={item}
            variant={view === item ? "secondary" : "outline"}
            aria-pressed={view === item}
            onClick={() => setView(item)}
          >
            {t(item)}
          </Button>
        ))}
      </div>
      <form className="flex flex-wrap items-end gap-3" onSubmit={applyDates}>
        <label>
          {t("startDate")}
          <Input type="date" name="start" />
        </label>
        <label>
          {t("endDate")}
          <Input type="date" name="end" />
        </label>
        <Button type="submit">{t("applyDates")}</Button>
      </form>
      <ErrorNotice error={result.error} />
      {result.isPending && <p role="status">{t("loadingUsage")}</p>}
      {result.data && (
        <>
          <p>
            {result.data.window.start_date} {t("to")} {result.data.window.end_date}
          </p>
          <Coverage result={result.data} />
        </>
      )}
      {view === "summary" && summary.data && (
        <ResultsTable
          {...tableRefresh}
          columns={[t("dataplane"), t("spendUsd"), t("requests"), t("tokens")]}
          rows={[
            [t("fleetTotal"), ...metrics(summary.data.totals)],
            ...summary.data.by_dataplane.map((row) => [row.id, ...metrics(row.metrics)]),
          ]}
        />
      )}
      {view === "daily" && daily.data && (
        <ResultsTable
          {...tableRefresh}
          columns={[t("date"), t("group"), t("spendUsd"), t("requests"), t("tokens")]}
          rows={daily.data.rows.map((row) => [row.date, row.group, ...metrics(row.metrics)])}
        />
      )}
      {view === "top" && top.data && (
        <ResultsTable
          {...tableRefresh}
          columns={[t("group"), t("spendUsd"), t("requests"), t("tokens")]}
          rows={top.data.items.map((row) => [row.group, ...metrics(row.metrics)])}
        />
      )}
    </>
  );
}

export function ControlPlanePage({ section }: { section: string }) {
  const session = useControlPlane();
  if (!session) return null;
  const { api, principal } = session;
  const isAdminSection = section === "dataplanes" || section === "roles" || section === "audit";
  return (
    <Page>
      <PageContent className="space-y-4">
        {(section === "fleet-models" || section === "fleet-teams") && <Fleet section={section} api={api} />}
        {section === "fleet-usage" && <Usage api={api} />}
        {isAdminSection && (
          <ControlPlaneAdmin section={section} api={api} principal={principal} planes={session.planes} />
        )}
      </PageContent>
    </Page>
  );
}
