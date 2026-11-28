import { useState } from "react";
import { formatDistanceToNow } from "date-fns";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useUrlTab } from "@/hooks/useUrlTab";
import { Source, SpendReport, useCloudMutation, useCloudSpend } from "./hooks";
import { CloudTable } from "./Table";

const usd = (value: string | number) =>
  new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", currencyDisplay: "code" }).format(Number(value));
type Member = SpendReport["members"][number];
type Team = SpendReport["teams"][number];

export default function Spend({ accessToken, source }: { accessToken: string; source: Source }) {
  const { t } = useTranslation();
  const [apiKey, setApiKey] = useState("");
  const [days, setDays] = useState(30);
  const [group, setGroup] = useUrlTab(["member", "team"], "member", "cloud_group");
  const data = useCloudSpend(accessToken, source, days);
  const save = useCloudMutation(accessToken, "PUT", `/sources/${source.source_id}/admin_api_key`);
  const remove = useCloudMutation(accessToken, "DELETE", `/sources/${source.source_id}/admin_api_key`);
  const sync = useCloudMutation(accessToken, "POST", `/sources/${source.source_id}/sync_spend`);
  if (!source.has_admin_api_key)
    return (
      <section className="max-w-2xl space-y-4 rounded-xl border bg-card p-6">
        <h2 className="text-lg font-semibold">{t("agents.external.spendTitle")}</h2>
        <p className="text-sm leading-relaxed text-muted-foreground">{t("agents.external.spendDescription")}</p>
        <form
          className="space-y-3"
          onSubmit={(event) => {
            event.preventDefault();
            save.mutate({ api_key: apiKey }, { onSuccess: () => setApiKey("") });
          }}
        >
          <label htmlFor="cloud-admin-key" className="block text-sm font-medium">
            {t("agents.external.apiKey")}
          </label>
          <Input
            id="cloud-admin-key"
            type="password"
            autoComplete="off"
            value={apiKey}
            onChange={(event) => setApiKey(event.target.value)}
          />
          {save.error && (
            <p role="alert" className="text-sm text-destructive">
              {save.error.message}
            </p>
          )}
          <Button type="submit" disabled={!apiKey.trim() || save.isPending}>
            {t("agents.external.connectSpend")}
          </Button>
        </form>
      </section>
    );
  const members = [...(data.data?.members ?? [])].sort((a, b) => Number(b.grok_cost_usd) - Number(a.grok_cost_usd));
  const teams = [...(data.data?.teams ?? [])].sort((a, b) => Number(b.grok_cost_usd) - Number(a.grok_cost_usd));
  const numbers = <T extends Member | Team>() => [
    { key: "grok", heading: t("agents.external.grokCost"), cell: (row: T) => usd(row.grok_cost_usd) },
    {
      key: "gateway",
      heading: t("agents.external.liteLLMSpend"),
      cell: (row: T) => usd(row.litellm_gateway_spend_usd),
    },
    { key: "tokens", heading: t("agents.external.tokens"), cell: (row: T) => row.tokens.toLocaleString() },
    { key: "requests", heading: t("agents.external.requests"), cell: (row: T) => row.requests.toLocaleString() },
  ];
  const memberColumns = [
    {
      key: "member",
      heading: t("agents.external.member"),
      cell: (row: Member) => (
        <div>
          <div className="font-medium">{row.display_name || row.email || row.vendor_user_id}</div>
          {row.email && <div className="text-xs text-muted-foreground">{row.email}</div>}
        </div>
      ),
    },
    ...numbers<Member>(),
  ];
  const teamColumns = [
    {
      key: "team",
      heading: t("agents.external.team"),
      cell: (row: Team) => (
        <div>
          <div className="font-medium">{row.team_name}</div>
          {!!row.teams?.length && (
            <div className="text-xs text-muted-foreground">{row.teams.map((team) => team.team_name).join(", ")}</div>
          )}
        </div>
      ),
    },
    ...numbers<Team>(),
  ];
  const error = [save.error, sync.error, remove.error, data.error].find(Boolean);
  const totals = data.data?.totals ?? { grok_cost_usd: "0", litellm_gateway_spend_usd: "0" };
  const tiles = [
    { name: t("agents.external.grokCost"), value: totals.grok_cost_usd },
    { name: t("agents.external.gatewayCost"), value: totals.litellm_gateway_spend_usd },
    {
      name: t("agents.external.total"),
      value: Number(totals.grok_cost_usd) + Number(totals.litellm_gateway_spend_usd),
    },
  ];
  return (
    <div className="space-y-6">
      <div className="space-y-1 text-sm text-muted-foreground">
        {data.data && <p>{t("agents.external.period", { start: data.data.start_day, end: data.data.end_day })}</p>}
        <p>{t("agents.external.spendCoverage")}</p>
        <p>{t("agents.external.teamAttribution")}</p>
      </div>
      <div className="flex flex-wrap items-center gap-3">
        <label className="text-sm font-medium" htmlFor="cloud-spend-range">
          {t("agents.external.range")}
        </label>
        <select
          id="cloud-spend-range"
          className="h-9 rounded-md border bg-background px-3 text-sm"
          value={days}
          onChange={(event) => setDays(Number(event.target.value))}
        >
          {[7, 30].map((count) => (
            <option key={count} value={count}>
              {t("agents.external.days", { count })}
            </option>
          ))}
        </select>
        <Button variant="outline" disabled={sync.isPending} onClick={() => sync.mutate()}>
          {t("agents.external.refresh")}
        </Button>
        <span className="text-sm text-muted-foreground">
          {source.last_spend_sync_at
            ? t("agents.external.lastSynced", {
                time: formatDistanceToNow(new Date(source.last_spend_sync_at), { addSuffix: true }),
              })
            : t("agents.external.neverSynced")}
        </span>
        <Button variant="outline" className="ml-auto" disabled={remove.isPending} onClick={() => remove.mutate()}>
          {t("agents.external.removeKey")}
        </Button>
      </div>
      {(error || source.last_spend_sync_error) && (
        <p role="alert" className="text-sm text-destructive">
          {error?.message || source.last_spend_sync_error}
        </p>
      )}
      {data.isLoading ? (
        <p role="status">{t("agents.external.loading")}</p>
      ) : (
        <>
          <div className="grid gap-4 sm:grid-cols-3">
            {tiles.map((tile) => (
              <div key={tile.name} className="rounded-xl border bg-card p-5">
                <p className="text-sm text-muted-foreground">{tile.name}</p>
                <p className="mt-2 text-2xl font-semibold tabular-nums">{usd(tile.value)}</p>
              </div>
            ))}
          </div>
          <div className="flex gap-2" aria-label={t("agents.external.spend")}>
            <Button
              variant={group === "member" ? "default" : "outline"}
              aria-pressed={group === "member"}
              onClick={() => setGroup("member")}
            >
              {t("agents.external.byMember")}
            </Button>
            <Button
              variant={group === "team" ? "default" : "outline"}
              aria-pressed={group === "team"}
              onClick={() => setGroup("team")}
            >
              {t("agents.external.byTeam")}
            </Button>
          </div>
          {group === "member" ? (
            <CloudTable rows={members} columns={memberColumns} rowKey={(row) => row.vendor_user_id} />
          ) : (
            <CloudTable rows={teams} columns={teamColumns} rowKey={(row) => row.team_id ?? row.team_name} />
          )}
        </>
      )}
    </div>
  );
}
