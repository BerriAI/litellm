"use client";

import { useMemo } from "react";
import { useQueryStates } from "nuqs";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import type { ColumnDef } from "@tanstack/react-table";
import { ArrowUpRight, Check, Copy, RotateCcw, X } from "lucide-react";
import { useTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import { DataTable, useUrlTableState } from "@/components/shared/DataTable";
import { Button, buttonVariants } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { backgroundFetchClient } from "@/lib/http/api";
import type { components } from "@/lib/http/schema";
import { keyDetailHref, modelGroupHref } from "@/utils/entityLinks";
import { copyToClipboard } from "@/utils/dataUtils";
import { AlertStatus, effectiveAlertStatus } from "@/components/ModelDeprecationAlerts/migrationState";
import { migrationUrlParsers } from "./migrationRoute";
import type { DeprecationAlert, deprecationUsageRange } from "@/components/ModelDeprecationAlerts/deprecationAlerts";
import RecommendedReplacement from "@/components/ModelDeprecationAlerts/RecommendedReplacement";
import { DeprecationTiming } from "./DeprecationModel";
import UsageTags from "./UsageTags";
import ModelUsageChart from "./ModelUsageChart";
import { affectedTeams } from "./migrationData";

const TABLE_STATE = { sortFields: [], defaultSort: { id: "", desc: false }, defaultPageSize: 50, filterColumns: [] };
type KeyRow = components["schemas"]["KeySpendActivityRow"];

function migrationHeading(alert: DeprecationAlert, status: AlertStatus): string {
  if (status === AlertStatus.Resolved) return "Migration marked resolved";
  if (status === AlertStatus.Dismissed) return "Model deprecation dismissed";
  if (alert.model.status === "deprecated") return "Migration needed: model is deprecated";
  return `Migration due ${alert.model.deprecation_date}`;
}

function migrationFollowUp(
  alert: DeprecationAlert,
  context: { team: string; count: number | undefined; lastSeen: string; url: string },
): string {
  const retirement = alert.model.status === "deprecated" ? "retired on" : "retires on";
  const keys = context.count === undefined ? "the affected" : context.count;
  return `${context.team}: please migrate ${keys} virtual keys using ${alert.usageModel}. This model ${retirement} ${alert.model.deprecation_date}. Last seen ${context.lastSeen} UTC. Please confirm once migration is complete. Affected keys: ${context.url}`;
}

export default function MigrationDetail({
  selected,
  scope,
  range,
  isViewOnly,
}: {
  selected: DeprecationAlert;
  scope: string;
  range: ReturnType<typeof deprecationUsageRange>;
  isViewOnly: boolean;
}) {
  const [params, setParams] = useQueryStates(migrationUrlParsers);
  const { data: teams = [] } = useTeams();
  const teamNames = useMemo(
    () => new Map(teams.map((team) => [team.team_id, team.team_alias || team.team_id])),
    [teams],
  );
  const queryClient = useQueryClient();
  const table = useUrlTableState(TABLE_STATE);
  const model = selected.usageModel;
  const keyQuery = useQuery({
    queryKey: ["deprecationImpactKeys", scope, model, params.team, range, table.pagination],

    queryFn: async ({ signal }) => {
      const query = {
        ...range,
        model,
        offset: table.pagination.pageIndex * table.pagination.pageSize,
        limit: table.pagination.pageSize,
        timezone: 0,
      };
      if (params.team) {
        const { data } = await backgroundFetchClient.GET("/team/daily/activity/aggregated/keys", {
          params: { query: { ...query, team_ids: params.team } },
          signal,
        });
        if (!data) throw new Error("Affected keys are unavailable");
        return data;
      }
      const { data } = await backgroundFetchClient.GET("/user/daily/activity/aggregated/keys", {
        params: { query },
        signal,
      });
      if (!data) throw new Error("Affected keys are unavailable");
      return data;
    },
  });
  const keys = useMemo(() => keyQuery.data?.api_keys.map((key) => key.api_key) ?? [], [keyQuery.data]);
  const tagQuery = useQuery({
    queryKey: ["deprecationUsageTags", scope, model, range, keys],
    enabled: keys.length > 0,
    queryFn: async ({ signal }) => {
      const { data } = await backgroundFetchClient.GET("/model/deprecations/alerts/tags", {
        params: { query: { ...range, model, api_keys: keys } },
        signal,
      });
      if (!data) throw new Error("Usage tags are unavailable");
      return data;
    },
    staleTime: 5 * 60 * 1000,
  });
  const tagsByKey = useMemo(() => {
    const tags = new Map<string, string[]>();
    for (const usage of tagQuery.data ?? []) {
      const existing = tags.get(usage.api_key) ?? [];
      existing.push(usage.tag);
      tags.set(usage.api_key, existing);
    }
    return tags;
  }, [tagQuery.data]);
  const teamQuery = useQuery({
    queryKey: ["deprecationImpactTeams", scope, model, range],

    queryFn: async ({ signal }) => {
      const query = { ...range, model, timezone: 0, api_key_limit: 1 };
      const { data } = await backgroundFetchClient.GET("/team/daily/activity/aggregated", {
        params: { query },
        signal,
      });
      if (!data) throw new Error("Affected teams are unavailable");
      return affectedTeams(data);
    },
  });
  const statusMutation = useMutation({
    mutationFn: async (status: AlertStatus) => {
      const body = {
        model: selected.usageModel,
        deprecation_date: selected.model.deprecation_date,
        status,
        last_seen_date: selected.lastUsedDate,
        last_seen_requests: selected.lastDayRequests,
      };
      const { data } = await backgroundFetchClient.PUT("/model/deprecations/alerts/state", { body });
      if (!data) throw new Error("Could not save alert status");
      return data;
    },
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["deprecationAlerts"] }),
  });
  const columns = useMemo<ColumnDef<KeyRow>[]>(
    () => [
      {
        id: "key",
        header: "Virtual key",
        cell: ({ row }) => (
          <a href={keyDetailHref(row.original.api_key)} className="font-medium text-primary hover:underline">
            {row.original.metadata.key_alias || "Unnamed key"}
          </a>
        ),
      },
      {
        id: "team",
        header: "Team",
        cell: ({ row }) =>
          teamNames.get(row.original.metadata.team_id ?? "") || row.original.metadata.team_id || "No team",
      },
      {
        id: "owner",
        header: "Owner",
        cell: ({ row }) => row.original.metadata.user_email || row.original.metadata.user_id || "Not recorded",
      },
      {
        id: "tags",
        header: "Usage tags",
        cell: ({ row }) => (
          <UsageTags
            tags={tagsByKey.get(row.original.api_key) ?? []}
            loading={tagQuery.isPending}
            failed={tagQuery.isError}
          />
        ),
      },
      {
        id: "requests",
        header: "Requests · 30d",
        cell: ({ row }) => <span className="tabular-nums">{row.original.metrics.api_requests.toLocaleString()}</span>,
      },
      { id: "spend", header: "Spend · 30d", cell: ({ row }) => `$${row.original.metrics.spend.toFixed(2)}` },
    ],
    [teamNames, tagsByKey, tagQuery.isPending, tagQuery.isError],
  );
  const status = effectiveAlertStatus(selected);
  const heading = migrationHeading(selected, status);
  const copyFollowUp = () => {
    const teamName = params.team ? teamNames.get(params.team) || params.team : "Team owners";
    const count = keyQuery.data?.total_api_keys;
    const lastSeen = params.team
      ? teamQuery.data?.find((team) => team.id === params.team)?.lastUsedDate ?? selected.lastUsedDate
      : selected.lastUsedDate;
    const context = { team: teamName, count, lastSeen, url: window.location.href };
    const message = migrationFollowUp(selected, context);
    void copyToClipboard(message, "Follow-up copied");
  };
  return (
    <>
      <section className="flex flex-wrap items-start justify-between gap-3 rounded-xl border bg-card p-3">
        <div>
          <span className="rounded-full border px-2 py-0.5 text-xs capitalize">{status}</span>
          <h2 className="mt-3 text-lg font-semibold">{heading}</h2>
          <p className="mt-1 text-sm text-muted-foreground">
            Last seen {selected.lastUsedDate} UTC · {selected.requests.toLocaleString()} requests in the last 30 days
          </p>
          <RecommendedReplacement model={selected.model.successor_model} />
        </div>
        <div className="flex flex-wrap gap-2">
          <Button variant="outline" onClick={copyFollowUp}>
            <Copy />
            Copy follow-up
          </Button>
          {!isViewOnly && (
            <>
              {status === AlertStatus.Open ? (
                <>
                  <Button
                    disabled={statusMutation.isPending}
                    onClick={() => statusMutation.mutate(AlertStatus.Resolved)}
                  >
                    <Check />
                    Mark resolved
                  </Button>
                  <Button
                    variant="ghost"
                    disabled={statusMutation.isPending}
                    onClick={() => statusMutation.mutate(AlertStatus.Dismissed)}
                  >
                    <X />
                    Dismiss
                  </Button>
                </>
              ) : (
                <Button disabled={statusMutation.isPending} onClick={() => statusMutation.mutate(AlertStatus.Open)}>
                  <RotateCcw />
                  Reopen
                </Button>
              )}
            </>
          )}
          {modelGroupHref(selected.model.model_name) && (
            <a className={buttonVariants({ variant: "outline" })} href={modelGroupHref(selected.model.model_name)}>
              Model routing <ArrowUpRight size={14} />
            </a>
          )}
        </div>
      </section>
      <p className="text-xs text-muted-foreground">
        Resolve across all teams once owners confirm migration. New traffic reopens the alert.
      </p>
      {statusMutation.isError && (
        <p role="alert" className="text-destructive">
          Could not save alert status. Please retry.
        </p>
      )}
      <ModelUsageChart model={model} team={params.team} scope={scope} range={range} />
      <div className="grid shrink-0 grid-cols-3 divide-x rounded-xl border bg-card">
        <div className="p-3">
          <p className="text-xs text-muted-foreground">{params.team ? "Keys in this team" : "Affected keys"}</p>
          <p className="mt-1 text-xl font-semibold tabular-nums">
            {keyQuery.data?.total_api_keys.toLocaleString() ?? "—"}
          </p>
        </div>
        <div className="p-3">
          <p className="text-xs text-muted-foreground">Teams using this model</p>
          <p className="mt-1 text-xl font-semibold tabular-nums">{teamQuery.data?.length ?? "—"}</p>
        </div>
        <div className="p-3">
          <p className="text-xs text-muted-foreground">Deprecation</p>
          <p className="mt-1 text-lg font-semibold">
            <DeprecationTiming alert={selected} />
          </p>
        </div>
      </div>
      <section className="flex h-[480px] shrink-0 flex-col overflow-hidden rounded-xl border">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b p-4">
          <div>
            <h2 className="font-medium">
              Affected keys{" "}
              <span className="ml-1 text-muted-foreground">
                {keyQuery.data?.total_api_keys.toLocaleString() ?? "…"}
              </span>
            </h2>
            <p className="mt-1 text-xs text-muted-foreground">
              Keys and owners that used this model in the last 30 days. Tags show actual usage with this model.
            </p>
          </div>
          <Select
            value={params.team ?? "__all"}
            onValueChange={(value) => {
              void setParams({ team: value === "__all" ? null : value });
              table.onPaginationChange({ ...table.pagination, pageIndex: 0 });
            }}
          >
            <SelectTrigger className="min-w-52" aria-label="Filter affected keys by team">
              <SelectValue>{params.team ? teamNames.get(params.team) || params.team : "All teams"}</SelectValue>
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="__all">All teams</SelectItem>
              {(teamQuery.data ?? []).map((team) => (
                <SelectItem key={team.id} value={team.id}>
                  {teamNames.get(team.id) || team.id} · {team.keys} keys
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        {keyQuery.isError ? (
          <div className="p-5">
            <p>Could not load affected keys.</p>
            <Button variant="outline" onClick={() => void keyQuery.refetch()}>
              Retry
            </Button>
          </div>
        ) : (
          <DataTable
            columns={columns}
            data={keyQuery.data?.api_keys ?? []}
            isLoading={keyQuery.isPending}
            paginationMode="server"
            pagination={table.pagination}
            onPaginationChange={table.onPaginationChange}
            rowCount={keyQuery.data?.total_api_keys ?? 0}
            fillHeight
            size="compact"
            getRowId={(row) => row.api_key}
          />
        )}
      </section>
      {teamQuery.isError && (
        <p role="alert">Could not load team filters. The keys table still includes all matching teams.</p>
      )}
    </>
  );
}
