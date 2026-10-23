"use client";

import { useMemo } from "react";
import { ArrowLeft, Coins } from "lucide-react";
import { useTagSummary } from "@/app/(dashboard)/hooks/tags/useTagSummary";
import AdvancedDatePicker from "@/components/shared/advanced_date_picker";
import type { DateRangePickerValue } from "@/components/shared/date_picker_types";
import UserDropdown from "@/components/common_components/UserDropdown";
import { SearchSelect } from "@/components/shared/SearchSelect";
import type { ClientUsageScopeOption } from "../hooks/clientUsageScope";
import { stackedUsageColor } from "@/components/shared/charts";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { Skeleton } from "@/components/ui/skeleton";
import { EMPTY_DAILY_ACTIVITY_METADATA, toDailyData } from "@/components/UsagePage/dailyActivityApi";
import type { DailyData } from "@/components/UsagePage/types";
import claudeLogo from "../../../../../../public/assets/logos/claude.svg";
import type { UsageClient } from "../hooks/clientUsageData";
import { useClientUsage } from "../hooks/useClientUsage";
import { topAgents, type AgentRow } from "./overview/agentCatalog";
import { useClientAgentModels } from "../hooks/useClientAgentModels";
import UsageBreakdownChart from "./overview/UsageBreakdownChart";
import { clientAgentDetail, type ClientAgentDetail } from "./overview/clientAgentDetail";
import { modelColorMap, formatCompact, formatUsd, overviewTotals } from "./overview/overviewData";
import { reconciledTokenBreakdown } from "./overview/tokenBreakdown";
import { Panel, Stat } from "./overview/Primitives";
import { AgentMark } from "./overview/TopAgents";

interface ClientUsagePanelProps {
  client: UsageClient;
  userId: string | null;
  canChooseUser: boolean;
  teamId?: string | null;
  scopeError?: string | null;
  teams?: readonly ClientUsageScopeOption[];
  members?: readonly ClientUsageScopeOption[];
  defaultScopeLabel?: string;
  onTeamChange?: (teamId: string | null) => void;
  onUserChange: (userId: string | null) => void;
  onClose: () => void;
  accessToken: string | null;
  startTime: Date | null;
  endTime: Date | null;
  dateValue: DateRangePickerValue;
  presetShortLabel: string | null;
  onDateChange: (value: DateRangePickerValue, preset: string | null) => void;
  colorSource?: readonly DailyData[];
  agent: AgentRow | null;
  daily: readonly DailyData[];
  loading: boolean;
  summaryFailed?: boolean;
  onRetry?: () => void;
}

const dateLabel = (date: Date | null) => date?.toLocaleDateString("en-US", { month: "short", day: "numeric" }) ?? "—";
const countLabel = (value: number | null) => (value === null ? "Unavailable" : value.toLocaleString());

export default function ClientUsagePanel({
  client,
  userId,
  canChooseUser,
  teamId = null,
  scopeError: accessError = null,
  teams = [],
  members = [],
  defaultScopeLabel = "All users",
  onTeamChange,
  onUserChange,
  onClose,
  accessToken,
  startTime,
  endTime,
  dateValue,
  presetShortLabel,
  onDateChange,
  colorSource,
  agent,
  daily,
  loading,
  summaryFailed = false,
  onRetry,
}: ClientUsagePanelProps) {
  const label = client === "claude-desktop" ? "Claude Desktop" : "Claude Code";
  const scopedRequest = Boolean(userId || teamId) && !accessError;
  const personRequest = {
    accessToken,
    startTime,
    endTime,
    userId,
    teamId,
    client: scopedRequest ? client : null,
  };
  const personUsage = useClientUsage(personRequest);
  const allUsersEnabled = canChooseUser && !scopedRequest && !accessError;
  const teamLabel = teams.find((team) => team.value === teamId)?.label ?? teamId;
  const personLabel = members.find((member) => member.value === userId)?.label ?? "Selected user";
  const selectedScopeLabel = (() => {
    if (teamId) return `${teamLabel} · ${userId ? personLabel : "All members"}`;
    if (!canChooseUser) return "My usage";
    return userId ? "Selected user" : "All users";
  })();
  const summaryQuery = useTagSummary(startTime, endTime, allUsersEnabled && agent === null);
  const activeAgent = useMemo(
    () => agent ?? topAgents(summaryQuery.data ?? []).find((row) => row.id === client) ?? null,
    [summaryQuery.data, agent, client],
  );
  const modelRequest = { accessToken, startTime, endTime, agent: activeAgent, enabled: allUsersEnabled };
  const agentModels = useClientAgentModels(modelRequest);
  const agentDaily = agentModels.results.length > 0 ? agentModels.results : daily;
  const agentDetail = useMemo(
    () => (activeAgent ? clientAgentDetail(activeAgent, agentDaily) : null),
    [activeAgent, agentDaily],
  );
  const colorByModel = useMemo(() => (colorSource ? modelColorMap(colorSource) : undefined), [colorSource]);
  const personTotals = overviewTotals(personUsage.data.metadata ?? EMPTY_DAILY_ACTIVITY_METADATA, null);
  const pending =
    !accessError && (scopedRequest ? personUsage.loading : (loading || summaryQuery.isLoading) && activeAgent === null);
  const summaryError = summaryQuery.isError || summaryFailed ? "Could not load client totals" : null;
  const scopeError =
    accessError ?? (!canChooseUser && !scopedRequest ? "Sign in to view your client usage" : summaryError);
  const error = accessError ?? (scopedRequest ? personUsage.error : scopeError);
  const summary = scopedRequest ? personTotals : agentDetail?.summary;
  const rawBreakdown = scopedRequest
    ? {
        inputTokens: personTotals.inputTokens,
        outputTokens: personTotals.outputTokens,
        cacheReadTokens: personUsage.cacheAvailable ? personTotals.cacheReadTokens : null,
        cacheWriteTokens: personUsage.cacheAvailable ? personTotals.cacheWriteTokens : null,
      }
    : agentDetail?.tokenBreakdown;
  const breakdown = rawBreakdown && summary ? reconciledTokenBreakdown(summary.tokens, rawBreakdown) : undefined;
  const modelResults = scopedRequest ? toDailyData(personUsage.data) : agentModels.results;
  const chartLoading = scopedRequest ? personUsage.loading : agentModels.loading;
  const chartError = scopedRequest ? personUsage.error : agentModels.error;

  const canFilterPerson = canChooseUser && !teamId && !accessError;
  const canRetry = !accessError && (scopedRequest || canChooseUser);
  const ready = !pending && !error;
  const retryModels = () => {
    void summaryQuery.refetch();
    onRetry?.();
    agentModels.reload();
  };

  return (
    <Sheet
      open
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <SheetContent className="gap-0 data-[side=right]:w-[calc(100%-16px)] data-[side=right]:sm:max-w-[600px]">
        <SheetHeader className="shrink-0 border-b px-6 pt-5 pb-4">
          <Button variant="ghost" size="sm" className="mb-3 w-fit -ml-2 text-muted-foreground" onClick={onClose}>
            <ArrowLeft className="size-4" /> Back to Usage
          </Button>
          <div className="flex items-center gap-3">
            <AgentMark agent={{ id: client, label, logo: claudeLogo.src }} />
            <div>
              <SheetTitle className="text-sm font-medium">{label}</SheetTitle>
              <SheetDescription className="mt-0.5 text-xs">
                {dateLabel(startTime)} – {dateLabel(endTime)} · {selectedScopeLabel}
              </SheetDescription>
            </div>
          </div>
        </SheetHeader>
        <div className="min-h-0 flex-1 space-y-3 overflow-y-auto bg-background px-4 py-4 sm:px-6">
          <div className="border-b pb-3">
            <AdvancedDatePicker
              compact
              value={dateValue}
              presetShortLabel={presetShortLabel}
              onValueChange={onDateChange}
            />
          </div>
          {teams.length > 0 && onTeamChange && (
            <div className="space-y-2">
              <label htmlFor="client-usage-team" className="text-xs font-medium text-muted-foreground">
                Usage scope
              </label>
              <SearchSelect
                id="client-usage-team"
                aria-label="Usage scope"
                value={teamId ? `team:${teamId}` : "default"}
                options={[
                  { value: "default", label: defaultScopeLabel },
                  ...teams.map((team) => ({ ...team, value: `team:${team.value}` })),
                ]}
                allowClear={false}
                onValueChange={(value) => onTeamChange(value?.startsWith("team:") ? value.slice(5) : null)}
              />
            </div>
          )}
          {teamId && (
            <div className="space-y-2">
              <label htmlFor="client-usage-member" className="text-xs font-medium text-muted-foreground">
                Team member
              </label>
              <SearchSelect
                id="client-usage-member"
                aria-label="Team member"
                value={userId ? `user:${userId}` : "all"}
                options={[
                  { value: "all", label: "All members" },
                  ...members.map((member) => ({ ...member, value: `user:${member.value}` })),
                ]}
                allowClear={false}
                onValueChange={(value) => onUserChange(value?.startsWith("user:") ? value.slice(5) : null)}
              />
            </div>
          )}
          {canFilterPerson && (
            <div className="space-y-2">
              <div className="flex items-center justify-between gap-2">
                <label htmlFor="client-usage-user" className="text-xs font-medium text-muted-foreground">
                  Filter by person
                </label>
                {userId && (
                  <Button variant="ghost" size="sm" onClick={() => onUserChange(null)}>
                    All users
                  </Button>
                )}
              </div>
              <UserDropdown id="client-usage-user" value={userId} onChange={onUserChange} />
            </div>
          )}
          {pending && (
            <div role="status" aria-label="Loading client usage">
              <Skeleton className="h-24 w-full" />
            </div>
          )}
          {error && (
            <div role="alert" className="rounded-lg border p-4">
              <p className="font-medium">Could not load client usage</p>
              <p className="mt-1 text-muted-foreground">{error}</p>
              {canRetry && (
                <Button
                  className="mt-3"
                  variant="outline"
                  onClick={() => {
                    if (scopedRequest) {
                      personUsage.reload();
                    } else {
                      void summaryQuery.refetch();
                      onRetry?.();
                    }
                  }}
                >
                  Try again
                </Button>
              )}
            </div>
          )}
          {ready && !summary && (
            <p className="rounded-lg border p-4 text-muted-foreground">No {label} usage recorded in this date range.</p>
          )}
          {ready && summary && (
            <>
              <div className="grid grid-cols-3 divide-x overflow-hidden rounded-xl border bg-card py-4">
                <SummaryStat label="Spend" value={formatUsd(summary.spend)} />
                <SummaryStat label="Total tokens" value={formatCompact(summary.tokens)} />
                <SummaryStat label="Requests" value={summary.requests.toLocaleString()} />
              </div>
              {summary.requests === 0 ? (
                <p className="text-sm text-muted-foreground">
                  No {label} requests in the selected range. Try another date range.
                  {(canChooseUser || teamId) && " You can also choose another person."}
                </p>
              ) : (
                <>
                  {!chartError && (
                    <UsageBreakdownChart
                      dimension="models"
                      results={modelResults}
                      loading={chartLoading}
                      compact
                      colorByModel={colorByModel}
                    />
                  )}
                  {chartError && (
                    <div className="rounded-lg border p-4">
                      <p role="status" className="text-xs text-muted-foreground">
                        {chartError}
                      </p>
                      <Button variant="outline" size="sm" className="mt-3" onClick={retryModels}>
                        Retry model breakdown
                      </Button>
                    </div>
                  )}
                  {breakdown && <TokenExplanation total={summary.tokens} breakdown={breakdown} />}
                </>
              )}
              <p className="border-t pt-3 text-xs leading-5 text-muted-foreground">
                {scopedRequest
                  ? "Based on retained request logs within the exact selected time range."
                  : "All-user totals use UTC daily aggregates for the selected dates. Filtering by person uses the exact selected times. Cache values are reported counts."}{" "}
                Spend is the cost recorded by LiteLLM.
              </p>
            </>
          )}
        </div>
      </SheetContent>
    </Sheet>
  );
}

function SummaryStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0 px-4">
      <Stat label={label} value={value} />
    </div>
  );
}

function TokenExplanation({ total, breakdown }: { total: number; breakdown: ClientAgentDetail["tokenBreakdown"] }) {
  const { inputTokens, outputTokens, cacheReadTokens, cacheWriteTokens } = breakdown;
  const freshInput =
    inputTokens !== null && cacheReadTokens !== null && cacheWriteTokens !== null
      ? inputTokens - cacheReadTokens - cacheWriteTokens
      : null;
  const rows = [
    { label: "Cache reads", value: cacheReadTokens, color: stackedUsageColor(0) },
    { label: "Cache writes", value: cacheWriteTokens, color: stackedUsageColor(1) },
    { label: "Uncached input", value: freshInput, color: stackedUsageColor(2) },
    { label: "Output", value: outputTokens, color: stackedUsageColor(3) },
  ];
  const complete = rows.every((row) => row.value !== null);
  return (
    <Panel icon={Coins} title="Token breakdown" subtitle={`${formatCompact(total)} total tokens`} bodyClassName="gap-3">
      {complete && total > 0 && (
        <div aria-hidden="true" className="flex h-2 overflow-hidden rounded-full bg-muted">
          {rows.map((row) => (
            <span
              key={row.label}
              style={{ backgroundColor: row.color, width: `${((row.value ?? 0) / total) * 100}%` }}
            />
          ))}
        </div>
      )}
      {cacheReadTokens !== null && total > 0 && (
        <p className="text-xs text-muted-foreground">
          {((cacheReadTokens / total) * 100).toFixed(1)}% of these tokens are cache reads.
        </p>
      )}
      <dl className="divide-y rounded-lg border px-3 text-xs">
        {rows.map((row) => (
          <div key={row.label} className="flex items-center justify-between gap-3 py-2.5">
            <dt className="flex items-center gap-2 text-muted-foreground">
              <span className="size-2 rounded-[2px]" style={{ backgroundColor: row.color }} />
              {row.label}
            </dt>
            <dd className="font-medium tabular-nums">{countLabel(row.value)}</dd>
          </div>
        ))}
      </dl>
      <p className="text-xs leading-5 text-muted-foreground">
        Total tokens include input, cached input and output. Cached input is priced differently, so token count alone
        does not explain spend. Compare the same person and dates when checking the count shown by a client.
      </p>
    </Panel>
  );
}
