import {
  AreaChart,
  BarChart,
  type ChartColor,
  type ChartTooltipProps,
  chartColorValue,
  CustomTooltip,
  formatCategoryName,
  LineChart,
  stackedUsageColor,
  ValueTooltip,
} from "@/components/shared/charts";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import { resolveTeamAliasFromTeamID } from "@/utils/teamUtils";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Panel, Stat } from "@/app/(dashboard)/usage/_components/components/overview/Primitives";
import { ProviderLogo } from "@/components/molecules/models/ProviderLogo";
import { providerForModel } from "@/app/(dashboard)/usage/_components/components/overview/modelProvider";
import { cn } from "@/lib/cva.config";
import { Activity, ChevronDown, Coins, Database, KeyRound, Timer } from "lucide-react";
import React, { type ReactNode, useRef, useState } from "react";
import { Team } from "./key_team_helpers/key_list";
import KeyModelUsageView from "./UsagePage/components/KeyModelUsageView";
import { keyActivityLabel } from "./UsagePage/keyActivityLabel";
import type { ModelTopKeysResponse } from "./UsagePage/dailyActivityApi";
import { DailyData, KeyMetricWithMetadata, ModelActivityData, TopModelData } from "./UsagePage/types";
import { averageResponseTimeMs, formatResponseTime, valueFormatter } from "./UsagePage/utils/value_formatters";

interface ActivityMetricsProps {
  modelMetrics: Record<string, ModelActivityData>;
  summaryMetrics?: ModelActivityData;
  summaryTitle?: string;
  hidePromptCachingMetrics?: boolean;
  fetchTopApiKeys?: (model: string) => Promise<ModelTopKeysResponse>;
}

const BRAND: ChartColor = "#2b3fd6";
const TOKEN_CATEGORIES = ["metrics.prompt_tokens", "metrics.completion_tokens", "metrics.total_tokens"] as const;
const TOKEN_COLORS: readonly ChartColor[] = ["blue", "cyan", "indigo"];
const REQUEST_CATEGORIES = ["metrics.successful_requests", "metrics.failed_requests"] as const;
const CACHE_CATEGORIES = ["metrics.cache_read_input_tokens", "metrics.cache_creation_input_tokens"] as const;
const CHART_CLASS = "mt-1 h-56";

const modelAverageResponseTimeMs = (metrics: ModelActivityData): number | null =>
  averageResponseTimeMs(metrics.total_response_time_ms ?? 0, metrics.total_timed_requests ?? 0);

const formatSpend = (value: number) => `$${formatNumberWithCommas(value, 2, true)}`;

export const ResponseTimeTooltip = ({ active, payload, label }: ChartTooltipProps) => (
  <ValueTooltip
    active={active}
    payload={payload?.map((item) => ({ ...item, name: formatCategoryName(String(item.dataKey ?? "")) }))}
    label={label}
    valueFormatter={formatResponseTime}
  />
);

const SeriesKey = ({ categories, colors }: { categories: readonly string[]; colors: readonly ChartColor[] }) => (
  <div className="flex flex-wrap items-center justify-end gap-x-3 gap-y-1">
    {categories.map((category, idx) => (
      <span key={category} className="flex items-center gap-1.5 text-xs text-muted-foreground">
        <span
          aria-hidden="true"
          className="size-2 shrink-0 rounded-[2px]"
          style={{ backgroundColor: chartColorValue(colors[idx % colors.length]) }}
        />
        {formatCategoryName(category)}
      </span>
    ))}
  </div>
);

/** A stat value kept as a heading so it stays addressable by its accessible name. */
const StatValue = ({ children }: { children: ReactNode }) => <h4 className="truncate">{children}</h4>;

const STRIP_COLUMNS = { 4: "lg:grid-cols-4", 5: "lg:grid-cols-5" } as const;

const StatStrip = ({ columns, children }: { columns: keyof typeof STRIP_COLUMNS; children: ReactNode }) => (
  <div
    className={cn(
      "grid grid-cols-1 overflow-hidden rounded-xl border bg-card sm:grid-cols-2 lg:divide-x",
      STRIP_COLUMNS[columns],
    )}
  >
    {children}
  </div>
);

const StatCell = ({ children }: { children: ReactNode }) => (
  <div className="min-w-0 border-b px-4 pt-3.5 pb-3 last:border-b-0 lg:border-b-0">{children}</div>
);

/**
 * Same tile as the Overview leaderboard's model mark: provider logo for model names, a neutral
 * monogram for anything else (MCP servers, agents), and a series-colored dot so rows read as a set.
 */
const ItemMark = ({ label, color }: { label: string; color: string }) => {
  const provider = providerForModel(label);
  return (
    <span className="relative inline-flex size-7 shrink-0 items-center justify-center rounded-md border bg-background">
      {provider ? (
        <ProviderLogo provider={provider} className="size-4 rounded-[3px]" />
      ) : (
        <span aria-hidden="true" className="text-[11px] font-semibold uppercase text-muted-foreground">
          {label.charAt(0) || "?"}
        </span>
      )}
      <span
        aria-hidden="true"
        className="absolute -right-0.5 -bottom-0.5 size-2 rounded-full ring-2 ring-card"
        style={{ backgroundColor: color }}
      />
    </span>
  );
};

const TopKeysPanel = ({ children }: { children: ReactNode }) => (
  <Panel icon={KeyRound} title="Top Virtual Keys by Spend">
    {children}
  </Panel>
);

const ModelTopKeys = ({
  modelName,
  fetchTopApiKeys,
}: {
  modelName: string;
  fetchTopApiKeys: (model: string) => Promise<ModelTopKeysResponse>;
}) => {
  interface ModelTopKeyRow {
    api_key: string;
    key_alias: string | null;
    team_id: string | null;
    user: string | null;
    spend: number;
    requests: number;
    tokens: number;
  }
  const [settled, setSettled] = useState<{
    modelName: string;
    fetchTopApiKeys: (model: string) => Promise<ModelTopKeysResponse>;
    rows: ModelTopKeyRow[];
    failed: boolean;
  } | null>(null);
  const [retryToken, setRetryToken] = useState(0);

  React.useEffect(() => {
    let cancelled = false;
    fetchTopApiKeys(modelName)
      .then((response) => {
        if (cancelled) return;
        setSettled({
          modelName,
          fetchTopApiKeys,
          rows: response.api_keys.map((row) => ({
            api_key: row.api_key,
            key_alias: row.metadata.key_alias ?? null,
            team_id: row.metadata.team_id ?? null,
            user: row.metadata.user_email ?? row.metadata.user_id ?? null,
            spend: row.metrics.spend,
            requests: row.metrics.api_requests,
            tokens: row.metrics.total_tokens,
          })),
          failed: false,
        });
      })
      .catch((error) => {
        if (cancelled) return;
        console.error(`Failed to fetch top keys for ${modelName}:`, error);
        setSettled({ modelName, fetchTopApiKeys, rows: [], failed: true });
      });
    return () => {
      cancelled = true;
    };
  }, [modelName, fetchTopApiKeys, retryToken]);

  const current = settled?.modelName === modelName && settled.fetchTopApiKeys === fetchTopApiKeys ? settled : null;

  if (current === null) {
    return (
      <TopKeysPanel>
        <p className="py-2 text-xs text-muted-foreground">Loading top keys...</p>
      </TopKeysPanel>
    );
  }

  if (current.failed) {
    return (
      <TopKeysPanel>
        <p className="py-2 text-xs text-muted-foreground">
          Could not load top keys.{" "}
          <button
            type="button"
            className="font-medium text-foreground underline"
            onClick={() => {
              setSettled(null);
              setRetryToken((token) => token + 1);
            }}
          >
            Retry
          </button>
        </p>
      </TopKeysPanel>
    );
  }

  const rows = current.rows;
  if (rows.length === 0) return null;

  return (
    <TopKeysPanel>
      <ol className="divide-y divide-border/60">
        {rows.map((keyData) => {
          const keyLabel = keyData.key_alias || `${keyData.api_key.substring(0, 10)}...`;
          return (
            <li key={keyData.api_key} className="flex items-center justify-between gap-4 py-2.5">
              <div className="min-w-0">
                <p className="truncate text-sm font-medium text-foreground">{keyLabel}</p>
                {keyData.team_id && <p className="truncate text-xs text-muted-foreground">Team: {keyData.team_id}</p>}
                {keyData.user && keyData.user !== keyLabel && (
                  <p className="truncate text-xs text-muted-foreground">User: {keyData.user}</p>
                )}
              </div>
              <div className="shrink-0 text-right">
                <p className="text-sm tabular-nums text-foreground">${formatNumberWithCommas(keyData.spend, 2)}</p>
                <p className="text-xs tabular-nums text-muted-foreground">
                  {keyData.requests.toLocaleString()} requests | {keyData.tokens.toLocaleString()} tokens
                </p>
              </div>
            </li>
          );
        })}
      </ol>
    </TopKeysPanel>
  );
};

export const ModelSection = ({
  modelName,
  metrics,
  hidePromptCachingMetrics = false,
  fetchTopApiKeys,
}: {
  modelName: string;
  metrics: ModelActivityData;
  hidePromptCachingMetrics?: boolean;
  fetchTopApiKeys?: (model: string) => Promise<ModelTopKeysResponse>;
}) => {
  return (
    <div className="grid gap-3">
      <StatStrip columns={5}>
        <StatCell>
          <Stat label="Total Requests" value={<StatValue>{metrics.total_requests.toLocaleString()}</StatValue>} />
        </StatCell>
        <StatCell>
          <Stat
            label="Total Successful Requests"
            value={<StatValue>{metrics.total_successful_requests.toLocaleString()}</StatValue>}
          />
        </StatCell>
        <StatCell>
          <Stat
            label="Total Tokens"
            value={<StatValue>{metrics.total_tokens.toLocaleString()}</StatValue>}
            hint={`${Math.round(metrics.total_tokens / metrics.total_successful_requests)} avg per successful request`}
          />
        </StatCell>
        <StatCell>
          <Stat
            label="Total Spend"
            value={<StatValue>${formatNumberWithCommas(metrics.total_spend, 2)}</StatValue>}
            hint={`$${formatNumberWithCommas(metrics.total_spend / metrics.total_successful_requests, 3)} per successful request`}
          />
        </StatCell>
        <StatCell>
          <Stat
            label="Avg Response Time"
            value={<StatValue>{formatResponseTime(modelAverageResponseTimeMs(metrics))}</StatValue>}
            hint={`over ${(metrics.total_timed_requests ?? 0).toLocaleString()} timed successful requests`}
          />
        </StatCell>
      </StatStrip>

      {fetchTopApiKeys && <ModelTopKeys modelName={modelName} fetchTopApiKeys={fetchTopApiKeys} />}

      {metrics.top_models && metrics.top_models.length > 0 && <KeyModelUsageView topModels={metrics.top_models} />}

      <Panel icon={Coins} title="Spend per day" action={<SeriesKey categories={["metrics.spend"]} colors={[BRAND]} />}>
        <BarChart
          className={CHART_CLASS}
          data={metrics.daily_data}
          index="date"
          categories={["metrics.spend"]}
          colors={[BRAND]}
          maxBarSize={28}
          valueFormatter={formatSpend}
          yAxisWidth={72}
          showLegend={false}
        />
      </Panel>

      <div className="grid gap-3 lg:grid-cols-2">
        <Panel title="Total Tokens" action={<SeriesKey categories={TOKEN_CATEGORIES} colors={TOKEN_COLORS} />}>
          <AreaChart
            className={CHART_CLASS}
            data={metrics.daily_data}
            index="date"
            categories={TOKEN_CATEGORIES}
            colors={TOKEN_COLORS}
            valueFormatter={valueFormatter}
            customTooltip={CustomTooltip}
            showLegend={false}
          />
        </Panel>

        <Panel title="Requests per day" action={<SeriesKey categories={["metrics.api_requests"]} colors={[BRAND]} />}>
          <BarChart
            className={CHART_CLASS}
            data={metrics.daily_data}
            index="date"
            categories={["metrics.api_requests"]}
            colors={[BRAND]}
            maxBarSize={28}
            valueFormatter={valueFormatter}
            customTooltip={CustomTooltip}
            showLegend={false}
          />
        </Panel>

        {(metrics.total_timed_requests ?? 0) > 0 && (
          <Panel
            icon={Timer}
            title="Avg Response Time per day"
            action={<SeriesKey categories={["metrics.avg_response_time_ms"]} colors={[BRAND]} />}
          >
            <LineChart
              className={CHART_CLASS}
              data={metrics.daily_data}
              index="date"
              categories={["metrics.avg_response_time_ms"]}
              colors={[BRAND]}
              valueFormatter={formatResponseTime}
              customTooltip={ResponseTimeTooltip}
              connectNulls={true}
              showLegend={false}
            />
          </Panel>
        )}

        <Panel
          title="Success vs Failed Requests"
          action={<SeriesKey categories={REQUEST_CATEGORIES} colors={["green", "red"]} />}
        >
          <AreaChart
            className={CHART_CLASS}
            data={metrics.daily_data}
            index="date"
            categories={REQUEST_CATEGORIES}
            colors={["green", "red"]}
            valueFormatter={valueFormatter}
            customTooltip={CustomTooltip}
            showLegend={false}
          />
        </Panel>

        {!hidePromptCachingMetrics && (
          <Panel
            icon={Database}
            title="Prompt Caching Metrics"
            action={<SeriesKey categories={CACHE_CATEGORIES} colors={["cyan", "purple"]} />}
          >
            <div className="flex flex-wrap gap-x-4 gap-y-0.5 text-xs tabular-nums text-muted-foreground">
              <p>Cache Read: {metrics.total_cache_read_input_tokens?.toLocaleString() || 0} tokens</p>
              <p>Cache Creation: {metrics.total_cache_creation_input_tokens?.toLocaleString() || 0} tokens</p>
            </div>
            <AreaChart
              className={cn(CHART_CLASS, "mt-3")}
              data={metrics.daily_data}
              index="date"
              categories={CACHE_CATEGORIES}
              colors={["cyan", "purple"]}
              valueFormatter={valueFormatter}
              customTooltip={CustomTooltip}
              showLegend={false}
            />
          </Panel>
        )}
      </div>
    </div>
  );
};

export const ModelCollapsible = ({
  defaultOpen,
  header,
  children,
  onFirstOpen,
}: {
  defaultOpen: boolean;
  header: React.ReactNode;
  children: React.ReactNode;
  onFirstOpen?: () => void;
}) => {
  const [open, setOpen] = useState(defaultOpen);
  const [everOpened, setEverOpened] = useState(defaultOpen);
  const firstOpenRef = useRef(defaultOpen);

  return (
    <Collapsible
      open={open}
      onOpenChange={(next: boolean) => {
        setOpen(next);
        if (next) {
          setEverOpened(true);
          if (!firstOpenRef.current) {
            firstOpenRef.current = true;
            onFirstOpen?.();
          }
        }
      }}
      className="border-b last:border-b-0"
    >
      <CollapsibleTrigger className="flex w-full items-center gap-2 px-4 py-3 text-left outline-none transition-colors hover:bg-muted/40 focus-visible:bg-muted/40">
        <ChevronDown
          className={`size-4 shrink-0 text-muted-foreground transition-transform ${open ? "" : "-rotate-90"}`}
        />
        {header}
      </CollapsibleTrigger>
      <CollapsibleContent keepMounted={everOpened} className="border-t bg-muted/30 p-3">
        {children}
      </CollapsibleContent>
    </Collapsible>
  );
};

export const ActivityMetrics: React.FC<ActivityMetricsProps> = ({
  modelMetrics,
  summaryMetrics,
  summaryTitle = "Overall Usage",
  hidePromptCachingMetrics = false,
  fetchTopApiKeys,
}) => {
  const modelNames = Object.keys(modelMetrics).sort((a, b) => {
    if (a === "") return 1;
    if (b === "") return -1;
    return modelMetrics[b].total_spend - modelMetrics[a].total_spend;
  });

  // Calculate total metrics across all models
  const totalMetrics = {
    total_requests: 0,
    total_successful_requests: 0,
    total_tokens: 0,
    total_spend: 0,
    total_cache_read_input_tokens: 0,
    total_cache_creation_input_tokens: 0,
    daily_data: {} as Record<
      string,
      {
        prompt_tokens: number;
        completion_tokens: number;
        total_tokens: number;
        api_requests: number;
        spend: number;
        successful_requests: number;
        failed_requests: number;
        cache_read_input_tokens: number;
        cache_creation_input_tokens: number;
      }
    >,
  };

  // Aggregate data
  Object.values(modelMetrics).forEach((model) => {
    totalMetrics.total_requests += model.total_requests;
    totalMetrics.total_successful_requests += model.total_successful_requests;
    totalMetrics.total_tokens += model.total_tokens;
    totalMetrics.total_spend += model.total_spend;
    totalMetrics.total_cache_read_input_tokens += model.total_cache_read_input_tokens || 0;
    totalMetrics.total_cache_creation_input_tokens += model.total_cache_creation_input_tokens || 0;

    // Aggregate daily data
    model.daily_data.forEach((day) => {
      if (!totalMetrics.daily_data[day.date]) {
        totalMetrics.daily_data[day.date] = {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0,
          api_requests: 0,
          spend: 0,
          successful_requests: 0,
          failed_requests: 0,
          cache_read_input_tokens: 0,
          cache_creation_input_tokens: 0,
        };
      }
      totalMetrics.daily_data[day.date].prompt_tokens += day.metrics.prompt_tokens;
      totalMetrics.daily_data[day.date].completion_tokens += day.metrics.completion_tokens;
      totalMetrics.daily_data[day.date].total_tokens += day.metrics.total_tokens;
      totalMetrics.daily_data[day.date].api_requests += day.metrics.api_requests;
      totalMetrics.daily_data[day.date].spend += day.metrics.spend;
      totalMetrics.daily_data[day.date].successful_requests += day.metrics.successful_requests;
      totalMetrics.daily_data[day.date].failed_requests += day.metrics.failed_requests;
      totalMetrics.daily_data[day.date].cache_read_input_tokens += day.metrics.cache_read_input_tokens || 0;
      totalMetrics.daily_data[day.date].cache_creation_input_tokens += day.metrics.cache_creation_input_tokens || 0;
    });
  });

  // Convert daily_data object to array and sort by date
  const sortedDailyData =
    summaryMetrics?.daily_data ??
    Object.entries(totalMetrics.daily_data)
      .map(([date, metrics]) => ({ date, metrics }))
      .sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());
  const totalRequests = summaryMetrics?.total_requests ?? totalMetrics.total_requests;
  const totalSuccessfulRequests = summaryMetrics?.total_successful_requests ?? totalMetrics.total_successful_requests;
  const totalTokens = summaryMetrics?.total_tokens ?? totalMetrics.total_tokens;
  const totalSpend = summaryMetrics?.total_spend ?? totalMetrics.total_spend;

  const maxSpend = Math.max(0, ...modelNames.map((name) => modelMetrics[name].total_spend));

  return (
    <div className="grid gap-3">
      <section className="grid gap-3">
        <h3 className="flex items-center gap-2 text-sm font-medium text-foreground">
          <Activity aria-hidden="true" className="size-4 text-muted-foreground" strokeWidth={1.75} />
          {summaryTitle}
        </h3>
        <StatStrip columns={4}>
          <StatCell>
            <Stat label="Total Requests" value={<StatValue>{totalRequests.toLocaleString()}</StatValue>} />
          </StatCell>
          <StatCell>
            <Stat
              label="Total Successful Requests"
              value={<StatValue>{totalSuccessfulRequests.toLocaleString()}</StatValue>}
            />
          </StatCell>
          <StatCell>
            <Stat label="Total Tokens" value={<StatValue>{totalTokens.toLocaleString()}</StatValue>} />
          </StatCell>
          <StatCell>
            <Stat label="Total Spend" value={<StatValue>${formatNumberWithCommas(totalSpend, 2)}</StatValue>} />
          </StatCell>
        </StatStrip>

        <div className="grid gap-3 lg:grid-cols-2">
          <Panel
            title="Total Tokens Over Time"
            action={<SeriesKey categories={TOKEN_CATEGORIES} colors={TOKEN_COLORS} />}
          >
            <AreaChart
              className={CHART_CLASS}
              data={sortedDailyData}
              index="date"
              categories={TOKEN_CATEGORIES}
              colors={TOKEN_COLORS}
              valueFormatter={valueFormatter}
              customTooltip={CustomTooltip}
              showLegend={false}
              yAxisWidth={80}
            />
          </Panel>
          <Panel
            title="Total Requests Over Time"
            action={<SeriesKey categories={REQUEST_CATEGORIES} colors={["emerald", "red"]} />}
          >
            <AreaChart
              className={CHART_CLASS}
              data={sortedDailyData}
              index="date"
              categories={REQUEST_CATEGORIES}
              colors={["emerald", "red"]}
              valueFormatter={valueFormatter}
              customTooltip={CustomTooltip}
              showLegend={false}
              yAxisWidth={80}
            />
          </Panel>
        </div>
      </section>

      {modelNames.length > 0 && (
        <div className="overflow-hidden rounded-xl border bg-card">
          {modelNames.map((modelName, index) => {
            const metrics = modelMetrics[modelName];
            const label = metrics.label || "Unknown Item";
            const provider = providerForModel(metrics.label || "");
            const avgResponse = modelAverageResponseTimeMs(metrics);
            const share = maxSpend > 0 ? (metrics.total_spend / maxSpend) * 100 : 0;
            return (
              <ModelCollapsible
                key={modelName}
                defaultOpen={modelName === modelNames[0]}
                header={
                  <div className="grid w-full grid-cols-[minmax(0,1fr)_auto] items-center gap-4">
                    <div className="flex min-w-0 items-center gap-3">
                      <ItemMark label={metrics.label || ""} color={stackedUsageColor(index)} />
                      <div className="min-w-0 flex-1">
                        <div className="flex min-w-0 items-baseline gap-2">
                          <span className="truncate text-sm font-medium text-foreground">{label}</span>
                          {provider && <span className="shrink-0 text-xs text-muted-foreground">{provider}</span>}
                        </div>
                        <div
                          aria-hidden="true"
                          className="mt-1 h-1 w-full max-w-xs overflow-hidden rounded-full bg-muted"
                        >
                          <div
                            className="h-full rounded-full opacity-80"
                            style={{ width: `${share}%`, backgroundColor: BRAND }}
                          />
                        </div>
                      </div>
                    </div>
                    <div className="flex shrink-0 items-center gap-5 text-sm tabular-nums text-muted-foreground">
                      <span className="text-foreground">${formatNumberWithCommas(metrics.total_spend, 2)}</span>
                      <span>{metrics.total_requests.toLocaleString()} requests</span>
                      {avgResponse != null && (
                        <span className="hidden md:inline">{formatResponseTime(avgResponse)} avg response</span>
                      )}
                    </div>
                  </div>
                }
              >
                <ModelSection
                  modelName={modelName || "Unknown Model"}
                  metrics={metrics}
                  hidePromptCachingMetrics={hidePromptCachingMetrics}
                  fetchTopApiKeys={fetchTopApiKeys}
                />
              </ModelCollapsible>
            );
          })}
        </div>
      )}
    </div>
  );
};

// Helper function to format key label
export const formatKeyLabel = (
  modelData: Pick<KeyMetricWithMetadata, "metadata">,
  model: string,
  teams: Team[],
): string => {
  const keyAlias = keyActivityLabel(modelData.metadata, `key-hash-${model}`);
  const teamId = modelData.metadata.team_id;
  if (teamId) {
    const teamAlias = resolveTeamAliasFromTeamID(teamId, teams);
    return teamAlias ? `${keyAlias} (team: ${teamAlias})` : `${keyAlias} (team_id: ${teamId})`;
  }
  return keyAlias;
};

// Process data function
export const processActivityData = (
  dailyActivity: { results: DailyData[] },
  key: "models" | "model_groups" | "api_keys" | "mcp_servers" | "entities",
  teams: Team[] = [],
): Record<string, ModelActivityData> => {
  const modelMetrics: Record<string, ModelActivityData> = {};

  dailyActivity.results.forEach((day) => {
    Object.entries(day.breakdown[key] || {}).forEach(([model, modelData]) => {
      if (!modelMetrics[model]) {
        modelMetrics[model] = {
          label:
            key === "api_keys"
              ? formatKeyLabel(modelData as KeyMetricWithMetadata, model, teams)
              : key === "entities"
                ? (modelData as any).metadata?.agent_name || (modelData as any).metadata?.team_alias || model
                : model,
          ...(key === "api_keys" ? { key_metadata: (modelData as KeyMetricWithMetadata).metadata } : {}),
          total_requests: 0,
          total_successful_requests: 0,
          total_failed_requests: 0,
          total_tokens: 0,
          prompt_tokens: 0,
          completion_tokens: 0,
          total_spend: 0,
          total_cache_read_input_tokens: 0,
          total_cache_creation_input_tokens: 0,
          total_response_time_ms: 0,
          total_timed_requests: 0,
          top_models: [],
          daily_data: [],
        };
      }
      const dayResponseTimeMs = modelData.metrics.total_response_time_ms || 0;
      const dayTimedRequests = modelData.metrics.timed_requests || 0;
      // Update totals
      modelMetrics[model].total_requests += modelData.metrics.api_requests;
      modelMetrics[model].prompt_tokens += modelData.metrics.prompt_tokens;
      modelMetrics[model].completion_tokens += modelData.metrics.completion_tokens;
      modelMetrics[model].total_tokens += modelData.metrics.total_tokens;
      modelMetrics[model].total_spend += modelData.metrics.spend;
      modelMetrics[model].total_successful_requests += modelData.metrics.successful_requests;
      modelMetrics[model].total_failed_requests += modelData.metrics.failed_requests;
      modelMetrics[model].total_cache_read_input_tokens += modelData.metrics.cache_read_input_tokens || 0;
      modelMetrics[model].total_cache_creation_input_tokens += modelData.metrics.cache_creation_input_tokens || 0;
      modelMetrics[model].total_response_time_ms =
        (modelMetrics[model].total_response_time_ms ?? 0) + dayResponseTimeMs;
      modelMetrics[model].total_timed_requests = (modelMetrics[model].total_timed_requests ?? 0) + dayTimedRequests;

      // Add daily data
      modelMetrics[model].daily_data.push({
        date: day.date,
        metrics: {
          prompt_tokens: modelData.metrics.prompt_tokens,
          completion_tokens: modelData.metrics.completion_tokens,
          total_tokens: modelData.metrics.total_tokens,
          api_requests: modelData.metrics.api_requests,
          spend: modelData.metrics.spend,
          successful_requests: modelData.metrics.successful_requests,
          failed_requests: modelData.metrics.failed_requests,
          cache_read_input_tokens: modelData.metrics.cache_read_input_tokens || 0,
          cache_creation_input_tokens: modelData.metrics.cache_creation_input_tokens || 0,
          avg_response_time_ms: averageResponseTimeMs(dayResponseTimeMs, dayTimedRequests),
        },
      });
    });
  });

  // Process Model breakdowns for each API key (only when key is 'api_keys')
  if (key === "api_keys") {
    Object.entries(modelMetrics).forEach(([apiKeyHash, _]) => {
      const modelBreakdown: Record<string, TopModelData> = {};

      // Aggregate Model data for this key across all days
      // We need to look in breakdown.models[model].api_key_breakdown[apiKeyHash]
      dailyActivity.results.forEach((day) => {
        Object.entries(day.breakdown.models || {}).forEach(([modelName, modelData]) => {
          if (modelData && "api_key_breakdown" in modelData) {
            const keyDataForModel = modelData.api_key_breakdown?.[apiKeyHash];
            if (keyDataForModel) {
              if (!modelBreakdown[modelName]) {
                modelBreakdown[modelName] = {
                  model: modelName,
                  spend: 0,
                  requests: 0,
                  successful_requests: 0,
                  failed_requests: 0,
                  tokens: 0,
                };
              }

              modelBreakdown[modelName].spend += keyDataForModel.metrics.spend;
              modelBreakdown[modelName].requests += keyDataForModel.metrics.api_requests;
              modelBreakdown[modelName].successful_requests += keyDataForModel.metrics.successful_requests || 0;
              modelBreakdown[modelName].failed_requests += keyDataForModel.metrics.failed_requests || 0;
              modelBreakdown[modelName].tokens += keyDataForModel.metrics.total_tokens;
            }
          }
        });
      });

      // Sort by spend
      modelMetrics[apiKeyHash].top_models = Object.values(modelBreakdown).sort((a, b) => b.spend - a.spend);
    });
  }

  // Sort daily data
  Object.values(modelMetrics).forEach((metrics) => {
    metrics.daily_data.sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());
  });

  return modelMetrics;
};
