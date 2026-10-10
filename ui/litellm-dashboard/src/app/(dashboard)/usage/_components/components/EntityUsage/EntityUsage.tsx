import useTeams from "@/app/(dashboard)/hooks/useTeams";
import { StackedUsageChart, type StackedUsageScale } from "@/components/shared/charts";
import { DataTable } from "@/components/shared/DataTable";
import { getProviderSpend, getTopAgents, getTopAPIKeys, getTopModels } from "./entityUsageAggregations";
import { buildCostBreakdownTiles, buildSummaryTiles, hasFlatCost, type SummaryTile } from "./entityUsageSummary";
import { MoneyCell } from "@/components/shared/table_cells";
import { hasCapability, type Capability } from "@/utils/capabilities";
import type { DateRangePickerValue } from "@/components/shared/date_picker_types";
import { Bot, Boxes, ChevronDown, ChevronRight, ExternalLink, Info, KeyRound, Layers, Server } from "lucide-react";
import type { ColumnDef } from "@tanstack/react-table";
import { Alert, AlertDescription } from "@/components/shared/Alert";
import { ChartLoader } from "@/components/shared/chart_loader";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { cn } from "@/lib/cva.config";
import React, { type ReactNode, useCallback, useMemo, useState } from "react";
import TeamMultiSelect from "@/components/common_components/team_multi_select";
import UserDropdown from "@/components/common_components/UserDropdown";
import { ActivityMetrics, processActivityData } from "@/components/activity_metrics";
import { UsageExportHeader } from "@/components/EntityUsageExport";
import type { EntityType } from "@/components/EntityUsageExport/types";
import { useAggregatedDailyActivity } from "../../hooks/useAggregatedDailyActivity";
import { ENTITY_API } from "./entityFetchFns";
import {
  EMPTY_DAILY_ACTIVITY_METADATA,
  toDailyData,
  type DailyActivityRequest,
} from "@/components/UsagePage/dailyActivityApi";
import { keyDetailFromResponse, overallUsageMetrics } from "@/components/UsagePage/keyActivityData";
import type { DailyData, EntityMetricWithMetadata } from "@/components/UsagePage/types";
import EndpointUsage from "../EndpointUsage/EndpointUsage";
import ModelViewToggle, { ModelViewType } from "../ModelViewToggle";
import TopKeyView from "@/components/UsagePage/components/EntityUsage/TopKeyView";
import KeyActivityPanel from "@/components/UsagePage/components/KeyActivityPanel";
import { BreakdownControls, Leaderboard, useBreakdown, type BreakdownState } from "../overview/BreakdownChart";
import {
  bucketSeries,
  bucketTotals,
  labelForDate,
  dailyTotals,
  formatMetricValue,
  type Granularity,
  type Series,
  type UsageMetric,
} from "../overview/overviewData";
import { Panel, Segmented, Sparkline, Stat } from "../overview/Primitives";
import TopModelView from "./TopModelView";
import TeamUserSpendCard from "./TeamUserSpendCard";
import { ProviderSpendBreakdown } from "./SpendByProvider";

const BRAND = "#2b3fd6";
const FLAT_COST_SERIES = "Flat cost";
const FLAT_COST_COLOR = "#8b5cf6";
const METRIC_TITLE: Record<UsageMetric, string> = { spend: "Spend", tokens: "Tokens", requests: "Requests" };
/** Summary tiles that carry a sparkline, keyed by tile title, valued by the dailyTotals field. */
const TILE_TRENDS: Readonly<Record<string, string>> = {
  "Total Spend": "spend",
  "Total Cost": "spend",
  "Total Requests": "requests",
  "Total Tokens": "tokens",
};
const GRANULARITY_OPTIONS = [
  { value: "day", label: "Daily" },
  { value: "week", label: "Weekly" },
] as const satisfies readonly { value: Granularity; label: string }[];
const SCALE_OPTIONS = [
  { value: "linear", label: "Linear" },
  { value: "log", label: "Log" },
] as const satisfies readonly { value: StackedUsageScale; label: string }[];
const QUIET_HEADER = { headerClassName: "font-normal" };
const FLAT_COST_KEY = "flat_cost";

/** Stacks reserved-capacity flat cost on top of the per-model spend, so each bar is the day's full cost. */
const withFlatCost = (series: Series, results: readonly DailyData[]): Series => {
  const flatByDate = new Map(results.map((day) => [day.date, day.metrics.flat_cost ?? 0]));
  return {
    data: series.data.map((day) => ({ ...day, [FLAT_COST_KEY]: flatByDate.get(day.date) ?? 0 })),
    keys: [...series.keys, FLAT_COST_KEY],
    labels: [...series.labels, FLAT_COST_SERIES],
    colors: [...series.colors, FLAT_COST_COLOR],
  };
};

function ShareBar({ value, max }: { value: number; max: number }) {
  return (
    <div aria-hidden="true" className="h-1 w-full min-w-16 overflow-hidden rounded-full bg-muted">
      <div
        className="h-full rounded-full opacity-80"
        style={{ width: `${max > 0 ? (value / max) * 100 : 0}%`, backgroundColor: BRAND }}
      />
    </div>
  );
}

function StatCell({ children, trend, className }: { children: ReactNode; trend?: ReactNode; className?: string }) {
  return (
    <div className={cn("flex min-w-0 flex-col justify-between gap-3 bg-card px-4 pt-3.5 pb-3", className)}>
      <div className="min-w-0">{children}</div>
      {trend}
    </div>
  );
}

export interface EntityList {
  label: string;
  value: string;
}

interface EntityUsageProps {
  accessToken: string | null;
  entityType: EntityType;
  entityId?: string | null;
  userID: string | null;
  userRole: string | null;
  entityList: EntityList[] | null;
  premiumUser: boolean;
  dateValue: DateRangePickerValue;
  isOrgAdmin?: boolean;
}

const ENTITY_CAPABILITIES: Partial<Record<EntityType, Capability>> = {
  organization: "viewOrganizationUsage",
  agent: "viewAgentUsage",
};

const EntityUsage: React.FC<EntityUsageProps> = ({
  accessToken,
  entityType,
  entityId,
  entityList,
  userRole,
  dateValue,
  isOrgAdmin = false,
}) => {
  const { teams } = useTeams();
  const teamList = useMemo(() => teams ?? [], [teams]);
  const [selectedTags, setSelectedTags] = useState<string[]>([]);
  const [modelViewType, setModelViewType] = useState<ModelViewType>("groups");
  const [topKeysLimit, setTopKeysLimit] = useState<number>(5);
  const [topModelsLimit, setTopModelsLimit] = useState<number>(5);
  const [topAgentsLimit, setTopAgentsLimit] = useState<number>(5);
  const [showCostBreakdown, setShowCostBreakdown] = useState(false);

  const startTime = useMemo(() => (dateValue.from ? new Date(dateValue.from) : null), [dateValue.from]);
  const endTime = useMemo(() => (dateValue.to ? new Date(dateValue.to) : null), [dateValue.to]);
  const api = ENTITY_API[entityType];
  const entityCapability = ENTITY_CAPABILITIES[entityType];
  const canViewEntity = entityCapability === undefined || hasCapability(userRole, entityCapability, isOrgAdmin);
  const showAgentBreakdown = entityType === "team" && hasCapability(userRole, "viewAgentUsage");
  const hasRequestWindow = !!accessToken && !!startTime && !!endTime;
  const enabled = hasRequestWindow && canViewEntity;

  const request = useMemo<DailyActivityRequest | null>(
    () =>
      hasRequestWindow
        ? {
            accessToken: accessToken as string,
            startTime: startTime as Date,
            endTime: endTime as Date,
            entityIds: selectedTags.length > 0 ? selectedTags : null,
          }
        : null,
    [hasRequestWindow, accessToken, startTime, endTime, selectedTags],
  );
  const agentRequest = useMemo<DailyActivityRequest | null>(
    () =>
      hasRequestWindow
        ? {
            accessToken: accessToken as string,
            startTime: startTime as Date,
            endTime: endTime as Date,
            entityIds: null,
          }
        : null,
    [hasRequestWindow, accessToken, startTime, endTime],
  );

  const {
    data: spendDataRaw,
    loading,
    failed,
  } = useAggregatedDailyActivity({
    fetch: () => api.aggregated(request as DailyActivityRequest),
    enabled: enabled && request !== null,
    deps: [entityType, accessToken, startTime, endTime, selectedTags],
  });

  const spendData = useMemo(
    () => ({
      results: toDailyData(spendDataRaw),
      metadata: spendDataRaw.metadata ?? EMPTY_DAILY_ACTIVITY_METADATA,
    }),
    [spendDataRaw],
  );
  const summaryMetrics = useMemo(() => overallUsageMetrics(spendData.results, spendData.metadata), [spendData]);

  const {
    data: agentSpendDataRaw,
    loading: agentLoading,
    failed: agentFailed,
  } = useAggregatedDailyActivity({
    fetch: () => ENTITY_API.agent.aggregated(agentRequest as DailyActivityRequest),
    enabled: enabled && showAgentBreakdown && agentRequest !== null,
    deps: [accessToken, startTime, endTime, showAgentBreakdown],
  });

  const agentSpendData = useMemo(
    () => ({
      results: toDailyData(agentSpendDataRaw),
      metadata: agentSpendDataRaw.metadata ?? EMPTY_DAILY_ACTIVITY_METADATA,
    }),
    [agentSpendDataRaw],
  );

  const fetchTopApiKeys = useCallback(
    (model: string) => api.modelTopKeys(request as DailyActivityRequest, model, modelViewType === "groups"),
    [api, request, modelViewType],
  );
  const searchKeys = useCallback(
    (query: string) => (request === null ? Promise.resolve({ api_keys: [] }) : api.searchKeys(request, query)),
    [api, request],
  );
  const fetchKeyPage = useCallback(
    (offset: number, limit: number) =>
      request === null
        ? Promise.resolve({ api_keys: [], total_api_keys: 0, offset, limit })
        : api.keyPage(request, offset, limit),
    [api, request],
  );
  const fetchKeyDetail = useCallback(
    (apiKey: string) =>
      request === null
        ? Promise.resolve(undefined)
        : api
            .aggregated({ ...request, apiKey, apiKeyLimit: 1 })
            .then((response) => keyDetailFromResponse(response, apiKey, teamList)),
    [api, request, teamList],
  );

  const modelBreakdownKey = modelViewType === "groups" ? "model_groups" : "models";
  const modelMetrics = processActivityData(spendData, modelBreakdownKey, teamList);
  const agentMetrics = showAgentBreakdown ? processActivityData(agentSpendData, "entities", teamList) : {};

  const getAllTags = () => {
    if (entityList) {
      return entityList;
    }
  };

  const getEntityLabel = (entity: string, metadata?: Record<string, any>): string => {
    if (entityList) {
      const entityItem = entityList.find((item) => item.value === entity);
      if (entityItem) {
        return entityItem.label;
      }
    }
    // Fallback to team_alias for backward compatibility
    if (metadata?.team_alias) {
      return metadata.team_alias;
    }
    // Resolve user_id to email/alias so the Spend Per User chart never shows a raw UUID
    // when an email is on file (the entityList is paginated and may miss spenders)
    if (metadata?.user_email) {
      return metadata.user_email;
    }
    if (metadata?.user_alias) {
      return metadata.user_alias;
    }
    return entity;
  };

  const filterDataByTags = (data: EntityMetricWithMetadata[]) => {
    if (selectedTags.length === 0) return data;
    return data.filter((item) => selectedTags.includes(item.metadata.id));
  };

  const getEntityBreakdown = () => {
    const entitySpend: { [key: string]: EntityMetricWithMetadata } = {};
    spendData.results.forEach((day) => {
      Object.entries(day.breakdown.entities || {}).forEach(([entity, data]) => {
        if (!entitySpend[entity]) {
          entitySpend[entity] = {
            metrics: {
              spend: 0,
              prompt_tokens: 0,
              completion_tokens: 0,
              total_tokens: 0,
              api_requests: 0,
              successful_requests: 0,
              failed_requests: 0,
              cache_read_input_tokens: 0,
              cache_creation_input_tokens: 0,
            },
            metadata: {
              alias: getEntityLabel(entity, data.metadata as any),
              id: entity,
            },
          };
        }
        entitySpend[entity].metrics.spend += data.metrics.spend;
        entitySpend[entity].metrics.api_requests += data.metrics.api_requests;
        entitySpend[entity].metrics.successful_requests += data.metrics.successful_requests;
        entitySpend[entity].metrics.failed_requests += data.metrics.failed_requests;
        entitySpend[entity].metrics.total_tokens += data.metrics.total_tokens;
      });
    });

    const result = Object.values(entitySpend).sort((a, b) => b.metrics.spend - a.metrics.spend);

    return filterDataByTags(result);
  };
  const entityRows = getEntityBreakdown().filter((entity) => entity.metrics.spend > 0);
  const maxEntitySpend = Math.max(...entityRows.map((entity) => entity.metrics.spend), 0);

  const [breakdown, setBreakdown] = useState<BreakdownState>({ metric: "spend", dimension: "model_groups" });
  const [granularity, setGranularity] = useState<Granularity>("day");
  const [scale, setScale] = useState<StackedUsageScale>("linear");
  const breakdownDimension = modelBreakdownKey;
  const { series: dailySeries, ranking } = useBreakdown(
    spendData.results,
    { metric: breakdown.metric, dimension: breakdownDimension },
    8,
  );

  const getFilterLabel = (entityType: string) => {
    return `Filter by ${entityType}`;
  };

  const getFilterPlaceholder = (entityType: string) => {
    return `Select ${entityType} to filter...`;
  };

  const entityFilterSlots: Partial<Record<EntityType, ReactNode>> = {
    team: <TeamMultiSelect value={selectedTags} onChange={setSelectedTags} />,
    user: (
      <UserDropdown value={selectedTags[0] ?? null} onChange={(userId) => setSelectedTags(userId ? [userId] : [])} />
    ),
  };
  const filterSlot = entityFilterSlots[entityType];

  const capitalizedEntityLabel = entityType.charAt(0).toUpperCase() + entityType.slice(1);
  const showFlatCost = entityType === "team" && hasFlatCost(spendData.metadata);
  const userSpendTeamIds = useMemo(
    () =>
      selectedTags.length > 0
        ? selectedTags
        : (teams ?? []).map((team) => team.team_id).filter((id) => id !== "litellm-dashboard"),
    [selectedTags, teams],
  );
  const providerSpend = useMemo(() => getProviderSpend(spendData.results), [spendData.results]);
  const dailyTrend = useMemo(() => dailyTotals(spendData.results), [spendData.results]);
  const stackFlatCost = showFlatCost && breakdown.metric === "spend";
  const chartSeries = useMemo(() => {
    const series = stackFlatCost ? withFlatCost(dailySeries, spendData.results) : dailySeries;
    return bucketSeries(series, granularity);
  }, [stackFlatCost, dailySeries, spendData.results, granularity]);
  const chartTotals = useMemo(() => bucketTotals(chartSeries), [chartSeries]);

  const entityBreakdownColumns = useMemo<ColumnDef<EntityMetricWithMetadata>[]>(
    () => [
      {
        header: capitalizedEntityLabel,
        accessorKey: "metadata.alias",
        meta: QUIET_HEADER,
        cell: ({ row }) => <span className="font-medium text-foreground">{row.original.metadata.alias}</span>,
      },
      {
        header: "Share",
        id: "share",
        meta: { ...QUIET_HEADER, className: "w-40" },
        cell: ({ row }) => <ShareBar value={row.original.metrics.spend} max={maxEntitySpend} />,
      },
      {
        header: "Spend",
        accessorKey: "metrics.spend",
        meta: { numeric: true, ...QUIET_HEADER },
        cell: ({ row }) => <MoneyCell value={row.original.metrics.spend} decimals={4} />,
      },
      {
        header: "Successful",
        accessorKey: "metrics.successful_requests",
        meta: { numeric: true, className: "text-success", ...QUIET_HEADER },
        cell: ({ row }) => row.original.metrics.successful_requests.toLocaleString(),
      },
      {
        header: "Failed",
        accessorKey: "metrics.failed_requests",
        meta: { numeric: true, className: "text-destructive", ...QUIET_HEADER },
        cell: ({ row }) => row.original.metrics.failed_requests.toLocaleString(),
      },
      {
        header: "Tokens",
        accessorKey: "metrics.total_tokens",
        meta: { numeric: true, ...QUIET_HEADER },
        cell: ({ row }) => row.original.metrics.total_tokens.toLocaleString(),
      },
    ],
    [capitalizedEntityLabel, maxEntitySpend],
  );

  const chev = "size-3 text-muted-foreground";
  const expandIcon = showCostBreakdown ? <ChevronDown className={chev} /> : <ChevronRight className={chev} />;

  const tileLabel = ({ title, tooltip, expandable }: SummaryTile) => (
    <>
      <span>{title}</span>
      {tooltip ? (
        <Tooltip>
          <TooltipTrigger render={<Info className="size-3.5 text-muted-foreground hover:text-foreground" />} />
          <TooltipContent>{tooltip}</TooltipContent>
        </Tooltip>
      ) : null}
      {expandable ? expandIcon : null}
    </>
  );

  const renderSummaryTile = (tile: SummaryTile, index: number) => {
    const trendKey = TILE_TRENDS[tile.title];
    const stat = <Stat label={tileLabel(tile)} value={<span className={tile.className}>{tile.value}</span>} />;
    const trend = trendKey ? (
      <Sparkline data={dailyTrend} dataKey={trendKey} color={BRAND} className="h-12" />
    ) : undefined;
    const span = index === 0 ? "col-span-2 lg:col-span-1" : undefined;
    if (!tile.expandable) {
      return (
        <StatCell key={tile.title} trend={trend} className={span}>
          {stat}
        </StatCell>
      );
    }
    return (
      <div
        key={tile.title}
        role="button"
        tabIndex={0}
        aria-expanded={showCostBreakdown}
        className={cn("flex cursor-pointer transition-colors [&>div]:flex-1 hover:[&>div]:bg-accent/50", span)}
        onClick={() => setShowCostBreakdown(!showCostBreakdown)}
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            setShowCostBreakdown(!showCostBreakdown);
          }
        }}
      >
        <StatCell trend={trend}>{stat}</StatCell>
      </div>
    );
  };

  const breakdownTiles = showFlatCost && showCostBreakdown ? buildCostBreakdownTiles(spendData.metadata) : [];
  const summaryTiles = buildSummaryTiles(spendData.metadata, showFlatCost);

  const modelViewTitle = modelViewType === "groups" ? "Top Public Model Names" : "Top Litellm Models";
  const chartFormat = (value: number) => formatMetricValue(value, breakdown.metric);

  const costPanel = loading ? (
    <ChartLoader />
  ) : (
    <div className="grid gap-3">
      <section className="grid gap-2">
        <h2 className="text-sm font-medium text-foreground">{capitalizedEntityLabel} Spend Overview</h2>
        <div className="grid grid-cols-2 gap-px overflow-hidden rounded-xl border bg-border lg:grid-cols-5">
          {summaryTiles.map(renderSummaryTile)}
        </div>
        {breakdownTiles.length > 0 && (
          <div className="grid grid-cols-2 gap-px overflow-hidden rounded-xl border bg-border">
            {breakdownTiles.map((tile) => (
              <StatCell key={tile.title}>
                <Stat label={tileLabel(tile)} value={<span className={tile.className}>{tile.value}</span>} />
              </StatCell>
            ))}
          </div>
        )}
      </section>

      <section className="rounded-xl border bg-card">
        <header className="flex flex-wrap items-start justify-between gap-3 px-5 pt-4">
          <div>
            <h3 className="text-sm font-medium text-foreground">
              {METRIC_TITLE[breakdown.metric]} by {modelViewType === "groups" ? "model" : "deployment"}
            </h3>
            <p className="text-xs text-muted-foreground">
              {granularity === "day" ? "Daily" : "Weekly"} {breakdown.metric}, top 8 stacked
              {stackFlatCost ? ", flat cost on top" : ""}
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <BreakdownControls state={breakdown} onChange={setBreakdown} showDimension={false} />
            <Segmented
              label="Bucket size"
              value={granularity}
              options={GRANULARITY_OPTIONS}
              onChange={setGranularity}
            />
            <Segmented label="Scale" value={scale} options={SCALE_OPTIONS} onChange={setScale} />
          </div>
        </header>
        <div className="px-3 pt-4 pb-2">
          <StackedUsageChart
            data={chartSeries.data}
            series={chartSeries.keys}
            labels={chartSeries.labels}
            colors={chartSeries.colors}
            xKey="date"
            xLabel={(date) => labelForDate(chartSeries, date)}
            scale={scale}
            format={chartFormat}
            totalFor={(date) => chartTotals.get(date)}
            className="h-[320px]"
          />
        </div>
        <div className="border-t px-5 py-3">
          <Leaderboard
            ranking={ranking}
            series={dailySeries}
            metric={breakdown.metric}
            dimension={breakdownDimension}
            columns={2}
            limit={8}
          />
        </div>
      </section>

      <Panel
        icon={Layers}
        title={`Spend Per ${capitalizedEntityLabel}`}
        action={
          <a
            href="https://docs.litellm.ai/docs/proxy/enterprise#spend-tracking"
            className="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
          >
            Track cost per {entityType}
            <ExternalLink aria-hidden="true" className="size-3" />
          </a>
        }
      >
        <DataTable
          columns={entityBreakdownColumns}
          data={entityRows}
          getRowId={(row) => row.metadata.id}
          maxBodyHeight={280}
          noDataMessage={`No ${entityType} spend data`}
          size="compact"
        />
      </Panel>

      {entityType === "team" && (
        <TeamUserSpendCard
          accessToken={accessToken}
          startTime={startTime}
          endTime={endTime}
          teamIds={userSpendTeamIds}
        />
      )}

      <div className="grid gap-3 lg:grid-cols-2">
        <Panel icon={KeyRound} title="Top Virtual Keys">
          <TopKeyView
            topKeys={getTopAPIKeys(spendData.results, topKeysLimit)}
            teams={null}
            showTags={entityType === "tag"}
            topKeysLimit={topKeysLimit}
            setTopKeysLimit={setTopKeysLimit}
          />
        </Panel>
        <Panel
          icon={Boxes}
          title={entityType === "agent" ? "Top Agents" : modelViewTitle}
          action={<ModelViewToggle value={modelViewType} onChange={setModelViewType} />}
        >
          <TopModelView
            topModels={getTopModels(spendData.results, modelBreakdownKey, topModelsLimit)}
            topModelsLimit={topModelsLimit}
            setTopModelsLimit={setTopModelsLimit}
          />
        </Panel>
      </div>

      {showAgentBreakdown && (
        <Panel icon={Bot} title="Top Agents Driving Spend">
          {agentLoading ? (
            <ChartLoader />
          ) : (
            <TopModelView
              topModels={getTopAgents(agentSpendData.results, topAgentsLimit)}
              topModelsLimit={topAgentsLimit}
              setTopModelsLimit={setTopAgentsLimit}
            />
          )}
        </Panel>
      )}

      <Panel icon={Server} title="Provider Usage">
        <ProviderSpendBreakdown rows={providerSpend} />
      </Panel>
    </div>
  );

  const tabs: readonly { key: string; label: string; content: ReactNode }[] = [
    { key: "cost", label: "Cost", content: costPanel },
    {
      key: "models",
      label: entityType === "agent" ? "Request / Token Consumption" : "Model Activity",
      content: (
        <>
          <div className="mb-3 flex justify-end">
            <ModelViewToggle value={modelViewType} onChange={setModelViewType} />
          </div>
          <ActivityMetrics
            modelMetrics={modelMetrics}
            hidePromptCachingMetrics={entityType === "agent"}
            fetchTopApiKeys={request ? fetchTopApiKeys : undefined}
          />
        </>
      ),
    },
    ...(showAgentBreakdown
      ? [
          {
            key: "agents",
            label: "Agent Activity",
            content: <ActivityMetrics modelMetrics={agentMetrics} />,
          },
        ]
      : []),
    {
      key: "keys",
      label: "Key Activity",
      content: (
        <KeyActivityPanel
          summary={summaryMetrics}
          summaryLoading={loading}
          fetchKeyPage={fetchKeyPage}
          fetchKeyDetail={fetchKeyDetail}
          searchKeys={searchKeys}
          teams={teamList}
          hidePromptCachingMetrics={entityType === "agent"}
        />
      ),
    },
    { key: "endpoints", label: "Endpoint Activity", content: <EndpointUsage userSpendData={spendData} /> },
  ];

  return (
    <div className="relative grid w-full gap-3">
      {failed && (
        <Alert variant="error">
          <AlertDescription className="text-inherit">
            Fetching spend data failed, so the totals below may be empty rather than final. Reload the page to try
            again.
          </AlertDescription>
        </Alert>
      )}
      {showAgentBreakdown && agentFailed && (
        <Alert variant="error">
          <AlertDescription className="text-inherit">
            Fetching agent data failed, so the totals below may be empty rather than final. Reload the page to try
            again.
          </AlertDescription>
        </Alert>
      )}
      <div className="max-w-full [&>div:first-child]:mb-0 [&>div:first-child_label]:sr-only">
        <UsageExportHeader
          dateValue={dateValue}
          entityType={entityType}
          onExport={(exportType, format) =>
            request
              ? api.exportRows(request, exportType, format)
              : Promise.reject(new Error("Select a date range to export"))
          }
          showFilters={filterSlot === undefined && entityList !== null}
          filterSlot={filterSlot}
          filterLabel={getFilterLabel(entityType)}
          filterPlaceholder={getFilterPlaceholder(entityType)}
          selectedFilters={selectedTags}
          onFiltersChange={setSelectedTags}
          filterOptions={getAllTags() || undefined}
          teams={teamList}
        />
      </div>
      <Tabs defaultValue={tabs[0].key} className="gap-3">
        <TabsList variant="line">
          {tabs.map(({ key, label }) => (
            <TabsTrigger key={key} value={key} className="flex-none px-3">
              {label}
            </TabsTrigger>
          ))}
        </TabsList>
        {tabs.map(({ key, content }) => (
          <TabsContent key={key} value={key} keepMounted>
            {content}
          </TabsContent>
        ))}
      </Tabs>
    </div>
  );
};

export default EntityUsage;
