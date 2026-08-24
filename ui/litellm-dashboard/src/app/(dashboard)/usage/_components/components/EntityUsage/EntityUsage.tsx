import useTeams from "@/app/(dashboard)/hooks/useTeams";
import { BarChart, DonutChart } from "@/components/shared/charts";
import { DataTable } from "@/components/shared/DataTable";
import {
  getProviderSpend,
  getTopAgents,
  getTopAPIKeys,
  getTopModels,
  type ExtendedDailyData,
  type ProviderSpendRow,
} from "./entityUsageAggregations";
import { buildCostBreakdownTiles, buildSummaryTiles, hasFlatCost, type SummaryTile } from "./entityUsageSummary";
import { MoneyCell } from "@/components/shared/table_cells";
import { Card as ShadcnCard, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { hasCapability, type Capability } from "@/utils/capabilities";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import type { DateRangePickerValue } from "@/components/shared/date_picker_types";
import { ChevronDown, ChevronRight, Info } from "lucide-react";
import type { ColumnDef } from "@tanstack/react-table";
import PaginationStatusAlerts from "@/components/shared/PaginationStatusAlerts";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import React, { type ReactNode, useMemo, useState } from "react";
import TeamMultiSelect from "@/components/common_components/team_multi_select";
import UserDropdown from "@/components/common_components/UserDropdown";
import { ActivityMetrics, processActivityData } from "@/components/activity_metrics";
import { UsageExportHeader } from "@/components/EntityUsageExport";
import type { EntityType } from "@/components/EntityUsageExport/types";
import {
  agentDailyActivityCall,
  customerDailyActivityCall,
  organizationDailyActivityCall,
  tagDailyActivityCall,
  teamDailyActivityAggregatedCall,
  teamDailyActivityCall,
  userDailyActivityCall,
} from "@/components/networking";
import { Logo } from "@/components/molecules/logo/Logo";
import { usePaginatedDailyActivity } from "../../hooks/usePaginatedDailyActivity";
import { EntityMetricWithMetadata } from "@/components/UsagePage/types";
import { valueFormatterSpend } from "@/components/UsagePage/utils/value_formatters";
import EndpointUsage from "../EndpointUsage/EndpointUsage";
import ModelViewToggle, { ModelViewType } from "../ModelViewToggle";
import TopKeyView from "@/components/UsagePage/components/EntityUsage/TopKeyView";
import TopModelView from "./TopModelView";

interface EntityMetrics {
  metrics: {
    spend: number;
    prompt_tokens: number;
    completion_tokens: number;
    cache_read_input_tokens: number;
    cache_creation_input_tokens: number;
    total_tokens: number;
    successful_requests: number;
    failed_requests: number;
    api_requests: number;
  };
  metadata: Record<string, any>;
}

interface EntitySpendData {
  results: ExtendedDailyData[];
  metadata: {
    total_spend: number;
    total_flat_cost?: number;
    total_api_requests: number;
    total_successful_requests: number;
    total_failed_requests: number;
    total_tokens: number;
  };
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

const ENTITY_FETCH_FNS: Record<EntityType, (...args: any[]) => Promise<any>> = {
  tag: tagDailyActivityCall,
  team: teamDailyActivityCall,
  organization: organizationDailyActivityCall,
  customer: customerDailyActivityCall,
  agent: agentDailyActivityCall,
  user: userDailyActivityCall,
};

// Single-shot endpoints returning the whole range in one response; entity types
// without one fall back to page-draining the paginated endpoint.
const ENTITY_AGGREGATED_FETCH_FNS: Partial<Record<EntityType, (...args: any[]) => Promise<any>>> = {
  team: teamDailyActivityAggregatedCall,
};

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
  const [selectedTags, setSelectedTags] = useState<string[]>([]);
  const [modelViewType, setModelViewType] = useState<ModelViewType>("groups");
  const [topKeysLimit, setTopKeysLimit] = useState<number>(5);
  const [topModelsLimit, setTopModelsLimit] = useState<number>(5);
  const [topAgentsLimit, setTopAgentsLimit] = useState<number>(5);
  const [showCostBreakdown, setShowCostBreakdown] = useState(false);

  const startTime = useMemo(() => (dateValue.from ? new Date(dateValue.from) : null), [dateValue.from]);
  const endTime = useMemo(() => (dateValue.to ? new Date(dateValue.to) : null), [dateValue.to]);

  const entityFilterArg = useMemo(() => {
    if (entityType === "user") return selectedTags.length > 0 ? selectedTags[0] : null;
    return selectedTags.length > 0 ? selectedTags : null;
  }, [entityType, selectedTags]);

  const fetchFn = ENTITY_FETCH_FNS[entityType];
  const aggregatedFetchFn = ENTITY_AGGREGATED_FETCH_FNS[entityType];
  const entityCapability = ENTITY_CAPABILITIES[entityType];
  const canViewEntity = entityCapability === undefined || hasCapability(userRole, entityCapability, isOrgAdmin);
  const showAgentBreakdown = entityType === "team" && hasCapability(userRole, "viewAgentUsage");
  const hasRequestWindow = !!accessToken && !!startTime && !!endTime;
  const enabled = hasRequestWindow && canViewEntity;

  const {
    data: spendDataRaw,
    isFetchingMore,
    progress,
    cancelled,
    cancel,
  } = usePaginatedDailyActivity({
    fetchFn,
    args: [accessToken, startTime, endTime, entityFilterArg],
    enabled,
    aggregatedFetchFn,
  });

  const spendData = spendDataRaw as unknown as EntitySpendData;

  const {
    data: agentSpendDataRaw,
    isFetchingMore: agentIsFetchingMore,
    progress: agentProgress,
    cancelled: agentCancelled,
    cancel: agentCancel,
  } = usePaginatedDailyActivity({
    fetchFn: agentDailyActivityCall,
    args: [accessToken, startTime, endTime, null],
    enabled: enabled && showAgentBreakdown,
  });

  const agentSpendData = agentSpendDataRaw as unknown as EntitySpendData;

  const modelBreakdownKey = modelViewType === "groups" ? "model_groups" : "models";
  const modelMetrics = processActivityData(spendData, modelBreakdownKey, teams || []);
  const keyMetrics = processActivityData(spendData, "api_keys", teams || []);
  const agentMetrics = showAgentBreakdown ? processActivityData(agentSpendData, "entities", teams || []) : {};

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

  const getProcessedEntityBreakdownForChart = () => {
    const data = getEntityBreakdown();
    const topEntities = data.slice(0, 5);
    return topEntities.map((e) => ({
      ...e,
      metadata: {
        ...e.metadata,
        alias_display:
          e.metadata.alias && e.metadata.alias.length > 15 ? `${e.metadata.alias.slice(0, 15)}...` : e.metadata.alias,
      },
    }));
  };

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
  const providerSpend = useMemo(() => getProviderSpend(spendData.results), [spendData.results]);
  const entityBreakdownColumns = useMemo<ColumnDef<EntityMetricWithMetadata>[]>(
    () => [
      {
        header: capitalizedEntityLabel,
        accessorKey: "metadata.alias",
        cell: ({ row }) => row.original.metadata.alias,
      },
      {
        header: "Spend",
        accessorKey: "metrics.spend",
        meta: { numeric: true },
        cell: ({ row }) => <MoneyCell value={row.original.metrics.spend} decimals={4} />,
      },
      {
        header: "Successful",
        accessorKey: "metrics.successful_requests",
        meta: { numeric: true, className: "text-success" },
        cell: ({ row }) => row.original.metrics.successful_requests.toLocaleString(),
      },
      {
        header: "Failed",
        accessorKey: "metrics.failed_requests",
        meta: { numeric: true, className: "text-destructive" },
        cell: ({ row }) => row.original.metrics.failed_requests.toLocaleString(),
      },
      {
        header: "Tokens",
        accessorKey: "metrics.total_tokens",
        meta: { numeric: true },
        cell: ({ row }) => row.original.metrics.total_tokens.toLocaleString(),
      },
    ],
    [capitalizedEntityLabel],
  );
  const providerSpendColumns = useMemo<ColumnDef<ProviderSpendRow>[]>(
    () => [
      {
        header: "Provider",
        accessorKey: "provider",
        cell: ({ row }) => (
          <div className="flex items-center space-x-2">
            {row.original.provider && <Logo provider={row.original.provider} className="size-4" />}
            <span>{row.original.provider}</span>
          </div>
        ),
      },
      {
        header: "Spend",
        accessorKey: "spend",
        meta: { numeric: true },
        cell: ({ row }) => <MoneyCell value={row.original.spend} decimals={2} />,
      },
      {
        header: "Successful",
        accessorKey: "successful_requests",
        meta: { numeric: true, className: "text-success" },
        cell: ({ row }) => row.original.successful_requests.toLocaleString(),
      },
      {
        header: "Failed",
        accessorKey: "failed_requests",
        meta: { numeric: true, className: "text-destructive" },
        cell: ({ row }) => row.original.failed_requests.toLocaleString(),
      },
      {
        header: "Tokens",
        accessorKey: "tokens",
        meta: { numeric: true },
        cell: ({ row }) => row.original.tokens.toLocaleString(),
      },
    ],
    [],
  );

  const chev = "size-3 text-muted-foreground";
  const expandIcon = showCostBreakdown ? <ChevronDown className={chev} /> : <ChevronRight className={chev} />;

  const renderSummaryTile = ({ title, value, className, tooltip, expandable }: SummaryTile) => (
    <ShadcnCard
      key={title}
      className={expandable ? "cursor-pointer hover:bg-accent transition-colors" : undefined}
      onClick={expandable ? () => setShowCostBreakdown(!showCostBreakdown) : undefined}
    >
      <CardContent>
        <div className="flex items-center gap-2">
          <h3 className="text-lg font-medium text-foreground">{title}</h3>
          {tooltip ? (
            <Tooltip>
              <TooltipTrigger render={<Info className="size-4 text-muted-foreground hover:text-foreground" />} />
              <TooltipContent>{tooltip}</TooltipContent>
            </Tooltip>
          ) : null}
          {expandable ? expandIcon : null}
        </div>
        <p className={`text-2xl font-bold mt-2 ${className ?? ""}`}>{value}</p>
      </CardContent>
    </ShadcnCard>
  );

  const breakdownTiles = showFlatCost && showCostBreakdown ? buildCostBreakdownTiles(spendData.metadata) : [];
  const summaryTiles = [...buildSummaryTiles(spendData.metadata, showFlatCost), ...breakdownTiles];

  const modelViewTitle = modelViewType === "groups" ? "Top Public Model Names" : "Top Litellm Models";

  const costPanel = (
    <div className="grid grid-cols-2 gap-2 w-full">
      <div className="col-span-2">
        <ShadcnCard>
          <CardContent>
            <h3 className="text-lg font-medium text-foreground">{capitalizedEntityLabel} Spend Overview</h3>
            <div className="grid grid-cols-5 gap-4 mt-4">{summaryTiles.map(renderSummaryTile)}</div>
          </CardContent>
        </ShadcnCard>
      </div>

      {/* Daily Spend Chart */}
      <div className="col-span-2">
        <ShadcnCard>
          <CardHeader>
            <CardTitle className="text-base font-semibold">Daily Spend</CardTitle>
          </CardHeader>
          <CardContent>
            <BarChart
              data={[...spendData.results]
                .sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime())
                .map((row) => ({
                  ...row,
                  "Request cost": row.metrics.spend ?? 0,
                  "Flat cost": row.metrics.flat_cost ?? 0,
                }))}
              index="date"
              categories={showFlatCost ? ["Request cost", "Flat cost"] : ["metrics.spend"]}
              colors={showFlatCost ? ["cyan", "violet"] : ["cyan"]}
              stack={showFlatCost}
              valueFormatter={valueFormatterSpend}
              yAxisWidth={100}
              showLegend={showFlatCost}
              customTooltip={({ payload, active }) => {
                if (!active || !payload?.[0]) return null;
                const data = payload[0].payload;
                const entityCount = Object.keys(data.breakdown.entities || {}).length;
                const requestSpend = data.metrics.spend ?? 0;
                const flatCost = data.metrics.flat_cost ?? 0;
                return (
                  <div className="bg-card p-4 shadow-lg rounded-lg border">
                    <p className="font-bold">{data.date}</p>
                    {showFlatCost ? (
                      <>
                        <p className="text-info">Request cost: ${formatNumberWithCommas(requestSpend, 2)}</p>
                        <p className="text-violet-500">Flat cost: ${formatNumberWithCommas(flatCost, 2)}</p>
                        <p className="font-semibold">
                          Total cost: ${formatNumberWithCommas(requestSpend + flatCost, 2)}
                        </p>
                      </>
                    ) : (
                      <p className="text-info">Total Spend: ${formatNumberWithCommas(data.metrics.spend, 2)}</p>
                    )}
                    <p className="text-muted-foreground">Total Requests: {data.metrics.api_requests}</p>
                    <p className="text-muted-foreground">Successful: {data.metrics.successful_requests}</p>
                    <p className="text-muted-foreground">Failed: {data.metrics.failed_requests}</p>
                    <p className="text-muted-foreground">Total Tokens: {data.metrics.total_tokens}</p>
                    <p className="text-muted-foreground">
                      Total {capitalizedEntityLabel}s: {entityCount}
                    </p>
                    <div className="mt-2 border-t pt-2">
                      <p className="font-semibold">Spend by {capitalizedEntityLabel}:</p>
                      {Object.entries(data.breakdown.entities || {})
                        .sort(([, a], [, b]) => {
                          const spendA = (a as EntityMetrics).metrics.spend;
                          const spendB = (b as EntityMetrics).metrics.spend;
                          return spendB - spendA;
                        })
                        .slice(0, 5)
                        .map(([entity, entityData]) => {
                          const metrics = entityData as EntityMetrics;
                          return (
                            <p key={entity} className="text-sm text-muted-foreground">
                              {getEntityLabel(entity, metrics.metadata)}: $
                              {formatNumberWithCommas(metrics.metrics.spend, 2)}
                            </p>
                          );
                        })}
                      {entityCount > 5 && (
                        <p className="text-sm text-muted-foreground italic">...and {entityCount - 5} more</p>
                      )}
                    </div>
                  </div>
                );
              }}
            />
          </CardContent>
        </ShadcnCard>
      </div>

      {/* Entity Breakdown Section */}
      <div className="col-span-2">
        <ShadcnCard>
          <CardContent className="flex flex-col space-y-4">
            <div className="flex flex-col space-y-2">
              <h3 className="text-lg font-medium text-foreground">Spend Per {capitalizedEntityLabel}</h3>
              <p className="text-xs text-muted-foreground">Showing Top 5 by Spend</p>
              <div className="flex items-center text-sm text-muted-foreground">
                <span>Get Started by Tracking cost per {capitalizedEntityLabel} </span>
                <a
                  href="https://docs.litellm.ai/docs/proxy/enterprise#spend-tracking"
                  className="text-info hover:text-info/80 ml-1"
                >
                  here
                </a>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-6">
              <div>
                <BarChart
                  className="mt-4 h-52"
                  data={getProcessedEntityBreakdownForChart()}
                  index="metadata.alias_display"
                  categories={["metrics.spend"]}
                  colors={["cyan"]}
                  valueFormatter={valueFormatterSpend}
                  layout="vertical"
                  showLegend={false}
                  yAxisWidth={150}
                  customTooltip={({ payload, active }) => {
                    if (!active || !payload?.[0]) return null;
                    const data = payload[0].payload;
                    return (
                      <div className="bg-card p-4 shadow-lg rounded-lg border">
                        <p className="font-bold">{data.metadata.alias}</p>
                        <p className="text-info">Spend: ${formatNumberWithCommas(data.metrics.spend, 4)}</p>
                        <p className="text-muted-foreground">Requests: {data.metrics.api_requests.toLocaleString()}</p>
                        <p className="text-success">Successful: {data.metrics.successful_requests.toLocaleString()}</p>
                        <p className="text-destructive">Failed: {data.metrics.failed_requests.toLocaleString()}</p>
                        <p className="text-muted-foreground">Tokens: {data.metrics.total_tokens.toLocaleString()}</p>
                      </div>
                    );
                  }}
                />
              </div>
              <div>
                <DataTable
                  columns={entityBreakdownColumns}
                  data={getEntityBreakdown().filter((entity) => entity.metrics.spend > 0)}
                  getRowId={(row) => row.metadata.id}
                  maxBodyHeight={208}
                  noDataMessage={`No ${entityType} spend data`}
                  size="compact"
                />
              </div>
            </div>
          </CardContent>
        </ShadcnCard>
      </div>

      {/* Top API Keys */}
      <div>
        <ShadcnCard>
          <CardContent>
            <h3 className="text-lg font-medium text-foreground">Top Virtual Keys</h3>
            <TopKeyView
              topKeys={getTopAPIKeys(spendData.results, topKeysLimit)}
              teams={null}
              showTags={entityType === "tag"}
              topKeysLimit={topKeysLimit}
              setTopKeysLimit={setTopKeysLimit}
            />
          </CardContent>
        </ShadcnCard>
      </div>

      {/* Top Models */}
      <div>
        <ShadcnCard>
          <CardContent>
            <div className="flex justify-between items-center">
              <h3 className="text-lg font-medium text-foreground">
                {entityType === "agent" ? "Top Agents" : modelViewTitle}
              </h3>
              <ModelViewToggle value={modelViewType} onChange={setModelViewType} />
            </div>
            <TopModelView
              topModels={getTopModels(spendData.results, modelBreakdownKey, topModelsLimit)}
              topModelsLimit={topModelsLimit}
              setTopModelsLimit={setTopModelsLimit}
            />
          </CardContent>
        </ShadcnCard>
      </div>

      {showAgentBreakdown && (
        <div className="col-span-2">
          <ShadcnCard>
            <CardContent>
              <h3 className="text-lg font-medium text-foreground">Top Agents Driving Spend</h3>
              <TopModelView
                topModels={getTopAgents(agentSpendData.results, topAgentsLimit)}
                topModelsLimit={topAgentsLimit}
                setTopModelsLimit={setTopAgentsLimit}
              />
            </CardContent>
          </ShadcnCard>
        </div>
      )}

      {/* Spend by Provider */}
      <div className="col-span-2">
        <ShadcnCard>
          <CardContent className="flex flex-col space-y-4">
            <h3 className="text-lg font-medium text-foreground">Provider Usage</h3>
            <div className="grid grid-cols-2">
              <div>
                <DonutChart
                  className="mt-4 h-40"
                  data={providerSpend}
                  index="provider"
                  category="spend"
                  valueFormatter={(value) => `$${formatNumberWithCommas(value, 2)}`}
                  colors={["cyan", "blue", "indigo", "violet", "purple"]}
                  showLabel
                  startAngle={90}
                  endAngle={-270}
                />
              </div>
              <div>
                <DataTable
                  columns={providerSpendColumns}
                  data={providerSpend}
                  getRowId={(row) => row.provider}
                  noDataMessage="No provider usage data"
                  size="compact"
                />
              </div>
            </div>
          </CardContent>
        </ShadcnCard>
      </div>
    </div>
  );

  const tabs: readonly { key: string; label: string; content: ReactNode }[] = [
    { key: "cost", label: "Cost", content: costPanel },
    {
      key: "models",
      label: entityType === "agent" ? "Request / Token Consumption" : "Model Activity",
      content: (
        <>
          <div className="flex justify-end mt-2 mb-4">
            <ModelViewToggle value={modelViewType} onChange={setModelViewType} />
          </div>
          <ActivityMetrics modelMetrics={modelMetrics} hidePromptCachingMetrics={entityType === "agent"} />
        </>
      ),
    },
    ...(showAgentBreakdown
      ? [{ key: "agents", label: "Agent Activity", content: <ActivityMetrics modelMetrics={agentMetrics} /> }]
      : []),
    {
      key: "keys",
      label: "Key Activity",
      content: <ActivityMetrics modelMetrics={keyMetrics} hidePromptCachingMetrics={entityType === "agent"} />,
    },
    { key: "endpoints", label: "Endpoint Activity", content: <EndpointUsage userSpendData={spendData} /> },
  ];

  return (
    <div style={{ width: "100%" }} className="relative">
      <PaginationStatusAlerts
        isFetchingMore={isFetchingMore}
        cancelled={cancelled}
        progress={progress}
        cancel={cancel}
      />
      {showAgentBreakdown && (
        <PaginationStatusAlerts
          isFetchingMore={agentIsFetchingMore}
          cancelled={agentCancelled}
          progress={agentProgress}
          cancel={agentCancel}
          subject="agent data"
        />
      )}
      <UsageExportHeader
        dateValue={dateValue}
        entityType={entityType}
        spendData={spendData}
        showFilters={filterSlot === undefined && entityList !== null && entityList.length > 0}
        filterSlot={filterSlot}
        filterLabel={getFilterLabel(entityType)}
        filterPlaceholder={getFilterPlaceholder(entityType)}
        selectedFilters={selectedTags}
        onFiltersChange={setSelectedTags}
        filterOptions={getAllTags() || undefined}
        teams={teams || []}
      />
      <Tabs defaultValue={tabs[0].key}>
        <TabsList className="mt-1">
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
