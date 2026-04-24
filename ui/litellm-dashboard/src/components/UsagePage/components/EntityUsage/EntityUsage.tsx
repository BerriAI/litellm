import useTeams from "@/app/(dashboard)/hooks/useTeams";
import { formatNumberWithCommas } from "@/utils/dataUtils";
// eslint-disable-next-line litellm-ui/no-banned-ui-imports
import { BarChart, DateRangePickerValue, DonutChart } from "@tremor/react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Loader2 } from "lucide-react";
import React, { useMemo, useState } from "react";
import TeamMultiSelect from "../../../common_components/team_multi_select";
import { ActivityMetrics, processActivityData } from "../../../activity_metrics";
import { UsageExportHeader } from "../../../EntityUsageExport";
import type { EntityType } from "../../../EntityUsageExport/types";
import {
  agentDailyActivityCall,
  customerDailyActivityCall,
  organizationDailyActivityCall,
  tagDailyActivityCall,
  teamDailyActivityCall,
  userDailyActivityCall,
} from "../../../networking";
import { getProviderLogoAndName } from "../../../provider_info_helpers";
import { usePaginatedDailyActivity } from "../../hooks/usePaginatedDailyActivity";
import { BreakdownMetrics, DailyData, EntityMetricWithMetadata, KeyMetricWithMetadata, TagUsage } from "../../types";
import { valueFormatterSpend } from "../../utils/value_formatters";
import EndpointUsage from "../EndpointUsage/EndpointUsage";
import TopKeyView from "./TopKeyView";
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

interface ExtendedDailyData extends DailyData {
  breakdown: BreakdownMetrics;
}

interface EntitySpendData {
  results: ExtendedDailyData[];
  metadata: {
    total_spend: number;
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
}

const ENTITY_FETCH_FNS: Record<EntityType, (...args: any[]) => Promise<any>> = {
  tag: tagDailyActivityCall,
  team: teamDailyActivityCall,
  organization: organizationDailyActivityCall,
  customer: customerDailyActivityCall,
  agent: agentDailyActivityCall,
  user: userDailyActivityCall,
};

const EntityUsage: React.FC<EntityUsageProps> = ({ accessToken, entityType, entityId, entityList, dateValue }) => {
  const { teams } = useTeams();
  const [selectedTags, setSelectedTags] = useState<string[]>([]);
  const [topKeysLimit, setTopKeysLimit] = useState<number>(5);
  const [topModelsLimit, setTopModelsLimit] = useState<number>(5);
  const [topAgentsLimit, setTopAgentsLimit] = useState<number>(5);

  const startTime = useMemo(() => (dateValue.from ? new Date(dateValue.from) : null), [dateValue.from]);
  const endTime = useMemo(() => (dateValue.to ? new Date(dateValue.to) : null), [dateValue.to]);

  const entityFilterArg = useMemo(() => {
    if (entityType === "user") return selectedTags.length > 0 ? selectedTags[0] : null;
    return selectedTags.length > 0 ? selectedTags : null;
  }, [entityType, selectedTags]);

  const fetchFn = ENTITY_FETCH_FNS[entityType];
  const enabled = !!accessToken && !!startTime && !!endTime;

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
    enabled: enabled && entityType === "team",
  });

  const agentSpendData = agentSpendDataRaw as unknown as EntitySpendData;

  const modelMetrics = processActivityData(spendData, "models", teams || []);
  const keyMetrics = processActivityData(spendData, "api_keys", teams || []);
  const agentMetrics = entityType === "team" ? processActivityData(agentSpendData, "entities", teams || []) : {};

  const getTopModels = () => {
    const modelSpend: { [key: string]: any } = {};
    spendData.results.forEach((day) => {
      Object.entries(day.breakdown.models || {}).forEach(([model, metrics]) => {
        if (!modelSpend[model]) {
          modelSpend[model] = {
            spend: 0,
            requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            tokens: 0,
          };
        }
        try {
          modelSpend[model].spend += metrics.metrics.spend;
        } catch (e) {
          console.error(`Error adding spend for ${model}: ${e}, got metrics: ${JSON.stringify(metrics)}`);
        }
        modelSpend[model].requests += metrics.metrics.api_requests;
        modelSpend[model].successful_requests += metrics.metrics.successful_requests;
        modelSpend[model].failed_requests += metrics.metrics.failed_requests;
        modelSpend[model].tokens += metrics.metrics.total_tokens;
      });
    });

    return Object.entries(modelSpend)
      .map(([model, metrics]) => ({
        key: model,
        ...metrics,
      }))
      .sort((a, b) => b.spend - a.spend)
      .slice(0, topModelsLimit);
  };

  const getTopAgents = () => {
    const agentSpend: { [key: string]: any } = {};
    agentSpendData.results.forEach((day) => {
      Object.entries(day.breakdown.entities || {}).forEach(([agentId, data]) => {
        if (!agentSpend[agentId]) {
          agentSpend[agentId] = {
            spend: 0,
            requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            tokens: 0,
            agent_name: (data.metadata as any)?.agent_name || agentId,
          };
        }
        agentSpend[agentId].spend += data.metrics.spend;
        agentSpend[agentId].requests += data.metrics.api_requests;
        agentSpend[agentId].successful_requests += data.metrics.successful_requests;
        agentSpend[agentId].failed_requests += data.metrics.failed_requests;
        agentSpend[agentId].tokens += data.metrics.total_tokens;
      });
    });

    return Object.entries(agentSpend)
      .map(([agentId, metrics]) => ({
        key: metrics.agent_name,
        ...metrics,
      }))
      .sort((a, b) => b.spend - a.spend)
      .slice(0, topAgentsLimit);
  };

  const getTopAPIKeys = () => {
    console.log("debugTags", { spendData });
    const keySpend: { [key: string]: KeyMetricWithMetadata } = {};
    spendData.results.forEach((day) => {
      const { breakdown } = day;
      const { entities } = breakdown;
      console.log("debugTags", { entities });
      const tagDictionary = Object.keys(entities).reduce((acc: { [key: string]: TagUsage[] }, entity) => {
        const { api_key_breakdown } = entities[entity];
        Object.keys(api_key_breakdown).forEach((key) => {
          const tagUsage = { tag: entity, usage: api_key_breakdown[key].metrics.spend };
          if (acc[key]) {
            acc[key].push(tagUsage);
          } else {
            acc[key] = [tagUsage];
          }
        });
        return acc;
      }, {});
      console.log("debugTags", { tagDictionary });
      Object.entries(day.breakdown.api_keys || {}).forEach(([key, metrics]) => {
        if (!keySpend[key]) {
          keySpend[key] = {
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
              key_alias: metrics.metadata.key_alias,
              team_id: metrics.metadata.team_id || null,
              tags: tagDictionary[key] || [],
            },
          };
          console.log("debugTags", { keySpend });
        }
        keySpend[key].metrics.spend += metrics.metrics.spend;
        keySpend[key].metrics.prompt_tokens += metrics.metrics.prompt_tokens;
        keySpend[key].metrics.completion_tokens += metrics.metrics.completion_tokens;
        keySpend[key].metrics.total_tokens += metrics.metrics.total_tokens;
        keySpend[key].metrics.api_requests += metrics.metrics.api_requests;
        keySpend[key].metrics.successful_requests += metrics.metrics.successful_requests;
        keySpend[key].metrics.failed_requests += metrics.metrics.failed_requests;
        keySpend[key].metrics.cache_read_input_tokens += metrics.metrics.cache_read_input_tokens || 0;
        keySpend[key].metrics.cache_creation_input_tokens += metrics.metrics.cache_creation_input_tokens || 0;
      });
    });

    return Object.entries(keySpend)
      .map(([api_key, metrics]) => ({
        api_key,
        key_alias: metrics.metadata.key_alias || "-", // Using truncated key as alias
        tags: metrics.metadata.tags || "-",
        spend: metrics.metrics.spend,
      }))
      .sort((a, b) => b.spend - a.spend)
      .slice(0, topKeysLimit);
  };

  const getProviderSpend = () => {
    const providerSpend: { [key: string]: any } = {};
    spendData.results.forEach((day) => {
      Object.entries(day.breakdown.providers || {}).forEach(([provider, metrics]) => {
        if (!providerSpend[provider]) {
          providerSpend[provider] = {
            provider,
            spend: 0,
            requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            tokens: 0,
          };
        }
        try {
          providerSpend[provider].spend += metrics.metrics.spend;
          providerSpend[provider].requests += metrics.metrics.api_requests;
          providerSpend[provider].successful_requests += metrics.metrics.successful_requests;
          providerSpend[provider].failed_requests += metrics.metrics.failed_requests;
          providerSpend[provider].tokens += metrics.metrics.total_tokens;
        } catch (e) {
          console.error(`Error processing provider ${provider}: ${e}`);
        }
      });
    });

    return Object.values(providerSpend)
      .filter((provider) => provider.spend > 0)
      .sort((a, b) => b.spend - a.spend);
  };

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

  const capitalizedEntityLabel = entityType.charAt(0).toUpperCase() + entityType.slice(1);

  return (
    <div style={{ width: "100%" }} className="relative">
      {isFetchingMore && (
        <Alert className="mb-2">
          <AlertDescription>
            <div className="flex items-center justify-between">
              <span>
                <Loader2 className="animate-spin mr-2 inline h-4 w-4" />
                Currently fetching spend data: fetched {progress.currentPage} / {progress.totalPages} pages. Charts will
                update periodically as data loads. Moving off of this page will stop and reset this. To continue using
                the UI in the meantime,{" "}
                <a href={window.location.href} target="_blank" rel="noopener noreferrer">
                  open a new tab
                </a>
                .
              </span>
              <Button variant="destructive" onClick={cancel}>
                Stop
              </Button>
            </div>
          </AlertDescription>
        </Alert>
      )}
      {cancelled && (
        <Alert className="mb-2">
          <AlertDescription>
            Showing partial data ({progress.currentPage}/{progress.totalPages} pages loaded)
          </AlertDescription>
        </Alert>
      )}
      {agentIsFetchingMore && entityType === "team" && (
        <Alert className="mb-2">
          <AlertDescription>
            <div className="flex items-center justify-between">
              <span>
                <Loader2 className="animate-spin mr-2 inline h-4 w-4" />
                Currently fetching agent data: fetched {agentProgress.currentPage} / {agentProgress.totalPages} pages.
                Charts will update periodically as data loads. Moving off of this page will stop and reset this. To
                continue using the UI in the meantime,{" "}
                <a href={window.location.href} target="_blank" rel="noopener noreferrer">
                  open a new tab
                </a>
                .
              </span>
              <Button variant="destructive" onClick={agentCancel}>
                Stop
              </Button>
            </div>
          </AlertDescription>
        </Alert>
      )}
      {agentCancelled && entityType === "team" && (
        <Alert className="mb-2">
          <AlertDescription>
            Showing partial agent data ({agentProgress.currentPage}/{agentProgress.totalPages} pages loaded)
          </AlertDescription>
        </Alert>
      )}
      {entityType === "team" && (
        <div className="mb-4">
          <p className="mb-2 text-sm">Filter by team</p>
          <TeamMultiSelect value={selectedTags} onChange={setSelectedTags} />
        </div>
      )}
      <UsageExportHeader
        dateValue={dateValue}
        entityType={entityType}
        spendData={spendData}
        showFilters={entityType !== "team" && entityList !== null && entityList.length > 0}
        filterLabel={getFilterLabel(entityType)}
        filterPlaceholder={getFilterPlaceholder(entityType)}
        selectedFilters={selectedTags}
        onFiltersChange={setSelectedTags}
        filterOptions={getAllTags() || undefined}
        filterMode={entityType === "user" ? "single" : "multiple"}
        teams={teams || []}
      />
      <Tabs defaultValue="cost" className="mt-1">
        <TabsList>
          <TabsTrigger value="cost">Cost</TabsTrigger>
          <TabsTrigger value="model">
            {entityType === "agent" ? "Request / Token Consumption" : "Model Activity"}
          </TabsTrigger>
          {entityType === "team" && <TabsTrigger value="agent">Agent Activity</TabsTrigger>}
          <TabsTrigger value="key">Key Activity</TabsTrigger>
          <TabsTrigger value="endpoint">Endpoint Activity</TabsTrigger>
        </TabsList>
        <TabsContent value="cost">
          <div className="grid grid-cols-2 gap-2 w-full">
            <div className="col-span-2">
              <Card className="p-6">
                <h3 className="text-lg font-semibold">{capitalizedEntityLabel} Spend Overview</h3>
                <div className="grid grid-cols-5 gap-4 mt-4">
                  <Card className="p-6">
                    <h3 className="text-lg font-semibold">Total Spend</h3>
                    <p className="text-2xl font-bold mt-2">
                      ${formatNumberWithCommas(spendData.metadata.total_spend, 2)}
                    </p>
                  </Card>
                  <Card className="p-6">
                    <h3 className="text-lg font-semibold">Total Requests</h3>
                    <p className="text-2xl font-bold mt-2">
                      {spendData.metadata.total_api_requests.toLocaleString()}
                    </p>
                  </Card>
                  <Card className="p-6">
                    <h3 className="text-lg font-semibold">Successful Requests</h3>
                    <p className="text-2xl font-bold mt-2 text-green-600">
                      {spendData.metadata.total_successful_requests.toLocaleString()}
                    </p>
                  </Card>
                  <Card className="p-6">
                    <h3 className="text-lg font-semibold">Failed Requests</h3>
                    <p className="text-2xl font-bold mt-2 text-red-600">
                      {spendData.metadata.total_failed_requests.toLocaleString()}
                    </p>
                  </Card>
                  <Card className="p-6">
                    <h3 className="text-lg font-semibold">Total Tokens</h3>
                    <p className="text-2xl font-bold mt-2">
                      {spendData.metadata.total_tokens.toLocaleString()}
                    </p>
                  </Card>
                </div>
              </Card>
            </div>

            {/* Daily Spend Chart */}
            <div className="col-span-2">
              <Card className="p-6">
                <h3 className="text-lg font-semibold">Daily Spend</h3>
                <BarChart
                  data={[...spendData.results].sort(
                    (a, b) => new Date(a.date).getTime() - new Date(b.date).getTime(),
                  )}
                  index="date"
                  categories={["metrics.spend"]}
                  colors={["cyan"]}
                  valueFormatter={valueFormatterSpend}
                  yAxisWidth={100}
                  showLegend={false}
                  customTooltip={({ payload, active }) => {
                    if (!active || !payload?.[0]) return null;
                    const data = payload[0].payload;
                    const entityCount = Object.keys(data.breakdown.entities || {}).length;
                    return (
                      <div className="bg-background p-4 shadow-lg rounded-lg border border-border">
                        <p className="font-bold">{data.date}</p>
                        <p className="text-cyan-500">Total Spend: ${formatNumberWithCommas(data.metrics.spend, 2)}</p>
                        <p className="text-muted-foreground">Total Requests: {data.metrics.api_requests}</p>
                        <p className="text-muted-foreground">Successful: {data.metrics.successful_requests}</p>
                        <p className="text-muted-foreground">Failed: {data.metrics.failed_requests}</p>
                        <p className="text-muted-foreground">Total Tokens: {data.metrics.total_tokens}</p>
                        <p className="text-muted-foreground">
                          Total {capitalizedEntityLabel}s: {entityCount}
                        </p>
                        <div className="mt-2 border-t border-border pt-2">
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
              </Card>
            </div>

            {/* Entity Breakdown Section */}
            <div className="col-span-2">
              <Card className="p-6">
                <div className="flex flex-col space-y-4">
                  <div className="flex flex-col space-y-2">
                    <h3 className="text-lg font-semibold">Spend Per {capitalizedEntityLabel}</h3>
                    <p className="text-xs text-muted-foreground">Showing Top 5 by Spend</p>
                    <div className="flex items-center text-sm text-muted-foreground">
                      <span>Get Started by Tracking cost per {capitalizedEntityLabel} </span>
                      <a
                        href="https://docs.litellm.ai/docs/proxy/enterprise#spend-tracking"
                        className="text-primary hover:underline ml-1"
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
                            <div className="bg-background p-4 shadow-lg rounded-lg border border-border">
                              <p className="font-bold">{data.metadata.alias}</p>
                              <p className="text-cyan-500">Spend: ${formatNumberWithCommas(data.metrics.spend, 4)}</p>
                              <p className="text-muted-foreground">
                                Requests: {data.metrics.api_requests.toLocaleString()}
                              </p>
                              <p className="text-green-600">
                                Successful: {data.metrics.successful_requests.toLocaleString()}
                              </p>
                              <p className="text-red-600">Failed: {data.metrics.failed_requests.toLocaleString()}</p>
                              <p className="text-muted-foreground">
                                Tokens: {data.metrics.total_tokens.toLocaleString()}
                              </p>
                            </div>
                          );
                        }}
                      />
                    </div>
                    <div>
                      <div className="h-52 overflow-y-auto">
                        <Table>
                          <TableHeader>
                            <TableRow>
                              <TableHead>{capitalizedEntityLabel}</TableHead>
                              <TableHead>Spend</TableHead>
                              <TableHead className="text-green-600">Successful</TableHead>
                              <TableHead className="text-red-600">Failed</TableHead>
                              <TableHead>Tokens</TableHead>
                            </TableRow>
                          </TableHeader>
                          <TableBody>
                            {getEntityBreakdown()
                              .filter((entity) => entity.metrics.spend > 0)
                              .map((entity) => (
                                <TableRow key={entity.metadata.id}>
                                  <TableCell>{entity.metadata.alias}</TableCell>
                                  <TableCell>${formatNumberWithCommas(entity.metrics.spend, 4)}</TableCell>
                                  <TableCell className="text-green-600">
                                    {entity.metrics.successful_requests.toLocaleString()}
                                  </TableCell>
                                  <TableCell className="text-red-600">
                                    {entity.metrics.failed_requests.toLocaleString()}
                                  </TableCell>
                                  <TableCell>{entity.metrics.total_tokens.toLocaleString()}</TableCell>
                                </TableRow>
                              ))}
                          </TableBody>
                        </Table>
                      </div>
                    </div>
                  </div>
                </div>
              </Card>
            </div>

            {/* Top API Keys */}
            <div>
              <Card className="p-6">
                <h3 className="text-lg font-semibold">Top Virtual Keys</h3>
                <TopKeyView
                  topKeys={getTopAPIKeys()}
                  teams={null}
                  showTags={entityType === "tag"}
                  topKeysLimit={topKeysLimit}
                  setTopKeysLimit={setTopKeysLimit}
                />
              </Card>
            </div>

            {/* Top Models */}
            <div>
              <Card className="p-6">
                <h3 className="text-lg font-semibold">{entityType === "agent" ? "Top Agents" : "Top Models"}</h3>
                <TopModelView
                  topModels={getTopModels()}
                  topModelsLimit={topModelsLimit}
                  setTopModelsLimit={setTopModelsLimit}
                />
              </Card>
            </div>

            {/* Top Agents - only for team entity type */}
            {entityType === "team" && (
              <div className="col-span-2">
                <Card className="p-6">
                  <h3 className="text-lg font-semibold">Top Agents Driving Spend</h3>
                  <TopModelView
                    topModels={getTopAgents()}
                    topModelsLimit={topAgentsLimit}
                    setTopModelsLimit={setTopAgentsLimit}
                  />
                </Card>
              </div>
            )}

            {/* Spend by Provider */}
            <div className="col-span-2">
              <Card className="p-6">
                <div className="flex flex-col space-y-4">
                  <h3 className="text-lg font-semibold">Provider Usage</h3>
                  <div className="grid grid-cols-2">
                    <div>
                      <DonutChart
                        className="mt-4 h-40"
                        data={getProviderSpend()}
                        index="provider"
                        category="spend"
                        valueFormatter={(value) => `$${formatNumberWithCommas(value, 2)}`}
                        colors={["cyan", "blue", "indigo", "violet", "purple"]}
                      />
                    </div>
                    <div>
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead>Provider</TableHead>
                            <TableHead>Spend</TableHead>
                            <TableHead className="text-green-600">Successful</TableHead>
                            <TableHead className="text-red-600">Failed</TableHead>
                            <TableHead>Tokens</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {getProviderSpend().map((provider) => (
                            <TableRow key={provider.provider}>
                              <TableCell>
                                <div className="flex items-center space-x-2">
                                  {provider.provider && (
                                    <img
                                      src={getProviderLogoAndName(provider.provider).logo}
                                      alt={`${provider.provider} logo`}
                                      className="w-4 h-4"
                                      onError={(e) => {
                                        const target = e.target as HTMLImageElement;
                                        const parent = target.parentElement;
                                        if (parent) {
                                          const fallbackDiv = document.createElement("div");
                                          fallbackDiv.className =
                                            "w-4 h-4 rounded-full bg-muted flex items-center justify-center text-xs";
                                          fallbackDiv.textContent = provider.provider?.charAt(0) || "-";
                                          parent.replaceChild(fallbackDiv, target);
                                        }
                                      }}
                                    />
                                  )}
                                  <span>{provider.provider}</span>
                                </div>
                              </TableCell>
                              <TableCell>${formatNumberWithCommas(provider.spend, 2)}</TableCell>
                              <TableCell className="text-green-600">
                                {provider.successful_requests.toLocaleString()}
                              </TableCell>
                              <TableCell className="text-red-600">
                                {provider.failed_requests.toLocaleString()}
                              </TableCell>
                              <TableCell>{provider.tokens.toLocaleString()}</TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </div>
                  </div>
                </div>
              </Card>
            </div>
          </div>
        </TabsContent>
        <TabsContent value="model">
          <ActivityMetrics modelMetrics={modelMetrics} hidePromptCachingMetrics={entityType === "agent"} />
        </TabsContent>
        {entityType === "team" && (
          <TabsContent value="agent">
            <ActivityMetrics modelMetrics={agentMetrics} />
          </TabsContent>
        )}
        <TabsContent value="key">
          <ActivityMetrics modelMetrics={keyMetrics} hidePromptCachingMetrics={entityType === "agent"} />
        </TabsContent>
        <TabsContent value="endpoint">
          <EndpointUsage userSpendData={spendData} />
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default EntityUsage;
