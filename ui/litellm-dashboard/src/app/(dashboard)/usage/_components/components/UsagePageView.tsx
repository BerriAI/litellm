/**
 * New Usage Page
 *
 * Uses the new `/user/daily/activity` endpoint to get daily activity data for a user.
 *
 * Works at 1m+ spend logs, by querying an aggregate table instead.
 */

import { ChevronDown, ChevronRight, Download, Info, Sparkles, X } from "lucide-react";
import type { DateRangePickerValue } from "@/components/shared/date_picker_types";
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { BarChart } from "@/components/shared/charts";
import { Alert, AlertAction, AlertDescription, AlertTitle } from "@/components/shared/Alert";
import PaginationStatusAlerts from "@/components/shared/PaginationStatusAlerts";
import { Button } from "@/components/ui/button";
import { Card as ShadcnCard, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";

import { useAgents } from "@/app/(dashboard)/hooks/agents/useAgents";
import { useCustomers } from "@/app/(dashboard)/hooks/customers/useCustomers";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import useIsOrgAdmin from "@/app/(dashboard)/hooks/useIsOrgAdmin";
import { useCurrentUser } from "@/app/(dashboard)/hooks/users/useCurrentUser";
import { hasCapability } from "@/utils/capabilities";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import { all_admin_roles, internalUserRoles } from "@/utils/roles";
import { ActivityMetrics, processActivityData } from "@/components/activity_metrics";
import CloudZeroExportModal from "@/components/cloudzero_export_modal";
import UserDropdown from "@/components/common_components/UserDropdown";
import EntityUsageExportModal from "@/components/EntityUsageExport";
import { Team } from "@/components/key_team_helpers/key_list";
import {
  gatewayDailyActivityCall,
  Organization,
  tagListCall,
  userDailyActivityAggregatedCall,
  userDailyActivityCall,
} from "@/components/networking";
import AdvancedDatePicker from "@/components/shared/advanced_date_picker";
import { ChartLoader } from "@/components/shared/chart_loader";
import { Tag } from "@/components/tag_management/types";
import UserAgentActivity from "@/components/user_agent_activity";
import ViewUserSpend from "@/components/view_user_spend";
import { usePaginatedDailyActivity } from "../hooks/usePaginatedDailyActivity";
import { DailyData, KeyMetricWithMetadata, MetricWithMetadata } from "@/components/UsagePage/types";
import { valueFormatterSpend } from "@/components/UsagePage/utils/value_formatters";
import {
  fetchedRangeKey,
  selectForRange,
  selectGatewayActivity,
  topGatewayRoutes,
  type FetchedForRange,
  type FetchedGatewayActivity,
  type GatewayActivity,
} from "./gatewayActivity";
import EndpointUsage from "./EndpointUsage/EndpointUsage";
import EntityUsage, { EntityList } from "./EntityUsage/EntityUsage";
import ModelViewToggle, { ModelViewType } from "./ModelViewToggle";
import SpendByProvider from "./EntityUsage/SpendByProvider";
import { TOP_MODEL_LIMITS } from "./EntityUsage/TopModelView";
import TopKeyView from "@/components/UsagePage/components/EntityUsage/TopKeyView";
import UsageAIChatPanel from "./UsageAIChatPanel";
import { UsageOption, UsageViewSelect } from "./UsageViewSelect/UsageViewSelect";

interface UsagePageProps {
  teams: Team[];
  organizations: Organization[];
}

const UsagePage: React.FC<UsagePageProps> = ({ teams, organizations }) => {
  const { accessToken, userRole, userId: userID, premiumUser } = useAuthorized();
  // Aggregated endpoint: try first, fall back to paginated if unavailable
  const [aggregatedData, setAggregatedData] = useState<FetchedForRange<{
    results: DailyData[];
    metadata: any;
  }> | null>(null);
  // Stamped like the data itself: the flag decides whether the paginated
  // fallback is read, and a flag left over from the previous range would let
  // that fallback's own leftover rows through.
  const [aggregatedFailure, setAggregatedFailure] = useState<FetchedForRange<true> | null>(null);
  const [aggregatedLoading, setAggregatedLoading] = useState(false);
  const [gatewayActivityData, setGatewayActivityData] = useState<FetchedGatewayActivity | null>(null);

  // Separate loading states for better UX
  const [isDateChanging, setIsDateChanging] = useState(false);

  // Create initial dates outside of state to prevent recreation
  const initialFromDate = useMemo(() => new Date(Date.now() - 7 * 24 * 60 * 60 * 1000), []);
  const initialToDate = useMemo(() => new Date(), []);

  // Single date state that directly triggers data fetching
  const [dateValue, setDateValue] = useState<DateRangePickerValue>({
    from: initialFromDate,
    to: initialToDate,
  });

  const [allTags, setAllTags] = useState<EntityList[]>([]);
  const { data: customers = [] } = useCustomers();
  const { data: agentsResponse } = useAgents();
  const { data: currentUser } = useCurrentUser();
  const isAdmin = all_admin_roles.includes(userRole || "");
  const canViewTagUsage = isAdmin || internalUserRoles.includes(userRole || "");
  const isOrgAdmin = useIsOrgAdmin();
  const canViewOrganizationUsage = hasCapability(userRole, "viewOrganizationUsage", isOrgAdmin);
  const canViewAgentUsage = hasCapability(userRole, "viewAgentUsage");

  // For admins: null means global view (all users), a string means filter by that user
  // For non-admins: always set to their own user ID
  const [selectedUserId, setSelectedUserId] = useState<string | null>(isAdmin ? null : userID || null);
  const [modelViewType, setModelViewType] = useState<ModelViewType>("groups");
  const [isCloudZeroModalOpen, setIsCloudZeroModalOpen] = useState(false);
  const [isGlobalExportModalOpen, setIsGlobalExportModalOpen] = useState(false);
  const [isAiChatOpen, setIsAiChatOpen] = useState(false);
  const [selectedUsageView, setUsageView] = useState<UsageOption>("global");
  // Org-admin membership is read from the server, so unlike the other usage
  // views this one can be revoked while the page is open. Derive the view in
  // render rather than storing it, so the fallback lands on the same paint and
  // the selector never holds a value it no longer offers.
  const usageView: UsageOption =
    selectedUsageView === "organization" && !canViewOrganizationUsage ? "global" : selectedUsageView;

  const [showCredentialBanner, setShowCredentialBanner] = useState(true);
  const [topKeysLimit, setTopKeysLimit] = useState<number>(5);
  const [topModelsLimit, setTopModelsLimit] = useState<number>(5);
  const [showTokenBreakdown, setShowTokenBreakdown] = useState(false);
  // Sync selectedUserId when auth state settles (isAdmin/userID may be null on initial render)
  useEffect(() => {
    if (!isAdmin && userID) {
      setSelectedUserId(userID);
    }
  }, [isAdmin, userID]);

  // For non-admins or "my-usage" view, always pass their own user_id
  const effectiveUserId = usageView === "my-usage" || !isAdmin ? userID || null : selectedUserId;

  const startTime = useMemo(() => (dateValue.from ? new Date(dateValue.from) : null), [dateValue.from]);
  const endTime = useMemo(() => (dateValue.to ? new Date(dateValue.to) : null), [dateValue.to]);

  useEffect(() => {
    if (!accessToken) return;
    let cancelled = false;
    (async () => {
      try {
        const tags = await tagListCall(accessToken, startTime, endTime);
        if (cancelled) return;
        setAllTags(
          Object.values(tags).map((tag: Tag) => ({
            label: tag.name,
            value: tag.name,
          })),
        );
      } catch (e) {
        if (!cancelled) {
          console.error("Failed to fetch tag list", e);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [accessToken, startTime, endTime]);

  // Everything the request tiles read is stamped with the range it answers and
  // selected during render, rather than cleared in an effect. An effect runs
  // after the render that follows a date change, so state cleared there is one
  // render too late: that render still holds the previous range's numbers and
  // can paint them. One source is not enough, since the tiles read the gateway
  // counts, fall through to the aggregate, and fall through again to the
  // paginated pages, so a stamp on any one of them is escaped by the next.
  const currentAggregatedRangeKey = fetchedRangeKey(startTime, endTime, effectiveUserId);
  const currentGatewayRangeKey = fetchedRangeKey(startTime, endTime);

  // Try aggregated endpoint first, fall back to paginated on failure
  const aggregatedFetchIdRef = useRef(0);
  useEffect(() => {
    if (!accessToken || !startTime || !endTime) return;
    const fetchId = ++aggregatedFetchIdRef.current;
    const rangeKey = currentAggregatedRangeKey;
    setAggregatedLoading(true);

    userDailyActivityAggregatedCall(accessToken, startTime, endTime, effectiveUserId)
      .then((data) => {
        if (aggregatedFetchIdRef.current !== fetchId) return;
        setAggregatedData({ rangeKey, value: data });
        setAggregatedLoading(false);
        setIsDateChanging(false);
      })
      .catch(() => {
        if (aggregatedFetchIdRef.current !== fetchId) return;
        setAggregatedFailure({ rangeKey, value: true });
        setAggregatedLoading(false);
      });
  }, [accessToken, startTime, endTime, effectiveUserId, currentAggregatedRangeKey]);

  // Gateway request counts (SGR). Admin-only: the source table is
  // deployment-wide, so a non-admin must not see it.
  const gatewayRequest = useMemo(
    () => (accessToken && startTime && endTime ? { accessToken, startTime, endTime } : null),
    [accessToken, startTime, endTime],
  );
  const gatewayFetchIdRef = useRef(0);
  useEffect(() => {
    if (!isAdmin || !gatewayRequest) return;
    const fetchId = ++gatewayFetchIdRef.current;
    gatewayDailyActivityCall(gatewayRequest.accessToken, gatewayRequest.startTime, gatewayRequest.endTime)
      .then((data) => {
        if (gatewayFetchIdRef.current !== fetchId) return;
        setGatewayActivityData({ rangeKey: currentGatewayRangeKey, value: data as GatewayActivity });
      })
      .catch(() => {
        if (gatewayFetchIdRef.current !== fetchId) return;
        setGatewayActivityData(null);
      });
  }, [isAdmin, gatewayRequest, currentGatewayRangeKey]);

  const gatewayActivity = selectGatewayActivity(isAdmin, gatewayActivityData, currentGatewayRangeKey);
  const activeAggregated = selectForRange(aggregatedData, currentAggregatedRangeKey);
  // A failure belongs to the range it happened on. Reading it through the same
  // rule keeps the paginated hook disabled while a new range is in flight, and
  // disabled is what empties it, so its previous rows never reach a tile.
  const aggregatedFailed = selectForRange(aggregatedFailure, currentAggregatedRangeKey) === true;

  // Paginated fallback — only enabled when aggregated endpoint fails
  const paginatedResult = usePaginatedDailyActivity({
    fetchFn: userDailyActivityCall,
    args: [accessToken, startTime, endTime, effectiveUserId],
    enabled: aggregatedFailed && !!accessToken && !!startTime && !!endTime,
  });

  // Derive userSpendData from whichever source is active
  const userSpendData = useMemo(() => {
    if (activeAggregated) return activeAggregated;
    if (aggregatedFailed) return paginatedResult.data;
    return { results: [] as DailyData[], metadata: {} as any };
  }, [activeAggregated, aggregatedFailed, paginatedResult.data]);

  const loading = aggregatedLoading || paginatedResult.loading;

  // Clear isDateChanging when paginated data starts arriving
  useEffect(() => {
    if (aggregatedFailed && !paginatedResult.loading && paginatedResult.data.results.length > 0) {
      setIsDateChanging(false);
    }
  }, [aggregatedFailed, paginatedResult.loading, paginatedResult.data.results.length]);

  // Super responsive date change handler
  const handleDateChange = useCallback((newValue: DateRangePickerValue) => {
    // Instant visual feedback
    setIsDateChanging(true);

    // Update date immediately for UI responsiveness
    setDateValue(newValue);
  }, []);

  // Derived states from userSpendData
  const totalSpend = userSpendData.metadata?.total_spend || 0;

  // Calculate top models from the breakdown data
  const topModels = useMemo(() => {
    const modelSpend: { [key: string]: MetricWithMetadata } = {};
    userSpendData.results.forEach((day) => {
      Object.entries(day.breakdown.models || {}).forEach(([model, metrics]) => {
        if (!modelSpend[model]) {
          modelSpend[model] = {
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
            metadata: {},
            api_key_breakdown: {},
          };
        }
        modelSpend[model].metrics.spend += metrics.metrics.spend;
        modelSpend[model].metrics.prompt_tokens += metrics.metrics.prompt_tokens;
        modelSpend[model].metrics.completion_tokens += metrics.metrics.completion_tokens;
        modelSpend[model].metrics.total_tokens += metrics.metrics.total_tokens;
        modelSpend[model].metrics.api_requests += metrics.metrics.api_requests;
        modelSpend[model].metrics.successful_requests += metrics.metrics.successful_requests || 0;
        modelSpend[model].metrics.failed_requests += metrics.metrics.failed_requests || 0;
        modelSpend[model].metrics.cache_read_input_tokens += metrics.metrics.cache_read_input_tokens || 0;
        modelSpend[model].metrics.cache_creation_input_tokens += metrics.metrics.cache_creation_input_tokens || 0;
      });
    });

    return Object.entries(modelSpend)
      .map(([model, metrics]) => ({
        key: model,
        spend: metrics.metrics.spend,
        requests: metrics.metrics.api_requests,
        successful_requests: metrics.metrics.successful_requests,
        failed_requests: metrics.metrics.failed_requests,
        tokens: metrics.metrics.total_tokens,
      }))
      .sort((a, b) => b.spend - a.spend)
      .slice(0, topModelsLimit);
  }, [userSpendData.results, topModelsLimit]);

  const topModelGroups = useMemo(() => {
    const modelGroupSpend: { [key: string]: MetricWithMetadata } = {};
    userSpendData.results.forEach((day) => {
      Object.entries(day.breakdown.model_groups || {}).forEach(([modelGroup, metrics]) => {
        if (!modelGroupSpend[modelGroup]) {
          modelGroupSpend[modelGroup] = {
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
            metadata: {},
            api_key_breakdown: {},
          };
        }
        modelGroupSpend[modelGroup].metrics.spend += metrics.metrics.spend;
        modelGroupSpend[modelGroup].metrics.prompt_tokens += metrics.metrics.prompt_tokens;
        modelGroupSpend[modelGroup].metrics.completion_tokens += metrics.metrics.completion_tokens;
        modelGroupSpend[modelGroup].metrics.total_tokens += metrics.metrics.total_tokens;
        modelGroupSpend[modelGroup].metrics.api_requests += metrics.metrics.api_requests;
        modelGroupSpend[modelGroup].metrics.successful_requests += metrics.metrics.successful_requests || 0;
        modelGroupSpend[modelGroup].metrics.failed_requests += metrics.metrics.failed_requests || 0;
        modelGroupSpend[modelGroup].metrics.cache_read_input_tokens += metrics.metrics.cache_read_input_tokens || 0;
        modelGroupSpend[modelGroup].metrics.cache_creation_input_tokens +=
          metrics.metrics.cache_creation_input_tokens || 0;
      });
    });

    return Object.entries(modelGroupSpend)
      .map(([modelGroup, metrics]) => ({
        key: modelGroup,
        spend: metrics.metrics.spend,
        requests: metrics.metrics.api_requests,
        successful_requests: metrics.metrics.successful_requests,
        failed_requests: metrics.metrics.failed_requests,
        tokens: metrics.metrics.total_tokens,
      }))
      .sort((a, b) => b.spend - a.spend)
      .slice(0, topModelsLimit);
  }, [userSpendData.results, topModelsLimit]);

  // Calculate provider spend from the breakdown data
  const providerSpend = useMemo(() => {
    const providerSpendMap: { [key: string]: MetricWithMetadata } = {};
    userSpendData.results.forEach((day) => {
      Object.entries(day.breakdown.providers || {}).forEach(([provider, metrics]) => {
        if (!providerSpendMap[provider]) {
          providerSpendMap[provider] = {
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
            metadata: {},
            api_key_breakdown: {},
          };
        }
        providerSpendMap[provider].metrics.spend += metrics.metrics.spend;
        providerSpendMap[provider].metrics.prompt_tokens += metrics.metrics.prompt_tokens;
        providerSpendMap[provider].metrics.completion_tokens += metrics.metrics.completion_tokens;
        providerSpendMap[provider].metrics.total_tokens += metrics.metrics.total_tokens;
        providerSpendMap[provider].metrics.api_requests += metrics.metrics.api_requests;
        providerSpendMap[provider].metrics.successful_requests += metrics.metrics.successful_requests || 0;
        providerSpendMap[provider].metrics.failed_requests += metrics.metrics.failed_requests || 0;
        providerSpendMap[provider].metrics.cache_read_input_tokens += metrics.metrics.cache_read_input_tokens || 0;
        providerSpendMap[provider].metrics.cache_creation_input_tokens +=
          metrics.metrics.cache_creation_input_tokens || 0;
      });
    });

    return Object.entries(providerSpendMap).map(([provider, metrics]) => ({
      provider,
      spend: metrics.metrics.spend,
      requests: metrics.metrics.api_requests,
      successful_requests: metrics.metrics.successful_requests,
      failed_requests: metrics.metrics.failed_requests,
      tokens: metrics.metrics.total_tokens,
    }));
  }, [userSpendData.results]);

  // Calculate top API keys from the breakdown data
  const topKeys = useMemo(() => {
    const keySpend: { [key: string]: KeyMetricWithMetadata } = {};
    userSpendData.results.forEach((day) => {
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
              team_id: null,
              tags: metrics.metadata.tags || [],
            },
          };
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
        key_alias: metrics.metadata.key_alias || "-",
        tags: metrics.metadata.tags || [],
        spend: metrics.metrics.spend,
      }))
      .sort((a, b) => b.spend - a.spend)
      .slice(0, topKeysLimit);
  }, [userSpendData.results, topKeysLimit]);

  const sortedDailyResults = useMemo(
    () => [...userSpendData.results].sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime()),
    [userSpendData.results],
  );
  const gatewayRequestsByRoute = useMemo(() => topGatewayRoutes(gatewayActivity), [gatewayActivity]);
  const modelMetrics = useMemo(
    () => processActivityData(userSpendData, modelViewType === "groups" ? "model_groups" : "models", teams),
    [userSpendData, modelViewType, teams],
  );
  const keyMetrics = useMemo(() => processActivityData(userSpendData, "api_keys", teams), [userSpendData, teams]);
  const mcpServerMetrics = useMemo(
    () => processActivityData(userSpendData, "mcp_servers", teams),
    [userSpendData, teams],
  );

  return (
    <div style={{ width: "100%" }} className="p-8 relative">
      {/* Global Date Picker and Tabs - Single Row */}
      <div className="flex items-end justify-between gap-6 mb-6">
        <div className="flex-1">
          <div className="flex items-end justify-between gap-6 mb-4 w-full">
            <UsageViewSelect
              value={usageView}
              onChange={(value) => setUsageView(value)}
              userRole={userRole}
              canViewTagUsage={canViewTagUsage}
              isOrgAdmin={isOrgAdmin}
            />
            <AdvancedDatePicker value={dateValue} onValueChange={handleDateChange} />
          </div>
          <PaginationStatusAlerts
            isFetchingMore={paginatedResult.isFetchingMore}
            cancelled={paginatedResult.cancelled}
            progress={paginatedResult.progress}
            cancel={paginatedResult.cancel}
          />
          {/* Your Usage / Global Usage Panel */}
          {(usageView === "global" || usageView === "my-usage") && (
            <>
              {isAdmin && usageView === "global" && (
                <div className="mb-4">
                  <p className="mb-2 text-sm text-foreground">Filter by user</p>
                  <UserDropdown value={selectedUserId} onChange={setSelectedUserId} />
                </div>
              )}
              <Tabs defaultValue="cost">
                <div className="flex justify-between items-center">
                  <TabsList className="mt-1">
                    <TabsTrigger value="cost" className="flex-none px-3">
                      Cost
                    </TabsTrigger>
                    <TabsTrigger value="models" className="flex-none px-3">
                      Model Activity
                    </TabsTrigger>
                    <TabsTrigger value="keys" className="flex-none px-3">
                      Key Activity
                    </TabsTrigger>
                    <TabsTrigger value="mcp" className="flex-none px-3">
                      MCP Server Activity
                    </TabsTrigger>
                    <TabsTrigger value="endpoints" className="flex-none px-3">
                      Endpoint Activity
                    </TabsTrigger>
                  </TabsList>
                  <div className="flex items-center gap-2">
                    <Button variant="outline" onClick={() => setIsAiChatOpen(true)}>
                      <Sparkles />
                      Ask AI
                    </Button>
                    <Button variant="outline" onClick={() => setIsGlobalExportModalOpen(true)}>
                      <Download />
                      Export Data
                    </Button>
                  </div>
                </div>
                {/* Cost Panel */}
                <TabsContent value="cost" keepMounted>
                  <div className="grid grid-cols-2 gap-2 w-full">
                    {/* Total Spend Card */}
                    <div className="col-span-2">
                      <div className="flex items-center gap-4 mt-2 mb-2">
                        <p className="text-lg text-muted-foreground">
                          Project Spend{" "}
                          {dateValue.from && dateValue.to && (
                            <>
                              {dateValue.from.toLocaleDateString("en-US", {
                                month: "short",
                                day: "numeric",
                                year:
                                  dateValue.from.getFullYear() !== dateValue.to.getFullYear() ? "numeric" : undefined,
                              })}
                              {" - "}
                              {dateValue.to.toLocaleDateString("en-US", {
                                month: "short",
                                day: "numeric",
                                year: "numeric",
                              })}
                            </>
                          )}
                        </p>
                      </div>

                      <ViewUserSpend
                        userSpend={totalSpend}
                        selectedTeam={null}
                        userMaxBudget={currentUser?.max_budget || null}
                      />
                    </div>

                    <div className="col-span-2">
                      <ShadcnCard>
                        <CardContent>
                          <h3 className="text-lg font-medium text-foreground">Usage Metrics</h3>
                          <div className="grid grid-cols-5 gap-4 mt-4">
                            <ShadcnCard>
                              <CardContent>
                                <h3 className="text-lg font-medium text-foreground">Total Requests</h3>
                                <p className="text-2xl font-bold mt-2">
                                  {userSpendData.metadata?.total_api_requests?.toLocaleString() || 0}
                                </p>
                              </CardContent>
                            </ShadcnCard>
                            <ShadcnCard>
                              <CardContent>
                                <div className="flex items-center gap-2">
                                  <h3 className="text-lg font-medium text-foreground">Successful Requests</h3>
                                  {gatewayActivity && (
                                    <Tooltip>
                                      <TooltipTrigger
                                        render={<Info className="size-4 text-muted-foreground hover:text-foreground" />}
                                      />
                                      <TooltipContent>
                                        Counted by the gateway when it answers a request, independent of spend logging.
                                        Deployment-wide, so it will not match the per-key or per-model breakdowns below.
                                      </TooltipContent>
                                    </Tooltip>
                                  )}
                                </div>
                                {/*
                                  TODO: drop the userSpendData fallback once every deployment
                                  is writing LiteLLM_DailyGatewayRequests. It covers two cases
                                  today: a non-admin (who may not read deployment-wide counts)
                                  and an admin on a proxy whose table is still backfilling.
                                */}
                                <p className="text-2xl font-bold mt-2 text-success">
                                  {(
                                    gatewayActivity?.total_successful_requests ??
                                    userSpendData.metadata?.total_successful_requests
                                  )?.toLocaleString() || 0}
                                </p>
                              </CardContent>
                            </ShadcnCard>
                            <ShadcnCard>
                              <CardContent>
                                <div className="flex items-center gap-2">
                                  <h3 className="text-lg font-medium text-foreground">Failed Requests</h3>
                                  <Tooltip>
                                    <TooltipTrigger
                                      render={<Info className="size-4 text-muted-foreground hover:text-foreground" />}
                                    />
                                    <TooltipContent>
                                      {gatewayActivity
                                        ? "Counted by the gateway when it answers a request, independent of spend logging. Deployment-wide, so it will not match the per-key or per-model breakdowns below."
                                        : "Includes requests that failed to route to a provider, tool usage failures, and other request errors where the provider cannot be determined."}
                                    </TooltipContent>
                                  </Tooltip>
                                </div>
                                {/* Same source as Successful Requests: the two must agree, or the
                                    tile disagrees with the endpoint breakdown chart below it. */}
                                <p className="text-2xl font-bold mt-2 text-destructive">
                                  {(
                                    gatewayActivity?.total_failed_requests ??
                                    userSpendData.metadata?.total_failed_requests
                                  )?.toLocaleString() || 0}
                                </p>
                              </CardContent>
                            </ShadcnCard>
                            <ShadcnCard>
                              <CardContent>
                                <h3 className="text-lg font-medium text-foreground">Average Cost per Request</h3>
                                <p className="text-2xl font-bold mt-2">
                                  $
                                  {formatNumberWithCommas(
                                    (totalSpend || 0) / (userSpendData.metadata?.total_api_requests || 1),
                                    4,
                                  )}
                                </p>
                              </CardContent>
                            </ShadcnCard>
                            <ShadcnCard
                              className="cursor-pointer hover:bg-accent transition-colors"
                              onClick={() => setShowTokenBreakdown(!showTokenBreakdown)}
                            >
                              <CardContent>
                                <div className="flex items-center gap-2">
                                  <h3 className="text-lg font-medium text-foreground">Total Tokens</h3>
                                  {showTokenBreakdown ? (
                                    <ChevronDown className="size-3 text-muted-foreground" />
                                  ) : (
                                    <ChevronRight className="size-3 text-muted-foreground" />
                                  )}
                                </div>
                                <p className="text-2xl font-bold mt-2">
                                  {userSpendData.metadata?.total_tokens?.toLocaleString() || 0}
                                </p>
                              </CardContent>
                            </ShadcnCard>
                          </div>
                          {showTokenBreakdown && (
                            <div className="grid grid-cols-4 gap-4 mt-4">
                              <ShadcnCard>
                                <CardContent>
                                  <h3 className="text-lg font-medium text-foreground">Input Tokens</h3>
                                  <p className="text-2xl font-bold mt-2 text-info">
                                    {(userSpendData.metadata?.total_prompt_tokens || 0).toLocaleString()}
                                  </p>
                                </CardContent>
                              </ShadcnCard>
                              <ShadcnCard>
                                <CardContent>
                                  <h3 className="text-lg font-medium text-foreground">Output Tokens</h3>
                                  <p className="text-2xl font-bold mt-2 text-info">
                                    {userSpendData.metadata?.total_completion_tokens?.toLocaleString() || 0}
                                  </p>
                                </CardContent>
                              </ShadcnCard>
                              <ShadcnCard>
                                <CardContent>
                                  <h3 className="text-lg font-medium text-foreground">Cache Read Tokens</h3>
                                  <p className="text-2xl font-bold mt-2 text-success">
                                    {userSpendData.metadata?.total_cache_read_input_tokens?.toLocaleString() || 0}
                                  </p>
                                </CardContent>
                              </ShadcnCard>
                              <ShadcnCard>
                                <CardContent>
                                  <h3 className="text-lg font-medium text-foreground">Cache Write Tokens</h3>
                                  <p className="text-2xl font-bold mt-2 text-purple-600">
                                    {userSpendData.metadata?.total_cache_creation_input_tokens?.toLocaleString() || 0}
                                  </p>
                                </CardContent>
                              </ShadcnCard>
                            </div>
                          )}
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
                          {loading ? (
                            <ChartLoader isDateChanging={isDateChanging} />
                          ) : (
                            <BarChart
                              data={sortedDailyResults}
                              index="date"
                              categories={["metrics.spend"]}
                              colors={["cyan"]}
                              valueFormatter={valueFormatterSpend}
                              yAxisWidth={100}
                              showLegend={false}
                              customTooltip={({ payload, active }) => {
                                if (!active || !payload?.[0]) return null;
                                const data = payload[0].payload;
                                return (
                                  <div className="bg-card p-4 shadow-lg rounded-lg border">
                                    <p className="font-bold">{data.date}</p>
                                    <p className="text-info">Spend: ${formatNumberWithCommas(data.metrics.spend, 2)}</p>
                                    <p className="text-muted-foreground">Requests: {data.metrics.api_requests}</p>
                                    <p className="text-muted-foreground">
                                      Successful: {data.metrics.successful_requests}
                                    </p>
                                    <p className="text-muted-foreground">Failed: {data.metrics.failed_requests}</p>
                                    <p className="text-muted-foreground">Tokens: {data.metrics.total_tokens}</p>
                                  </div>
                                );
                              }}
                            />
                          )}
                        </CardContent>
                      </ShadcnCard>
                    </div>
                    {/* Gateway Requests by Endpoint (SGR) */}
                    {gatewayActivity && gatewayActivity.by_route.length > 0 && (
                      <div className="col-span-2">
                        <ShadcnCard data-testid="gateway-requests-by-endpoint">
                          <CardHeader>
                            <CardTitle className="text-base font-semibold">
                              Gateway Requests by Endpoint
                              <Tooltip>
                                <TooltipTrigger
                                  render={
                                    <Info className="ml-2 inline size-4 text-muted-foreground hover:text-foreground" />
                                  }
                                />
                                <TooltipContent>
                                  Counted by the gateway middleware as each request is answered. Covers LLM, MCP and A2A
                                  endpoints across the whole deployment.
                                </TooltipContent>
                              </Tooltip>
                            </CardTitle>
                          </CardHeader>
                          <CardContent>
                            <BarChart
                              data={gatewayRequestsByRoute}
                              index="route"
                              categories={["successful_requests", "failed_requests"]}
                              colors={["green", "red"]}
                              stack={true}
                              yAxisWidth={100}
                              valueFormatter={(value: number) => value.toLocaleString()}
                            />
                          </CardContent>
                        </ShadcnCard>
                      </div>
                    )}
                    {/* Top API Keys */}
                    <div>
                      <ShadcnCard className="h-full">
                        <CardContent>
                          <h3 className="text-lg font-medium text-foreground">Top Virtual Keys</h3>
                          <TopKeyView
                            topKeys={topKeys}
                            teams={null}
                            topKeysLimit={topKeysLimit}
                            setTopKeysLimit={setTopKeysLimit}
                          />
                        </CardContent>
                      </ShadcnCard>
                    </div>

                    {/* Top Models */}
                    <div>
                      <ShadcnCard className="h-full">
                        <CardContent>
                          <h3 className="text-lg font-medium text-foreground">
                            {modelViewType === "groups" ? "Top Public Model Names" : "Top Litellm Models"}
                          </h3>
                          <div className="flex justify-between items-center mb-4">
                            <Tabs
                              value={String(topModelsLimit)}
                              onValueChange={(value: string) => setTopModelsLimit(Number(value))}
                            >
                              <TabsList>
                                {TOP_MODEL_LIMITS.map((limit) => (
                                  <TabsTrigger key={limit} value={String(limit)} className="flex-none px-3">
                                    {limit}
                                  </TabsTrigger>
                                ))}
                              </TabsList>
                            </Tabs>
                            <ModelViewToggle value={modelViewType} onChange={setModelViewType} />
                          </div>
                          {loading ? (
                            <ChartLoader isDateChanging={isDateChanging} />
                          ) : (
                            <div className="relative max-h-[600px] overflow-y-auto">
                              {(() => {
                                const modelData = modelViewType === "groups" ? topModelGroups : topModels;
                                return (
                                  <BarChart
                                    className="mt-4"
                                    style={{ height: Math.min(modelData.length, topModelsLimit) * 52 }}
                                    data={modelData}
                                    index="key"
                                    categories={["spend"]}
                                    colors={["cyan"]}
                                    valueFormatter={valueFormatterSpend}
                                    layout="vertical"
                                    yAxisWidth={200}
                                    showLegend={false}
                                    customTooltip={({ payload, active }) => {
                                      if (!active || !payload?.[0]) return null;
                                      const data = payload[0].payload;
                                      return (
                                        <div className="bg-card p-4 shadow-lg rounded-lg border">
                                          <p className="font-bold">{data.key}</p>
                                          <p className="text-info">Spend: ${formatNumberWithCommas(data.spend, 2)}</p>
                                          <p className="text-muted-foreground">
                                            Total Requests: {data.requests.toLocaleString()}
                                          </p>
                                          <p className="text-success">
                                            Successful: {data.successful_requests.toLocaleString()}
                                          </p>
                                          <p className="text-destructive">
                                            Failed: {data.failed_requests.toLocaleString()}
                                          </p>
                                          <p className="text-muted-foreground">
                                            Tokens: {data.tokens.toLocaleString()}
                                          </p>
                                        </div>
                                      );
                                    }}
                                  />
                                );
                              })()}
                            </div>
                          )}
                        </CardContent>
                      </ShadcnCard>
                    </div>

                    {/* Spend by Provider */}
                    <div className="col-span-2">
                      <SpendByProvider
                        loading={loading}
                        isDateChanging={isDateChanging}
                        providerSpend={providerSpend}
                      />
                    </div>

                    {/* Usage Metrics */}
                  </div>
                </TabsContent>

                {/* Activity Panel */}
                <TabsContent value="models" keepMounted>
                  <div className="flex justify-end mt-2 mb-4">
                    <ModelViewToggle value={modelViewType} onChange={setModelViewType} />
                  </div>
                  <ActivityMetrics modelMetrics={modelMetrics} />
                </TabsContent>
                <TabsContent value="keys" keepMounted>
                  <ActivityMetrics modelMetrics={keyMetrics} />
                </TabsContent>
                <TabsContent value="mcp" keepMounted>
                  <ActivityMetrics modelMetrics={mcpServerMetrics} />
                </TabsContent>
                <TabsContent value="endpoints" keepMounted>
                  <EndpointUsage userSpendData={userSpendData} />
                </TabsContent>
              </Tabs>
            </>
          )}
          {/* Organization Usage Panel */}

          {usageView === "organization" && canViewOrganizationUsage && (
            <EntityUsage
              accessToken={accessToken}
              entityType="organization"
              userID={userID}
              userRole={userRole}
              isOrgAdmin={isOrgAdmin}
              dateValue={dateValue}
              entityList={
                organizations?.map((organization) => ({
                  label: organization.organization_alias,
                  value: organization.organization_id,
                })) || null
              }
              premiumUser={premiumUser}
            />
          )}

          {/* Team Usage Panel */}
          {usageView === "team" && (
            <EntityUsage
              accessToken={accessToken}
              entityType="team"
              userID={userID}
              userRole={userRole}
              entityList={
                teams?.map((team) => ({
                  label: team.team_alias,
                  value: team.team_id,
                })) || null
              }
              premiumUser={premiumUser}
              dateValue={dateValue}
            />
          )}

          {/* Customer Usage Panel */}
          {usageView === "customer" && (
            <EntityUsage
              accessToken={accessToken}
              entityType="customer"
              userID={userID}
              userRole={userRole}
              entityList={
                customers?.map((customer) => ({
                  label: customer.alias || customer.user_id,
                  value: customer.user_id,
                })) || null
              }
              premiumUser={premiumUser}
              dateValue={dateValue}
            />
          )}
          {/* Tag Usage Panel */}
          {usageView === "tag" && (
            <>
              {showCredentialBanner && (
                <Alert variant="info" className="mb-5">
                  <AlertTitle>Reusable credentials are automatically tracked as tags</AlertTitle>
                  <AlertDescription className="text-inherit">
                    When a reusable credential is used, it will appear as a tag prefixed with{" "}
                    <code className="rounded bg-black/5 px-1 py-0.5 font-mono text-xs">Credential: </code>
                    in this view.
                  </AlertDescription>
                  <AlertAction>
                    <Button
                      variant="ghost"
                      size="icon-xs"
                      aria-label="Close"
                      onClick={() => setShowCredentialBanner(false)}
                    >
                      <X />
                    </Button>
                  </AlertAction>
                </Alert>
              )}
              <EntityUsage
                accessToken={accessToken}
                entityType="tag"
                userID={userID}
                userRole={userRole}
                entityList={allTags}
                premiumUser={premiumUser}
                dateValue={dateValue}
              />
            </>
          )}
          {usageView === "agent" && canViewAgentUsage && (
            <EntityUsage
              accessToken={accessToken}
              entityType="agent"
              userID={userID}
              userRole={userRole}
              entityList={
                agentsResponse?.agents?.map((agent) => ({ label: agent.agent_name, value: agent.agent_id })) || null
              }
              premiumUser={premiumUser}
              dateValue={dateValue}
            />
          )}
          {/* User Usage Panel */}
          {usageView === "user" && (
            <EntityUsage
              accessToken={accessToken}
              entityType="user"
              userID={userID}
              userRole={userRole}
              entityList={null}
              premiumUser={premiumUser}
              dateValue={dateValue}
            />
          )}
          {/* User Agent Activity Panel */}
          {usageView === "user-agent-activity" && (
            <UserAgentActivity accessToken={accessToken} userRole={userRole} dateValue={dateValue} />
          )}
        </div>
      </div>

      {/* CloudZero Export Modal */}
      <CloudZeroExportModal
        isOpen={isCloudZeroModalOpen}
        onClose={() => setIsCloudZeroModalOpen(false)}
        accessToken={accessToken}
      />

      {/* Global Usage Export Modal */}
      <EntityUsageExportModal
        isOpen={isGlobalExportModalOpen}
        onClose={() => setIsGlobalExportModalOpen(false)}
        entityType="team"
        spendData={{
          results: userSpendData.results,
          metadata: userSpendData.metadata,
        }}
        dateRange={dateValue}
        selectedFilters={[]}
        customTitle="Export Usage Data"
      />

      {/* AI Chat Panel */}
      <UsageAIChatPanel open={isAiChatOpen} onClose={() => setIsAiChatOpen(false)} accessToken={accessToken} />
    </div>
  );
};

// Add this helper function to process model-specific activity data

export default UsagePage;
