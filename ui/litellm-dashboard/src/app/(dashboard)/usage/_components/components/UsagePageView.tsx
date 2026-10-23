/**
 * New Usage Page
 *
 * Uses the new `/user/daily/activity` endpoint to get daily activity data for a user.
 *
 * Works at 1m+ spend logs, by querying an aggregate table instead.
 */

import { Download, Link2, Search, Sparkles, X } from "lucide-react";
import { useQueryStates } from "nuqs";
import type { DateRangePickerValue } from "@/components/shared/date_picker_types";
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { StackedUsageChart } from "@/components/shared/charts";
import { Alert, AlertAction, AlertDescription, AlertTitle } from "@/components/shared/Alert";
import { ErrorState } from "@/components/shared/ErrorState";
import { Button } from "@/components/ui/button";
import { InputGroup, InputGroupAddon, InputGroupButton, InputGroupInput } from "@/components/ui/input-group";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

import { useAgents } from "@/app/(dashboard)/hooks/agents/useAgents";
import { useCustomers } from "@/app/(dashboard)/hooks/customers/useCustomers";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import useIsOrgAdmin from "@/app/(dashboard)/hooks/useIsOrgAdmin";
import { useCurrentUser } from "@/app/(dashboard)/hooks/users/useCurrentUser";
import { useOrganizations } from "@/app/(dashboard)/hooks/organizations/useOrganizations";
import { toast } from "@/lib/toast";
import { hasCapability } from "@/utils/capabilities";
import { copyToClipboard } from "@/utils/dataUtils";
import { all_admin_roles, internalUserRoles } from "@/utils/roles";
import { ActivityMetrics, processActivityData } from "@/components/activity_metrics";
import CloudZeroExportModal from "@/components/cloudzero_export_modal";
import UserDropdown from "@/components/common_components/UserDropdown";
import EntityUsageExportModal from "@/components/EntityUsageExport";
import type { EntityType } from "@/components/EntityUsageExport/types";
import KeyActivityPanel from "@/components/UsagePage/components/KeyActivityPanel";
import TopUsersView from "@/components/UsagePage/components/TopUsersView";
import UserActivityPanel from "@/components/UsagePage/components/UserActivityPanel";
import { filterModelActivity } from "@/components/UsagePage/modelActivityFilter";
import { Team } from "@/components/key_team_helpers/key_list";
import {
  gatewayDailyActivityCall,
  Organization,
  requestErrorActivityCall,
  tagListCall,
  userDailyActivityUserPageCall,
} from "@/components/networking";
import AdvancedDatePicker from "@/components/shared/advanced_date_picker";
import { Tag } from "@/components/tag_management/types";
import UserAgentActivity from "@/components/user_agent_activity";
import { useAggregatedDailyActivity } from "../hooks/useAggregatedDailyActivity";
import { ENTITY_API } from "./EntityUsage/entityFetchFns";
import {
  EMPTY_DAILY_ACTIVITY_METADATA,
  toDailyData,
  type DailyActivityRequest,
} from "@/components/UsagePage/dailyActivityApi";
import { keyDetailFromResponse, overallUsageMetrics } from "@/components/UsagePage/keyActivityData";
import {
  fetchedRangeKey,
  selectForRange,
  selectGatewayActivity,
  topGatewayRoutes,
  type FetchedForRange,
  type FetchedGatewayActivity,
  type GatewayActivity,
} from "./gatewayActivity";
import ErrorsTab from "./errors/ErrorsTab";
import type { RequestErrorActivity } from "./errors/errorsData";
import EndpointUsage from "./EndpointUsage/EndpointUsage";
import EntityUsage, { EntityList } from "./EntityUsage/EntityUsage";
import ModelViewToggle, { ModelViewType } from "./ModelViewToggle";
import TopKeyView, { type TopKeyItem } from "@/components/UsagePage/components/EntityUsage/TopKeyView";
import { getGlobalTopKeys } from "./EntityUsage/entityUsageAggregations";
import UsageAIChatPanel from "./UsageAIChatPanel";
import { allowedUsageOptions, UsageViewSelect } from "./UsageViewSelect/UsageViewSelect";
import UserRecordLink from "./UserRecordLink";
import {
  cleanUsageUrl,
  dateRangeFromParams,
  dateRangePatch,
  entitySelectionPatch,
  selectedEntitiesFromParams,
  DEFAULT_USAGE_VIEW,
  usageTabFromParams,
  usageTabPatch,
  usageUrlParsers,
  usageViewFromParams,
  usageViewPatch,
} from "../usageUrlState";
import { overviewTotals, rollUpBreakdown } from "./overview/overviewData";
import TopAgents from "./overview/TopAgents";
import type { AgentRow } from "./overview/agentCatalog";
import SpendByProvider from "./EntityUsage/SpendByProvider";
import { useTagSummary } from "@/app/(dashboard)/hooks/tags/useTagSummary";
import { Panel } from "./overview/Primitives";
import UsageOverview from "./overview/UsageOverview";

interface UsagePageProps {
  teams: Team[];
  organizations: Organization[];
}

const UsagePage: React.FC<UsagePageProps> = ({ teams, organizations }) => {
  const { accessToken, userRole, userId: userID, premiumUser } = useAuthorized();
  const [gatewayActivityData, setGatewayActivityData] = useState<FetchedGatewayActivity | null>(null);

  // Create initial dates outside of state to prevent recreation
  const initialFromDate = useMemo(() => new Date(Date.now() - 7 * 24 * 60 * 60 * 1000), []);
  const initialToDate = useMemo(() => new Date(), []);

  const [urlParams, setUrlParams] = useQueryStates(usageUrlParsers);
  const { range, from, to } = urlParams;
  const dateValue = useMemo<DateRangePickerValue>(
    () => dateRangeFromParams({ range, from, to }) ?? { from: initialFromDate, to: initialToDate },
    [range, from, to, initialFromDate, initialToDate],
  );

  const [fetchedTags, setFetchedTags] = useState<FetchedForRange<EntityList[]> | null>(null);
  // No [] default: an unresolved query must stay undefined so the customer
  // filter reads as loading rather than as a range with no customers.
  const { data: customers } = useCustomers();
  const { data: agentsResponse } = useAgents();
  const { data: currentUser } = useCurrentUser();
  const isAdmin = all_admin_roles.includes(userRole || "");
  const canViewTagUsage = isAdmin || internalUserRoles.includes(userRole || "");
  const isOrgAdmin = useIsOrgAdmin();
  const canViewAgentUsage = hasCapability(userRole, "viewAgentUsage");
  const allowedViews = useMemo(
    () => allowedUsageOptions(userRole, canViewTagUsage, isOrgAdmin),
    [userRole, canViewTagUsage, isOrgAdmin],
  );

  const [modelViewType, setModelViewType] = useState<ModelViewType>("groups");
  const [modelQuery, setModelQuery] = useState("");
  const [isCloudZeroModalOpen, setIsCloudZeroModalOpen] = useState(false);
  const [isGlobalExportModalOpen, setIsGlobalExportModalOpen] = useState(false);
  const [isAiChatOpen, setIsAiChatOpen] = useState(false);
  const selectedUsageView = usageViewFromParams(urlParams);
  const selectedTab = usageTabFromParams(urlParams, isAdmin);
  const usageView = allowedViews.includes(selectedUsageView) ? selectedUsageView : DEFAULT_USAGE_VIEW;

  const [showCredentialBanner, setShowCredentialBanner] = useState(true);
  const [topKeysLimit, setTopKeysLimit] = useState<number>(5);
  // For non-admins or "my-usage" view, always pass their own user_id
  const effectiveUserId = usageView === "my-usage" || !isAdmin ? userID || null : urlParams.user;

  const { isLoading: organizationsLoading } = useOrganizations();
  const accessSettled = userRole !== null && !organizationsLoading;
  useEffect(() => {
    if (!accessSettled) return;
    const cleanup = cleanUsageUrl(urlParams, { allowedViews, isAdmin });
    if (!cleanup) return;
    if (cleanup.deniedAccess)
      toast.warning("You don't have access to that usage view", { id: "usage-url-access-denied" });
    void setUrlParams(cleanup.patch);
  }, [accessSettled, urlParams, allowedViews, isAdmin, setUrlParams]);

  const entitySelection = (entityType: EntityType) => ({
    selectedEntities: selectedEntitiesFromParams(urlParams, entityType),
    onSelectedEntitiesChange: (ids: readonly string[]) => void setUrlParams(entitySelectionPatch(entityType, ids)),
  });

  const startTime = useMemo(() => (dateValue.from ? new Date(dateValue.from) : null), [dateValue.from]);
  const endTime = useMemo(() => (dateValue.to ? new Date(dateValue.to) : null), [dateValue.to]);
  // Stamped and selected during render like the request tiles below: the tag
  // filter reads "no tags" from an empty list, so a list left over from the
  // previous range would state that about a range nobody has measured yet.
  const currentTagRangeKey = fetchedRangeKey(startTime, endTime);
  const allTags = selectForRange(fetchedTags, currentTagRangeKey);

  useEffect(() => {
    if (!accessToken) return;
    let cancelled = false;
    (async () => {
      try {
        const tags = await tagListCall(accessToken, startTime, endTime);
        if (cancelled) return;
        setFetchedTags({
          rangeKey: currentTagRangeKey,
          value: Object.values(tags).map((tag: Tag) => ({
            label: tag.name,
            value: tag.name,
          })),
        });
      } catch (e) {
        if (!cancelled) {
          console.error("Failed to fetch tag list", e);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [accessToken, startTime, endTime, currentTagRangeKey]);

  // Everything the request tiles read is stamped with the range it answers and
  // selected during render, rather than cleared in an effect. An effect runs
  // after the render that follows a date change, so state cleared there is one
  // render too late: that render still holds the previous range's numbers and
  // can paint them. One source is not enough, since the tiles read the gateway
  // counts, fall through to the aggregate, and fall through again to the
  // paginated pages, so a stamp on any one of them is escaped by the next.
  const currentGatewayRangeKey = fetchedRangeKey(startTime, endTime);

  const dailyActivityRequest = useMemo<DailyActivityRequest | null>(
    () =>
      accessToken && startTime && endTime
        ? {
            accessToken,
            startTime,
            endTime,
            entityIds: effectiveUserId ? [effectiveUserId] : null,
          }
        : null,
    [accessToken, startTime, endTime, effectiveUserId],
  );
  const {
    data: aggregatedRaw,
    loading: aggregatedLoading,
    failed: aggregatedFailed,
    errorMessage: aggregatedError,
    reload: reloadAggregated,
  } = useAggregatedDailyActivity({
    fetch: () => ENTITY_API.user.aggregated(dailyActivityRequest as DailyActivityRequest),
    enabled: dailyActivityRequest !== null,
    deps: [accessToken, startTime, endTime, effectiveUserId],
  });
  // Tag data is deployment-wide with no per-user dimension, so Top agents only appears where the
  // rest of the page is deployment-wide too: an admin's global view with no user selected.
  const showTopAgents = isAdmin && usageView === "global" && effectiveUserId === null;
  const tagDailyRequest = useMemo<DailyActivityRequest | null>(
    () => (accessToken && startTime && endTime ? { accessToken, startTime, endTime, entityIds: null } : null),
    [accessToken, startTime, endTime],
  );
  const { data: tagDailyRaw, loading: tagDailyLoading } = useAggregatedDailyActivity({
    fetch: () => ENTITY_API.tag.aggregated(tagDailyRequest as DailyActivityRequest),
    enabled: tagDailyRequest !== null && showTopAgents,
    deps: [accessToken, startTime, endTime, showTopAgents],
  });
  const tagDaily = useMemo(() => toDailyData(tagDailyRaw), [tagDailyRaw]);
  const openAgentActivity = useCallback(
    (agent: AgentRow) => {
      void setUrlParams({
        ...usageViewPatch("user-agent-activity"),
        ...entitySelectionPatch("tag", agent.tags),
      });
    },
    [setUrlParams],
  );

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

  const [requestErrorData, setRequestErrorData] = useState<FetchedForRange<RequestErrorActivity | null> | null>(null);
  const requestErrorFetchIdRef = useRef(0);
  useEffect(() => {
    if (!isAdmin || !gatewayRequest) return;
    const fetchId = ++requestErrorFetchIdRef.current;
    requestErrorActivityCall(gatewayRequest.accessToken, gatewayRequest.startTime, gatewayRequest.endTime)
      .then((data) => {
        if (requestErrorFetchIdRef.current !== fetchId) return;
        setRequestErrorData({ rangeKey: currentGatewayRangeKey, value: data as RequestErrorActivity });
      })
      .catch(() => {
        if (requestErrorFetchIdRef.current !== fetchId) return;
        setRequestErrorData({ rangeKey: currentGatewayRangeKey, value: null });
      });
  }, [isAdmin, gatewayRequest, currentGatewayRangeKey]);
  const requestErrorsForRange = isAdmin ? requestErrorData : null;
  const requestErrorsFailed =
    requestErrorsForRange?.rangeKey === currentGatewayRangeKey && requestErrorsForRange.value === null;
  const requestErrorActivity =
    requestErrorsForRange?.rangeKey === currentGatewayRangeKey ? requestErrorsForRange.value : null;

  const userSpendData = useMemo(
    () => ({
      results: toDailyData(aggregatedRaw),
      metadata: aggregatedRaw.metadata ?? EMPTY_DAILY_ACTIVITY_METADATA,
    }),
    [aggregatedRaw],
  );

  const loading = aggregatedLoading;
  const requestCountsPending = loading && gatewayActivity === null;

  const summaryMetrics = useMemo(
    () => overallUsageMetrics(userSpendData.results, userSpendData.metadata),
    [userSpendData],
  );

  const handleDateChange = useCallback(
    (newValue: DateRangePickerValue, presetShortLabel: string | null) =>
      void setUrlParams(dateRangePatch(newValue, presetShortLabel)),
    [setUrlParams],
  );

  const totals = useMemo(
    () => overviewTotals(userSpendData.metadata, gatewayActivity),
    [userSpendData.metadata, gatewayActivity],
  );
  const providerSpend = useMemo(
    () => rollUpBreakdown(userSpendData.results, "providers").map(({ key, ...row }) => ({ provider: key, ...row })),
    [userSpendData.results],
  );
  const { data: tagSummary, isLoading: tagSummaryLoading } = useTagSummary(startTime, endTime, showTopAgents);

  // Calculate top API keys from the breakdown data
  const topKeys = useMemo<TopKeyItem[]>(
    () => getGlobalTopKeys(userSpendData.results, topKeysLimit),
    [userSpendData.results, topKeysLimit],
  );

  const sortedDailyResults = useMemo(
    () => [...userSpendData.results].sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime()),
    [userSpendData.results],
  );
  const gatewayRequestsByRoute = useMemo(() => topGatewayRoutes(gatewayActivity), [gatewayActivity]);
  const modelMetrics = useMemo(
    () => processActivityData(userSpendData, modelViewType === "groups" ? "model_groups" : "models", teams),
    [userSpendData, modelViewType, teams],
  );
  const filteredModelMetrics = useMemo(() => filterModelActivity(modelMetrics, modelQuery), [modelMetrics, modelQuery]);
  const trimmedModelQuery = modelQuery.trim();
  const mcpServerMetrics = useMemo(
    () => processActivityData(userSpendData, "mcp_servers", teams),
    [userSpendData, teams],
  );

  const fetchTopApiKeys = useCallback(
    (model: string) =>
      ENTITY_API.user.modelTopKeys(dailyActivityRequest as DailyActivityRequest, model, modelViewType === "groups"),
    [dailyActivityRequest, modelViewType],
  );
  const searchKeys = useCallback(
    (query: string) =>
      dailyActivityRequest === null
        ? Promise.resolve({ api_keys: [] })
        : ENTITY_API.user.searchKeys(dailyActivityRequest, query),
    [dailyActivityRequest],
  );
  const fetchKeyPage = useCallback(
    (offset: number, limit: number) => {
      if (dailyActivityRequest === null) {
        const emptyPage = { api_keys: [], total_api_keys: 0, offset, limit };
        return Promise.resolve(emptyPage);
      }
      return ENTITY_API.user.keyPage(dailyActivityRequest, offset, limit);
    },
    [dailyActivityRequest],
  );
  const fetchUserPage = useCallback(
    (offset: number, limit: number) => {
      if (dailyActivityRequest === null) {
        const emptyPage = { users: [], total_users: 0, offset, limit };
        return Promise.resolve(emptyPage);
      }
      return userDailyActivityUserPageCall(dailyActivityRequest, offset, limit);
    },
    [dailyActivityRequest],
  );
  const fetchKeyDetail = useCallback(
    (apiKey: string) =>
      dailyActivityRequest === null
        ? Promise.resolve(undefined)
        : ENTITY_API.user
            .aggregated({ ...dailyActivityRequest, apiKey, apiKeyLimit: 1 })
            .then((response) => keyDetailFromResponse(response, apiKey, teams)),
    [dailyActivityRequest, teams],
  );

  return (
    <div className="relative w-full px-4 pt-3 pb-8 sm:px-6">
      <header className="mb-3 flex w-full flex-wrap items-center justify-between gap-3 border-b pb-3">
        <UsageViewSelect
          value={usageView}
          onChange={(value) => void setUrlParams(usageViewPatch(value))}
          userRole={userRole}
          canViewTagUsage={canViewTagUsage}
          isOrgAdmin={isOrgAdmin}
        />
        <div className="flex flex-wrap items-center gap-2">
          {isAdmin && usageView === "global" && (
            <div className="flex items-center gap-2">
              <div className="w-64">
                <UserDropdown value={urlParams.user} onChange={(user) => void setUrlParams({ user })} />
              </div>
              {urlParams.user && <UserRecordLink userId={urlParams.user} />}
            </div>
          )}
          <div className="flex items-center gap-2">
            <AdvancedDatePicker value={dateValue} onValueChange={handleDateChange} presetShortLabel={urlParams.range} />
            <Button variant="outline" onClick={() => void copyToClipboard(window.location.href, "Link copied")}>
              <Link2 />
              Copy Share Link
            </Button>
          </div>
        </div>
      </header>
      {aggregatedFailed && (
        <ErrorState title="Failed to load spend data" message={aggregatedError} onRetry={reloadAggregated} />
      )}
      <div>
        <div>
          {/* Your Usage / Global Usage Panel */}
          {(usageView === "global" || usageView === "my-usage") && (
            <>
              <Tabs
                value={selectedTab}
                onValueChange={(tab: string) => void setUrlParams(usageTabPatch(tab))}
                className="gap-3"
              >
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <TabsList variant="line">
                    <TabsTrigger value="cost" className="flex-none px-3">
                      Overview
                    </TabsTrigger>
                    <TabsTrigger value="models" className="flex-none px-3">
                      Model Activity
                    </TabsTrigger>
                    <TabsTrigger value="keys" className="flex-none px-3">
                      Key Activity
                    </TabsTrigger>
                    <TabsTrigger value="users" className="flex-none px-3">
                      User Activity
                    </TabsTrigger>
                    <TabsTrigger value="mcp" className="flex-none px-3">
                      MCP Server Activity
                    </TabsTrigger>
                    <TabsTrigger value="endpoints" className="flex-none px-3">
                      Endpoint Activity
                    </TabsTrigger>
                    {isAdmin && (
                      <TabsTrigger value="errors" className="flex-none px-3">
                        Errors
                      </TabsTrigger>
                    )}
                  </TabsList>
                  <div className="flex items-center gap-2">
                    <Button variant="outline" size="sm" onClick={() => setIsAiChatOpen(true)}>
                      <Sparkles />
                      Ask AI
                    </Button>
                    <Button variant="outline" size="sm" onClick={() => setIsGlobalExportModalOpen(true)}>
                      <Download />
                      Export Data
                    </Button>
                  </div>
                </div>
                <TabsContent value="cost" keepMounted>
                  <UsageOverview
                    results={sortedDailyResults}
                    totals={totals}
                    loading={loading}
                    requestCountsPending={requestCountsPending}
                    budget={currentUser?.max_budget ?? null}
                    topKeys={
                      <TopKeyView
                        topKeys={topKeys}
                        teams={null}
                        topKeysLimit={topKeysLimit}
                        setTopKeysLimit={setTopKeysLimit}
                      />
                    }
                    topUsers={<TopUsersView fetchUserPage={fetchUserPage} />}
                    gatewayByEndpoint={
                      gatewayActivity && gatewayActivity.by_route.length > 0 ? (
                        <Panel
                          testId="gateway-requests-by-endpoint"
                          title="Gateway Requests by Endpoint"
                          subtitle="Successful and failed requests per route"
                          bodyClassName="px-2 pt-4"
                        >
                          <StackedUsageChart
                            data={gatewayRequestsByRoute.map((row) => ({
                              route: row.route,
                              Successful: row.successful_requests,
                              Failed: row.failed_requests,
                            }))}
                            series={["Successful", "Failed"]}
                            colors={["#2b3fd6", "#ef4444"]}
                            xKey="route"
                            format={(value) => value.toLocaleString()}
                            className="h-64"
                          />
                        </Panel>
                      ) : null
                    }
                    topAgents={
                      showTopAgents ? (
                        <TopAgents
                          rows={tagSummary}
                          daily={tagDaily}
                          loading={tagSummaryLoading || tagDailyLoading}
                          totalTokens={totals.tokens}
                          onOpenAgent={openAgentActivity}
                        />
                      ) : null
                    }
                    providerBreakdown={
                      <SpendByProvider loading={loading} isDateChanging={false} providerSpend={providerSpend} />
                    }
                  />
                </TabsContent>

                {/* Activity Panel */}
                <TabsContent value="models" keepMounted>
                  <div className="mt-2 mb-4 flex flex-wrap items-center justify-between gap-3">
                    <InputGroup className="max-w-md">
                      <InputGroupAddon>
                        <Search className="size-4 text-muted-foreground" />
                      </InputGroupAddon>
                      <InputGroupInput
                        aria-label="Search models"
                        placeholder="Search by model name"
                        value={modelQuery}
                        onChange={(event) => setModelQuery(event.target.value)}
                      />
                      {trimmedModelQuery !== "" && (
                        <InputGroupAddon align="inline-end">
                          <InputGroupButton
                            size="icon-xs"
                            aria-label="Clear model search"
                            onClick={() => setModelQuery("")}
                          >
                            <X />
                          </InputGroupButton>
                        </InputGroupAddon>
                      )}
                    </InputGroup>
                    <ModelViewToggle value={modelViewType} onChange={setModelViewType} />
                  </div>
                  {trimmedModelQuery !== "" &&
                  Object.keys(modelMetrics).length > 0 &&
                  Object.keys(filteredModelMetrics).length === 0 ? (
                    <p className="rounded-lg border p-6 text-center text-sm text-muted-foreground">
                      No models match &quot;{trimmedModelQuery}&quot; in this date range
                    </p>
                  ) : (
                    <ActivityMetrics
                      modelMetrics={filteredModelMetrics}
                      fetchTopApiKeys={dailyActivityRequest ? fetchTopApiKeys : undefined}
                    />
                  )}
                </TabsContent>
                <TabsContent value="keys" keepMounted>
                  <KeyActivityPanel
                    summary={summaryMetrics}
                    summaryLoading={loading}
                    fetchKeyPage={fetchKeyPage}
                    fetchKeyDetail={fetchKeyDetail}
                    teams={teams}
                    searchKeys={searchKeys}
                  />
                </TabsContent>
                <TabsContent value="users" keepMounted>
                  <UserActivityPanel fetchUserPage={fetchUserPage} />
                </TabsContent>
                <TabsContent value="mcp" keepMounted>
                  <ActivityMetrics modelMetrics={mcpServerMetrics} />
                </TabsContent>
                <TabsContent value="endpoints" keepMounted>
                  <EndpointUsage userSpendData={userSpendData} />
                </TabsContent>
                {isAdmin && (
                  <TabsContent value="errors">
                    <ErrorsTab
                      activity={requestErrorActivity}
                      loading={requestErrorActivity === null && !requestErrorsFailed}
                      failed={requestErrorsFailed}
                      userScoped={effectiveUserId !== null}
                    />
                  </TabsContent>
                )}
              </Tabs>
            </>
          )}
          {/* Organization Usage Panel */}

          {usageView === "organization" && (
            <EntityUsage
              accessToken={accessToken}
              entityType="organization"
              {...entitySelection("organization")}
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
              {...entitySelection("team")}
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
              {...entitySelection("customer")}
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
                {...entitySelection("tag")}
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
              {...entitySelection("agent")}
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
              {...entitySelection("user")}
              userID={userID}
              userRole={userRole}
              entityList={null}
              premiumUser={premiumUser}
              dateValue={dateValue}
            />
          )}
          {/* User Agent Activity Panel */}
          {usageView === "user-agent-activity" && (
            <UserAgentActivity
              accessToken={accessToken}
              userRole={userRole}
              dateValue={dateValue}
              selectedTags={selectedEntitiesFromParams(urlParams, "tag")}
              onSelectedTagsChange={(tags) => void setUrlParams(entitySelectionPatch("tag", tags))}
            />
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
        entityType="user"
        onExport={(exportType, format) =>
          dailyActivityRequest
            ? ENTITY_API.user.exportRows(dailyActivityRequest, exportType, format)
            : Promise.reject(new Error("Missing access token or date range"))
        }
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
