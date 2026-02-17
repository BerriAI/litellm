/**
 * New Usage Page
 *
 * Uses the new `/user/daily/activity` endpoint to get daily activity data for a user.
 *
 * Works at 1m+ spend logs, by querying an aggregate table instead.
 */

import { InfoCircleOutlined, LoadingOutlined, UserOutlined } from "@ant-design/icons";
import {
  BarChart,
  Card,
  Col,
  DateRangePickerValue,
  Grid,
  Tab,
  TabGroup,
  TabList,
  TabPanel,
  TabPanels,
  Text,
  Title
} from "@tremor/react";
import { Alert, Segmented, Select, Tooltip } from "antd";
import { useDebouncedState } from "@tanstack/react-pacer/debouncer";
import React, { useCallback, useEffect, useMemo, useState, type UIEvent } from "react";

import { useAgents } from "@/app/(dashboard)/hooks/agents/useAgents";
import { useCustomers } from "@/app/(dashboard)/hooks/customers/useCustomers";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useCurrentUser } from "@/app/(dashboard)/hooks/users/useCurrentUser";
import { useInfiniteUsers } from "@/app/(dashboard)/hooks/users/useUsers";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import { Button } from "@tremor/react";
import { all_admin_roles } from "../../../utils/roles";
import { ActivityMetrics, processActivityData } from "../../activity_metrics";
import CloudZeroExportModal from "../../cloudzero_export_modal";
import EntityUsageExportModal from "../../EntityUsageExport";
import { Team } from "../../key_team_helpers/key_list";
import { Organization, tagListCall, userDailyActivityAggregatedCall, userDailyActivityCall } from "../../networking";
import AdvancedDatePicker from "../../shared/advanced_date_picker";
import { ChartLoader } from "../../shared/chart_loader";
import { Tag } from "../../tag_management/types";
import UserAgentActivity from "../../user_agent_activity";
import ViewUserSpend from "../../view_user_spend";
import { DailyData, KeyMetricWithMetadata, MetricWithMetadata } from "../types";
import { valueFormatterSpend } from "../utils/value_formatters";
import EndpointUsage from "./EndpointUsage/EndpointUsage";
import EntityUsage, { EntityList } from "./EntityUsage/EntityUsage";
import SpendByProvider from "./EntityUsage/SpendByProvider";
import TopKeyView from "./EntityUsage/TopKeyView";
import { UsageOption, UsageViewSelect } from "./UsageViewSelect/UsageViewSelect";

interface UsagePageProps {
  teams: Team[];
  organizations: Organization[];
}

const UsagePage: React.FC<UsagePageProps> = ({ teams, organizations }) => {
  const { accessToken, userRole, userId: userID, premiumUser } = useAuthorized();
  const [userSpendData, setUserSpendData] = useState<{
    results: DailyData[];
    metadata: any;
  }>({ results: [], metadata: {} });

  // Separate loading states for better UX
  const [loading, setLoading] = useState(false);
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
  console.log(`currentUser: ${JSON.stringify(currentUser)}`);
  console.log(`currentUser max budget: ${currentUser?.max_budget}`);
  const isAdmin = all_admin_roles.includes(userRole || "");

  // Debounced search for user selector
  const [userSearchInput, setUserSearchInput] = useState("");
  const [debouncedUserSearch, setDebouncedUserSearch] = useDebouncedState("", {
    wait: 300,
  });

  const {
    data: usersInfiniteData,
    fetchNextPage: fetchNextUsersPage,
    hasNextPage: hasNextUsersPage,
    isFetchingNextPage: isFetchingNextUsersPage,
    isLoading: isLoadingUsers,
  } = useInfiniteUsers(50, debouncedUserSearch || undefined);

  const userOptions = useMemo(() => {
    if (!usersInfiniteData?.pages) return [];
    const seen = new Set<string>();
    const result: { value: string; label: string }[] = [];
    for (const page of usersInfiniteData.pages) {
      for (const user of page.users) {
        if (seen.has(user.user_id)) continue;
        seen.add(user.user_id);
        result.push({
          value: user.user_id,
          label: user.user_alias
            ? `${user.user_alias} (${user.user_id})`
            : user.user_email
              ? `${user.user_email} (${user.user_id})`
              : user.user_id,
        });
      }
    }
    return result;
  }, [usersInfiniteData]);

  const handleUserSearchChange = (value: string) => {
    setUserSearchInput(value);
    setDebouncedUserSearch(value);
  };

  const handleUserPopupScroll = (e: UIEvent<HTMLDivElement>) => {
    const target = e.currentTarget;
    const scrollRatio =
      (target.scrollTop + target.clientHeight) / target.scrollHeight;
    if (scrollRatio >= 0.8 && hasNextUsersPage && !isFetchingNextUsersPage) {
      fetchNextUsersPage();
    }
  };

  // For admins: null means global view (all users), a string means filter by that user
  // For non-admins: always set to their own user ID
  const [selectedUserId, setSelectedUserId] = useState<string | null>(
    isAdmin ? null : (userID || null)
  );
  const [modelViewType, setModelViewType] = useState<"groups" | "individual">("groups");
  const [isCloudZeroModalOpen, setIsCloudZeroModalOpen] = useState(false);
  const [isGlobalExportModalOpen, setIsGlobalExportModalOpen] = useState(false);
  const [showOrganizationBanner, setShowOrganizationBanner] = useState(true);
  const [showCustomerBanner, setShowCustomerBanner] = useState(true);
  const [usageView, setUsageView] = useState<UsageOption>("global");
  const [showAgentBanner, setShowAgentBanner] = useState(true);
  const [topKeysLimit, setTopKeysLimit] = useState<number>(5);
  const [topModelsLimit, setTopModelsLimit] = useState<number>(5);
  const getAllTags = async () => {
    if (!accessToken) {
      return;
    }
    const tags = await tagListCall(accessToken);
    setAllTags(
      Object.values(tags).map((tag: Tag) => ({
        label: tag.name,
        value: tag.name,
      })),
    );
  };

  useEffect(() => {
    getAllTags();
  }, [accessToken]);

  // Sync selectedUserId when auth state settles (isAdmin/userID may be null on initial render)
  useEffect(() => {
    if (!isAdmin && userID) {
      setSelectedUserId(userID);
    }
  }, [isAdmin, userID]);

  // Derived states from userSpendData
  const totalSpend = userSpendData.metadata?.total_spend || 0;

  // Calculate top models from the breakdown data
  const getTopModels = (limit: number = 5) => {
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
      .slice(0, limit);
  };

  const getTopModelGroups = (limit: number = 5) => {
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
      .slice(0, limit);
  };

  // Calculate provider spend from the breakdown data
  const getProviderSpend = () => {
    const providerSpend: { [key: string]: MetricWithMetadata } = {};
    userSpendData.results.forEach((day) => {
      Object.entries(day.breakdown.providers || {}).forEach(([provider, metrics]) => {
        if (!providerSpend[provider]) {
          providerSpend[provider] = {
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
        providerSpend[provider].metrics.spend += metrics.metrics.spend;
        providerSpend[provider].metrics.prompt_tokens += metrics.metrics.prompt_tokens;
        providerSpend[provider].metrics.completion_tokens += metrics.metrics.completion_tokens;
        providerSpend[provider].metrics.total_tokens += metrics.metrics.total_tokens;
        providerSpend[provider].metrics.api_requests += metrics.metrics.api_requests;
        providerSpend[provider].metrics.successful_requests += metrics.metrics.successful_requests || 0;
        providerSpend[provider].metrics.failed_requests += metrics.metrics.failed_requests || 0;
        providerSpend[provider].metrics.cache_read_input_tokens += metrics.metrics.cache_read_input_tokens || 0;
        providerSpend[provider].metrics.cache_creation_input_tokens += metrics.metrics.cache_creation_input_tokens || 0;
      });
    });

    return Object.entries(providerSpend).map(([provider, metrics]) => ({
      provider,
      spend: metrics.metrics.spend,
      requests: metrics.metrics.api_requests,
      successful_requests: metrics.metrics.successful_requests,
      failed_requests: metrics.metrics.failed_requests,
      tokens: metrics.metrics.total_tokens,
    }));
  };

  // Calculate top API keys from the breakdown data
  const getTopKeys = (limit: number = 5) => {
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
              tags: metrics.metadata.tags || [], // This gets key-level tags
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

    console.log("debugTags", { keySpend, userSpendData });

    return Object.entries(keySpend)
      .map(([api_key, metrics]) => ({
        api_key,
        key_alias: metrics.metadata.key_alias || "-", // Using truncated key as alias
        tags: metrics.metadata.tags || [], // This will show key-level tags
        spend: metrics.metrics.spend,
      }))
      .sort((a, b) => b.spend - a.spend)
      .slice(0, limit);
  };

  const fetchUserSpendData = useCallback(async () => {
    if (!accessToken || !dateValue.from || !dateValue.to) return;

    // For non-admins, always pass their own user_id
    const effectiveUserId = isAdmin ? selectedUserId : (userID || null);

    setLoading(true);

    // Create new Date objects to avoid mutating the original dates
    const startTime = new Date(dateValue.from);
    const endTime = new Date(dateValue.to);

    try {
      // Prefer aggregated endpoint to avoid many page requests
      try {
        const aggregated = await userDailyActivityAggregatedCall(accessToken, startTime, endTime, effectiveUserId);
        setUserSpendData(aggregated);
        return;
      } catch (e) {
        // Fallback to paginated calls if aggregated endpoint is unavailable
      }

      const firstPageData = await userDailyActivityCall(accessToken, startTime, endTime, 1, effectiveUserId);

      if (firstPageData.metadata.total_pages <= 1) {
        setUserSpendData(firstPageData);
        return;
      }

      const allResults = [...firstPageData.results];
      const aggregatedMetadata = { ...firstPageData.metadata };

      for (let page = 2; page <= firstPageData.metadata.total_pages; page++) {
        const pageData = await userDailyActivityCall(accessToken, startTime, endTime, page, effectiveUserId);
        allResults.push(...pageData.results);
        if (pageData.metadata) {
          aggregatedMetadata.total_spend += pageData.metadata.total_spend || 0;
          aggregatedMetadata.total_api_requests += pageData.metadata.total_api_requests || 0;
          aggregatedMetadata.total_successful_requests += pageData.metadata.total_successful_requests || 0;
          aggregatedMetadata.total_failed_requests += pageData.metadata.total_failed_requests || 0;
          aggregatedMetadata.total_tokens += pageData.metadata.total_tokens || 0;
        }
      }

      setUserSpendData({
        results: allResults,
        metadata: aggregatedMetadata,
      });
    } catch (error) {
      console.error("Error fetching user spend data:", error);
    } finally {
      setLoading(false);
      setIsDateChanging(false);
    }
  }, [accessToken, dateValue.from, dateValue.to, selectedUserId, isAdmin, userID]);

  // Super responsive date change handler
  const handleDateChange = useCallback((newValue: DateRangePickerValue) => {
    // Instant visual feedback
    setIsDateChanging(true);
    setLoading(true);

    // Update date immediately for UI responsiveness
    setDateValue(newValue);
  }, []);

  // Debounced effect for data fetching with shorter delay
  useEffect(() => {
    if (!dateValue.from || !dateValue.to) return;

    const timeoutId = setTimeout(() => {
      fetchUserSpendData();
    }, 50); // Very short debounce

    return () => clearTimeout(timeoutId);
  }, [fetchUserSpendData]);

  const modelMetrics = processActivityData(userSpendData, "models", teams);
  const keyMetrics = processActivityData(userSpendData, "api_keys", teams);
  const mcpServerMetrics = processActivityData(userSpendData, "mcp_servers", teams);

  return (
    <div style={{ width: "100%" }} className="p-8 relative">
      {/* Export Data Button - Positioned in top right corner */}
      {/* {all_admin_roles.includes(userRole || "") && (
        <div className="absolute top-4 right-4 z-10">
          <button
            onClick={() => setIsCloudZeroModalOpen(true)}
            className="bg-indigo-600 hover:bg-indigo-700 text-white px-4 py-2 rounded-full flex items-center gap-2 shadow-lg transition-colors duration-200 text-sm font-medium"
          >
            <svg 
              className="w-4 h-4" 
              fill="none" 
              stroke="currentColor" 
              viewBox="0 0 24 24"
              strokeWidth={2}
            >
              <path 
                strokeLinecap="round" 
                strokeLinejoin="round" 
                d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" 
              />
            </svg>
            Export Data
            <svg 
              className="w-3 h-3" 
              fill="none" 
              stroke="currentColor" 
              viewBox="0 0 24 24"
              strokeWidth={2}
            >
              <path 
                strokeLinecap="round" 
                strokeLinejoin="round" 
                d="M19 9l-7 7-7-7" 
              />
            </svg>
          </button>
        </div>
      )} */}

      {/* Global Date Picker and Tabs - Single Row */}
      <div className="flex items-end justify-between gap-6 mb-6">
        <div className="flex-1">
          <div className="flex items-end justify-between gap-6 mb-4 w-full">
            <UsageViewSelect
              value={usageView}
              onChange={(value) => setUsageView(value)}
              isAdmin={isAdmin}
            />
            <AdvancedDatePicker value={dateValue} onValueChange={handleDateChange} />
          </div>
          {/* Your Usage Panel */}
          {usageView === "global" && (
            <>
            <TabGroup>
              <div className="flex justify-between items-center">
                <TabList variant="solid" className="mt-1">
                  <Tab>Cost</Tab>
                  <Tab>Model Activity</Tab>
                  <Tab>Key Activity</Tab>
                  <Tab>MCP Server Activity</Tab>
                  <Tab>Endpoint Activity</Tab>
                </TabList>
                <Button
                  onClick={() => setIsGlobalExportModalOpen(true)}
                  icon={() => (
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
                      />
                    </svg>
                  )}
                >
                  Export Data
                </Button>
              </div>
              <TabPanels>
                {/* Cost Panel */}
                <TabPanel>
                  <Grid numItems={2} className="gap-2 w-full">
                    {/* Total Spend Card */}
                    <Col numColSpan={2}>
                      <div className="flex items-center gap-4 mt-2 mb-2">
                        <Text className="text-tremor-default text-tremor-content dark:text-dark-tremor-content text-lg">
                          Project Spend{" "}
                          {dateValue.from && dateValue.to && (
                            <>
                              {dateValue.from.toLocaleDateString("en-US", {
                                month: "short",
                                day: "numeric",
                                year: dateValue.from.getFullYear() !== dateValue.to.getFullYear() ? "numeric" : undefined,
                              })}
                              {" - "}
                              {dateValue.to.toLocaleDateString("en-US", {
                                month: "short",
                                day: "numeric",
                                year: "numeric",
                              })}
                            </>
                          )}
                        </Text>
                        {isAdmin && (
                          <div className="flex items-center gap-2">
                            <UserOutlined style={{ fontSize: "14px", color: "#6b7280" }} />
                            <Select
                              showSearch
                              allowClear
                              style={{ width: 300 }}
                              placeholder="All Users (Global View)"
                              value={selectedUserId}
                              onChange={(value) => setSelectedUserId(value ?? null)}
                              filterOption={false}
                              onSearch={handleUserSearchChange}
                              searchValue={userSearchInput}
                              onPopupScroll={handleUserPopupScroll}
                              loading={isLoadingUsers}
                              notFoundContent={isLoadingUsers ? <LoadingOutlined spin /> : "No users found"}
                              options={userOptions}
                              popupRender={(menu) => (
                                <>
                                  {menu}
                                  {isFetchingNextUsersPage && (
                                    <div style={{ textAlign: "center", padding: 8 }}>
                                      <LoadingOutlined spin />
                                    </div>
                                  )}
                                </>
                              )}
                            />
                            {selectedUserId && (
                              <span className="text-xs text-gray-500">
                                Filtering by user
                              </span>
                            )}
                          </div>
                        )}
                      </div>

                      <ViewUserSpend
                        userSpend={totalSpend}
                        selectedTeam={null}
                        userMaxBudget={currentUser?.max_budget || null}
                      />
                    </Col>

                    <Col numColSpan={2}>
                      <Card>
                        <Title>Usage Metrics</Title>
                        <Grid numItems={5} className="gap-4 mt-4">
                          <Card>
                            <Title>Total Requests</Title>
                            <Text className="text-2xl font-bold mt-2">
                              {userSpendData.metadata?.total_api_requests?.toLocaleString() || 0}
                            </Text>
                          </Card>
                          <Card>
                            <Title>Successful Requests</Title>
                            <Text className="text-2xl font-bold mt-2 text-green-600">
                              {userSpendData.metadata?.total_successful_requests?.toLocaleString() || 0}
                            </Text>
                          </Card>
                          <Card>
                            <div className="flex items-center gap-2">
                              <Title>Failed Requests</Title>
                              <Tooltip title="Includes requests that failed to route to a provider, tool usage failures, and other request errors where the provider cannot be determined.">
                                <InfoCircleOutlined className="text-gray-400 hover:text-gray-600" />
                              </Tooltip>
                            </div>
                            <Text className="text-2xl font-bold mt-2 text-red-600">
                              {userSpendData.metadata?.total_failed_requests?.toLocaleString() || 0}
                            </Text>
                          </Card>
                          <Card>
                            <Title>Total Tokens</Title>
                            <Text className="text-2xl font-bold mt-2">
                              {userSpendData.metadata?.total_tokens?.toLocaleString() || 0}
                            </Text>
                          </Card>
                          <Card>
                            <Title>Average Cost per Request</Title>
                            <Text className="text-2xl font-bold mt-2">
                              $
                              {formatNumberWithCommas(
                                (totalSpend || 0) / (userSpendData.metadata?.total_api_requests || 1),
                                4,
                              )}
                            </Text>
                          </Card>
                        </Grid>
                      </Card>
                    </Col>

                    {/* Daily Spend Chart */}
                    <Col numColSpan={2}>
                      <Card>
                        <Title>Daily Spend</Title>
                        {loading ? (
                          <ChartLoader isDateChanging={isDateChanging} />
                        ) : (
                          <BarChart
                            data={[...userSpendData.results].sort(
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
                              return (
                                <div className="bg-white p-4 shadow-lg rounded-lg border">
                                  <p className="font-bold">{data.date}</p>
                                  <p className="text-cyan-500">
                                    Spend: ${formatNumberWithCommas(data.metrics.spend, 2)}
                                  </p>
                                  <p className="text-gray-600">Requests: {data.metrics.api_requests}</p>
                                  <p className="text-gray-600">Successful: {data.metrics.successful_requests}</p>
                                  <p className="text-gray-600">Failed: {data.metrics.failed_requests}</p>
                                  <p className="text-gray-600">Tokens: {data.metrics.total_tokens}</p>
                                </div>
                              );
                            }}
                          />
                        )}
                      </Card>
                    </Col>
                    {/* Top API Keys */}
                    <Col numColSpan={1}>
                      <Card className="h-full">
                        <Title>Top Virtual Keys</Title>
                        <TopKeyView
                          topKeys={getTopKeys(topKeysLimit)}
                          teams={null}
                          topKeysLimit={topKeysLimit}
                          setTopKeysLimit={setTopKeysLimit}
                        />
                      </Card>
                    </Col>

                    {/* Top Models */}
                    <Col numColSpan={1}>
                      <Card className="h-full">
                        <Title>{modelViewType === "groups" ? "Top Public Model Names" : "Top Litellm Models"}</Title>
                        <div className="flex justify-between items-center mb-4">
                          <Segmented
                            options={[
                              { label: "5", value: 5 },
                              { label: "10", value: 10 },
                              { label: "25", value: 25 },
                              { label: "50", value: 50 },
                            ]}
                            value={topModelsLimit}
                            onChange={(value) => setTopModelsLimit(value as number)}
                          />
                          <div className="flex bg-gray-100 rounded-lg p-1">
                            <button
                              className={`px-3 py-1 text-sm rounded-md transition-colors ${modelViewType === "groups"
                                ? "bg-white shadow-sm text-gray-900"
                                : "text-gray-600 hover:text-gray-900"
                                }`}
                              onClick={() => setModelViewType("groups")}
                            >
                              Public Model Name
                            </button>
                            <button
                              className={`px-3 py-1 text-sm rounded-md transition-colors ${modelViewType === "individual"
                                ? "bg-white shadow-sm text-gray-900"
                                : "text-gray-600 hover:text-gray-900"
                                }`}
                              onClick={() => setModelViewType("individual")}
                            >
                              Litellm Model Name
                            </button>
                          </div>
                        </div>
                        {loading ? (
                          <ChartLoader isDateChanging={isDateChanging} />
                        ) : (
                          <div className="relative max-h-[600px] overflow-y-auto">
                            {(() => {
                              const modelData =
                                modelViewType === "groups"
                                  ? getTopModelGroups(topModelsLimit)
                                  : getTopModels(topModelsLimit);
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
                                      <div className="bg-white p-4 shadow-lg rounded-lg border">
                                        <p className="font-bold">{data.key}</p>
                                        <p className="text-cyan-500">Spend: ${formatNumberWithCommas(data.spend, 2)}</p>
                                        <p className="text-gray-600">
                                          Total Requests: {data.requests.toLocaleString()}
                                        </p>
                                        <p className="text-green-600">
                                          Successful: {data.successful_requests.toLocaleString()}
                                        </p>
                                        <p className="text-red-600">Failed: {data.failed_requests.toLocaleString()}</p>
                                        <p className="text-gray-600">Tokens: {data.tokens.toLocaleString()}</p>
                                      </div>
                                    );
                                  }}
                                />
                              );
                            })()}
                          </div>
                        )}
                      </Card>
                    </Col>

                    {/* Spend by Provider */}
                    <Col numColSpan={2}>
                      <SpendByProvider
                        loading={loading}
                        isDateChanging={isDateChanging}
                        providerSpend={getProviderSpend()}
                      />
                    </Col>

                    {/* Usage Metrics */}
                  </Grid>
                </TabPanel>

                {/* Activity Panel */}
                <TabPanel>
                  <ActivityMetrics modelMetrics={modelMetrics} />
                </TabPanel>
                <TabPanel>
                  <ActivityMetrics modelMetrics={keyMetrics} />
                </TabPanel>
                <TabPanel>
                  <ActivityMetrics modelMetrics={mcpServerMetrics} />
                </TabPanel>
                <TabPanel>
                  <EndpointUsage userSpendData={userSpendData} />
                </TabPanel>
              </TabPanels>
            </TabGroup>
            </>
          )}
          {/* Organization Usage Panel */}

          {usageView === "organization" && (
            <>
              {showOrganizationBanner && (
                <Alert
                  banner
                  type="info"
                  message="Organization usage is a new feature."
                  description="Spend is tracked from feature launch and previous data isn't backfilled, so only future usage appears here."
                  closable
                  onClose={() => setShowOrganizationBanner(false)}
                  className="mb-5"
                />
              )}
              <EntityUsage
                accessToken={accessToken}
                entityType="organization"
                userID={userID}
                userRole={userRole}
                dateValue={dateValue}
                entityList={
                  organizations?.map((organization) => ({
                    label: organization.organization_alias,
                    value: organization.organization_id,
                  })) || null
                }
                premiumUser={premiumUser}
              />
            </>
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
            <>
              {showCustomerBanner && (
                <Alert
                  banner
                  type="info"
                  message="Customer usage is a new feature."
                  description="Spend is tracked from feature launch and previous data isn't backfilled, so only future usage appears here."
                  closable
                  onClose={() => setShowCustomerBanner(false)}
                  className="mb-5"
                />
              )}
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
            </>
          )}
          {/* Tag Usage Panel */}
          {usageView === "tag" && (
            <EntityUsage
              accessToken={accessToken}
              entityType="tag"
              userID={userID}
              userRole={userRole}
              entityList={allTags}
              premiumUser={premiumUser}
              dateValue={dateValue}
            />
          )}
          {usageView === "agent" && (
            <>
              {showAgentBanner && (
                <Alert
                  banner
                  type="info"
                  message="Agent usage (A2A) is a new feature."
                  description="Spend is tracked from feature launch and previous data isn't backfilled, so only future usage appears here."
                  closable
                  onClose={() => setShowAgentBanner(false)}
                  className="mb-5"
                />
              )}
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
              />{" "}
            </>
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
    </div>
  );
};

// Add this helper function to process model-specific activity data
const getModelActivityData = (userSpendData: { results: DailyData[]; metadata: any }) => {
  const modelData: {
    [key: string]: {
      total_requests: number;
      total_tokens: number;
      daily_data: Array<{
        date: string;
        api_requests: number;
        total_tokens: number;
      }>;
    };
  } = {};

  userSpendData.results.forEach((day: DailyData) => {
    Object.entries(day.breakdown.models || {}).forEach(([model, metrics]) => {
      if (!modelData[model]) {
        modelData[model] = {
          total_requests: 0,
          total_tokens: 0,
          daily_data: [],
        };
      }

      modelData[model].total_requests += metrics.metrics.api_requests;
      modelData[model].total_tokens += metrics.metrics.total_tokens;
      modelData[model].daily_data.push({
        date: day.date,
        api_requests: metrics.metrics.api_requests,
        total_tokens: metrics.metrics.total_tokens,
      });
    });
  });

  return modelData;
};

export default UsagePage;
