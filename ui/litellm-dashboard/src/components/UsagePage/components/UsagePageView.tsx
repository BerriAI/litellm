/**
 * New Usage Page
 *
 * Uses the new `/user/daily/activity` endpoint to get daily activity data for a user.
 *
 * Works at 1m+ spend logs, by querying an aggregate table instead.
 */

import {
  BarChart,
  Card,
  Col,
  DateRangePickerValue,
  DonutChart,
  Grid,
  Tab,
  TabGroup,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeaderCell,
  TableRow,
  TabList,
  TabPanel,
  TabPanels,
  Text,
  Title,
} from "@tremor/react";
import { Alert, Badge } from "antd";
import React, { useCallback, useEffect, useMemo, useState } from "react";

import { useAgents } from "@/app/(dashboard)/hooks/agents/useAgents";
import { useCustomers } from "@/app/(dashboard)/hooks/customers/useCustomers";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import { Button } from "@tremor/react";
import { all_admin_roles } from "../../../utils/roles";
import { ActivityMetrics, processActivityData } from "../../activity_metrics";
import CloudZeroExportModal from "../../cloudzero_export_modal";
import EntityUsageExportModal from "../../EntityUsageExport";
import { Team } from "../../key_team_helpers/key_list";
import { Organization, tagListCall, userDailyActivityAggregatedCall, userDailyActivityCall } from "../../networking";
import { getProviderLogoAndName } from "../../provider_info_helpers";
import AdvancedDatePicker from "../../shared/advanced_date_picker";
import { ChartLoader } from "../../shared/chart_loader";
import { Tag } from "../../tag_management/types";
import UserAgentActivity from "../../user_agent_activity";
import ViewUserSpend from "../../view_user_spend";
import { DailyData, KeyMetricWithMetadata, MetricWithMetadata } from "../types";
import { valueFormatterSpend } from "../utils/value_formatters";
import EntityUsage, { EntityList } from "./EntityUsage/EntityUsage";
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
  const { data: customers = [] } = useCustomers(accessToken, userRole);
  const { data: agentsResponse } = useAgents(accessToken, userRole);
  const [modelViewType, setModelViewType] = useState<"groups" | "individual">("groups");
  const [isCloudZeroModalOpen, setIsCloudZeroModalOpen] = useState(false);
  const [isGlobalExportModalOpen, setIsGlobalExportModalOpen] = useState(false);
  const [showOrganizationBanner, setShowOrganizationBanner] = useState(true);
  const [showCustomerBanner, setShowCustomerBanner] = useState(true);
  const [usageView, setUsageView] = useState<UsageOption>("global");
  const [showAgentBanner, setShowAgentBanner] = useState(true);
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

  // Derived states from userSpendData
  const totalSpend = userSpendData.metadata?.total_spend || 0;

  // Calculate top models from the breakdown data
  const getTopModels = () => {
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
      .slice(0, 5);
  };

  const getTopModelGroups = () => {
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
      .slice(0, 5);
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
  const getTopKeys = () => {
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
      .slice(0, 5);
  };

  const fetchUserSpendData = useCallback(async () => {
    if (!accessToken || !dateValue.from || !dateValue.to) return;

    setLoading(true);

    // Create new Date objects to avoid mutating the original dates
    const startTime = new Date(dateValue.from);
    const endTime = new Date(dateValue.to);

    try {
      // Prefer aggregated endpoint to avoid many page requests
      try {
        const aggregated = await userDailyActivityAggregatedCall(accessToken, startTime, endTime);
        setUserSpendData(aggregated);
        return;
      } catch (e) {
        // Fallback to paginated calls if aggregated endpoint is unavailable
      }

      const firstPageData = await userDailyActivityCall(accessToken, startTime, endTime);

      if (firstPageData.metadata.total_pages <= 1) {
        setUserSpendData(firstPageData);
        return;
      }

      const allResults = [...firstPageData.results];
      const aggregatedMetadata = { ...firstPageData.metadata };

      for (let page = 2; page <= firstPageData.metadata.total_pages; page++) {
        const pageData = await userDailyActivityCall(accessToken, startTime, endTime, page);
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
  }, [accessToken, dateValue.from, dateValue.to]);

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

  const modelMetrics = processActivityData(userSpendData, "models");
  const keyMetrics = processActivityData(userSpendData, "api_keys");
  const mcpServerMetrics = processActivityData(userSpendData, "mcp_servers");

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
            <Badge color="blue" count="New">
              <UsageViewSelect
                value={usageView}
                onChange={(value) => setUsageView(value)}
                isAdmin={all_admin_roles.includes(userRole || "")}
              />
            </Badge>
            <AdvancedDatePicker value={dateValue} onValueChange={handleDateChange} />
          </div>
          {/* Your Usage Panel */}
          {usageView === "global" && (
            <TabGroup>
              <div className="flex justify-between items-center">
                <TabList variant="solid" className="mt-1">
                  <Tab>Cost</Tab>
                  <Tab>Model Activity</Tab>
                  <Tab>Key Activity</Tab>
                  <Tab>MCP Server Activity</Tab>
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
                      <Text className="text-tremor-default text-tremor-content dark:text-dark-tremor-content mb-2 mt-2 text-lg">
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

                      <ViewUserSpend userSpend={totalSpend} selectedTeam={null} userMaxBudget={null} />
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
                            <Title>Failed Requests</Title>
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
                        <TopKeyView topKeys={getTopKeys()} teams={null} />
                      </Card>
                    </Col>

                    {/* Top Models */}
                    <Col numColSpan={1}>
                      <Card className="h-full">
                        <div className="flex justify-between items-center mb-4">
                          <Title>{modelViewType === "groups" ? "Top Public Model Names" : "Top Litellm Models"}</Title>
                          <div className="flex bg-gray-100 rounded-lg p-1">
                            <button
                              className={`px-3 py-1 text-sm rounded-md transition-colors ${
                                modelViewType === "groups"
                                  ? "bg-white shadow-sm text-gray-900"
                                  : "text-gray-600 hover:text-gray-900"
                              }`}
                              onClick={() => setModelViewType("groups")}
                            >
                              Public Model Name
                            </button>
                            <button
                              className={`px-3 py-1 text-sm rounded-md transition-colors ${
                                modelViewType === "individual"
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
                          <BarChart
                            className="mt-4 h-40"
                            data={modelViewType === "groups" ? getTopModelGroups() : getTopModels()}
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
                                  <p className="text-gray-600">Total Requests: {data.requests.toLocaleString()}</p>
                                  <p className="text-green-600">
                                    Successful: {data.successful_requests.toLocaleString()}
                                  </p>
                                  <p className="text-red-600">Failed: {data.failed_requests.toLocaleString()}</p>
                                  <p className="text-gray-600">Tokens: {data.tokens.toLocaleString()}</p>
                                </div>
                              );
                            }}
                          />
                        )}
                      </Card>
                    </Col>

                    {/* Spend by Provider */}
                    <Col numColSpan={2}>
                      <Card className="h-full">
                        <div className="flex justify-between items-center mb-4">
                          <Title>Spend by Provider</Title>
                        </div>
                        {loading ? (
                          <ChartLoader isDateChanging={isDateChanging} />
                        ) : (
                          <Grid numItems={2}>
                            <Col numColSpan={1}>
                              <DonutChart
                                className="mt-4 h-40"
                                data={getProviderSpend()}
                                index="provider"
                                category="spend"
                                valueFormatter={(value) => `$${formatNumberWithCommas(value, 2)}`}
                                colors={["cyan"]}
                              />
                            </Col>
                            <Col numColSpan={1}>
                              <Table>
                                <TableHead>
                                  <TableRow>
                                    <TableHeaderCell>Provider</TableHeaderCell>
                                    <TableHeaderCell>Spend</TableHeaderCell>
                                    <TableHeaderCell className="text-green-600">Successful</TableHeaderCell>
                                    <TableHeaderCell className="text-red-600">Failed</TableHeaderCell>
                                    <TableHeaderCell>Tokens</TableHeaderCell>
                                  </TableRow>
                                </TableHead>
                                <TableBody>
                                  {getProviderSpend()
                                    .filter((provider) => provider.spend > 0)
                                    .map((provider) => (
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
                                                      "w-4 h-4 rounded-full bg-gray-200 flex items-center justify-center text-xs";
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
                            </Col>
                          </Grid>
                        )}
                      </Card>
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
              </TabPanels>
            </TabGroup>
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
