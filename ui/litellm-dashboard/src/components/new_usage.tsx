/**
 * New Usage Page
 * 
 * Uses the new `/user/daily/activity` endpoint to get daily activity data for a user.
 * 
 * Works at 1m+ spend logs, by querying an aggregate table instead.
 */

import React, { useState, useEffect } from "react";
import { 
  BarChart, Card, Title, Text, 
  Grid, Col, TabGroup, TabList, Tab, 
  TabPanel, TabPanels, DonutChart,
  Table, TableHead, TableRow, 
  TableHeaderCell, TableBody, TableCell,
  Subtitle, DateRangePicker, DateRangePickerValue
} from "@tremor/react";
import { AreaChart } from "@tremor/react";

import { userDailyActivityCall, tagListCall } from "./networking";
import { Tag } from "./tag_management/types";
import ViewUserSpend from "./view_user_spend";
import TopKeyView from "./top_key_view";
import { ActivityMetrics, processActivityData } from './activity_metrics';
import { SpendMetrics, DailyData, ModelActivityData, MetricWithMetadata, KeyMetricWithMetadata } from './usage/types';
import EntityUsage from './entity_usage';
import { old_admin_roles, v2_admin_role_names, all_admin_roles, rolesAllowedToSeeUsage, rolesWithWriteAccess, internalUserRoles } from '../utils/roles';
import { Team } from "./key_team_helpers/key_list";
import { EntityList } from "./entity_usage";

interface NewUsagePageProps {
  accessToken: string | null;
  userRole: string | null;
  userID: string | null;
  teams: Team[];
  premiumUser: boolean;
}

const NewUsagePage: React.FC<NewUsagePageProps> = ({
  accessToken,
  userRole,
  userID,
  teams,
  premiumUser
}) => {
  const [userSpendData, setUserSpendData] = useState<{
    results: DailyData[];
    metadata: any;
  }>({ results: [], metadata: {} });

  // Add date range state
  const [dateValue, setDateValue] = useState<DateRangePickerValue>({
    from: new Date(Date.now() - 28 * 24 * 60 * 60 * 1000),
    to: new Date(),
  });

  const [allTags, setAllTags] = useState<EntityList[]>([]); 

  const getAllTags = async () => {
    if (!accessToken) {
      return;
    }
    const tags = await tagListCall(accessToken);
    setAllTags(Object.values(tags).map((tag: Tag) => ({
      label: tag.name,
      value: tag.name
    })));
  };

  useEffect(() => {
    getAllTags();
  }, [accessToken]);

  // Derived states from userSpendData
  const totalSpend = userSpendData.metadata?.total_spend || 0;

  // Calculate top models from the breakdown data
  const getTopModels = () => {
    const modelSpend: { [key: string]: MetricWithMetadata } = {};
    userSpendData.results.forEach(day => {
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
              cache_creation_input_tokens: 0
            },
            metadata: {}
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
        tokens: metrics.metrics.total_tokens
      }))
      .sort((a, b) => b.spend - a.spend)
      .slice(0, 5);
  };

  // Calculate provider spend from the breakdown data
  const getProviderSpend = () => {
    const providerSpend: { [key: string]: MetricWithMetadata } = {};
    userSpendData.results.forEach(day => {
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
              cache_creation_input_tokens: 0
            },
            metadata: {}
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
    
    return Object.entries(providerSpend)
      .map(([provider, metrics]) => ({
        provider,
        spend: metrics.metrics.spend,
        requests: metrics.metrics.api_requests,
        successful_requests: metrics.metrics.successful_requests,
        failed_requests: metrics.metrics.failed_requests,
        tokens: metrics.metrics.total_tokens
      }));
  };

  // Calculate top API keys from the breakdown data
  const getTopKeys = () => {
    const keySpend: { [key: string]: KeyMetricWithMetadata } = {};
    userSpendData.results.forEach(day => {
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
              cache_creation_input_tokens: 0
            },
            metadata: {
              key_alias: metrics.metadata.key_alias
            }
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
        key_alias: metrics.metadata.key_alias || "-", // Using truncated key as alias
        spend: metrics.metrics.spend,
      }))
      .sort((a, b) => b.spend - a.spend)
      .slice(0, 5);
  };

  const fetchUserSpendData = async () => {
    if (!accessToken || !dateValue.from || !dateValue.to) return;
    const startTime = dateValue.from;
    const endTime = dateValue.to;
    
    try {
      // Get first page
      const firstPageData = await userDailyActivityCall(accessToken, startTime, endTime);
      
      // Check if we need to fetch more pages
      if (firstPageData.metadata.total_pages > 10) {
        throw new Error("Too many pages of data (>10). Please select a smaller date range.");
      }

      // If only one page, just set the data
      if (firstPageData.metadata.total_pages === 1) {
        setUserSpendData(firstPageData);
        return;
      }

      // Fetch all pages
      const allResults = [...firstPageData.results];
      
      for (let page = 2; page <= firstPageData.metadata.total_pages; page++) {
        const pageData = await userDailyActivityCall(accessToken, startTime, endTime, page);
        allResults.push(...pageData.results);
      }

      // Combine all results with the first page's metadata
      setUserSpendData({
        results: allResults,
        metadata: firstPageData.metadata
      });
    } catch (error) {
      console.error("Error fetching user spend data:", error);
      throw error;
    }
  };

  useEffect(() => {
    fetchUserSpendData();
  }, [accessToken, dateValue]);

  const modelMetrics = processActivityData(userSpendData, "models");
  const keyMetrics = processActivityData(userSpendData, "api_keys");

  return (
    <div style={{ width: "100%" }} className="p-8">
      {all_admin_roles.includes(userRole || "") ?
      <Text className="text-sm text-gray-500 mb-4">
        Note: If you see key/model-level inconsistencies between Global View and Team Usage, it&apos;s because the Global View was missing spend when user_id = null, prior to v1.71.2. <a href="https://github.com/BerriAI/litellm/issues/10876" className="text-blue-500 hover:text-blue-700 ml-1">Learn more here</a>.
      </Text>
      : null}
      <TabGroup>
        <TabList variant="solid" className="mt-1">
          {all_admin_roles.includes(userRole || "") ? <Tab>Global Usage</Tab> : <Tab>Your Usage</Tab>}
          <Tab>Team Usage</Tab>
          {all_admin_roles.includes(userRole || "") ? <Tab>Tag Usage</Tab> : <></>}
        </TabList>
        <TabPanels>
          {/* Your Usage Panel */}
          <TabPanel>
            <Grid numItems={2} className="gap-2 w-full mb-4">
              <Col>
                <Text>Select Time Range</Text>
                <DateRangePicker
                  enableSelect={true}
                  value={dateValue}
                  onValueChange={(value) => {
                    setDateValue(value);
                  }}
                />
              </Col>
            </Grid>
            <TabGroup>
              <TabList variant="solid" className="mt-1">
                <Tab>Cost</Tab>
                <Tab>Model Activity</Tab>
                <Tab>Key Activity</Tab>
              </TabList>
              <TabPanels>
                {/* Cost Panel */}
                <TabPanel>
                  <Grid numItems={2} className="gap-2 w-full">
                    {/* Total Spend Card */}
                    <Col numColSpan={2}>
                      <Text className="text-tremor-default text-tremor-content dark:text-dark-tremor-content mb-2 mt-2 text-lg">
                        Project Spend {new Date().toLocaleString('default', { month: 'long' })} 1 - {new Date(new Date().getFullYear(), new Date().getMonth() + 1, 0).getDate()}
                      </Text>
                      
                      <ViewUserSpend
                        userID={userID}
                        userRole={userRole}
                        accessToken={accessToken}
                        userSpend={totalSpend}
                        selectedTeam={null}
                        userMaxBudget={null}
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
                              ${((totalSpend || 0) / (userSpendData.metadata?.total_api_requests || 1)).toFixed(4)}
                            </Text>
                          </Card>
                        </Grid>
                      </Card>
                    </Col>

                    {/* Daily Spend Chart */}
                    <Col numColSpan={2}>
                      <Card>
                        <Title>Daily Spend</Title>
                        <BarChart
                          data={[...userSpendData.results].sort((a, b) => 
                            new Date(a.date).getTime() - new Date(b.date).getTime()
                          )}
                          index="date"
                          categories={["metrics.spend"]}
                          colors={["cyan"]}
                          valueFormatter={(value) => `$${value.toFixed(2)}`}
                          yAxisWidth={100}
                          showLegend={false}
                          customTooltip={({ payload, active }) => {
                            if (!active || !payload?.[0]) return null;
                            const data = payload[0].payload;
                            return (
                              <div className="bg-white p-4 shadow-lg rounded-lg border">
                                <p className="font-bold">{data.date}</p>
                                <p className="text-cyan-500">Spend: ${data.metrics.spend.toFixed(2)}</p>
                                <p className="text-gray-600">Requests: {data.metrics.api_requests}</p>
                                <p className="text-gray-600">Successful: {data.metrics.successful_requests}</p>
                                <p className="text-gray-600">Failed: {data.metrics.failed_requests}</p>
                                <p className="text-gray-600">Tokens: {data.metrics.total_tokens}</p>
                              </div>
                            );
                          }}
                        />
                      </Card>
                    </Col>
                    {/* Top API Keys */}
                    <Col numColSpan={1}>
                      <Card className="h-full">
                        <Title>Top API Keys</Title>
                        <TopKeyView
                          topKeys={getTopKeys()}
                          accessToken={accessToken}
                          userID={userID}
                          userRole={userRole}
                          teams={null}
                          premiumUser={premiumUser}
                        />
                      </Card>
                    </Col>

                    {/* Top Models */}
                    <Col numColSpan={1}>
                      <Card className="h-full">
                        <div className="flex justify-between items-center mb-4">
                          <Title>Top Models</Title>
                        </div>
                        <BarChart
                          className="mt-4 h-40"
                          data={getTopModels()}
                          index="key"
                          categories={["spend"]}
                          colors={["cyan"]}
                          valueFormatter={(value) => `$${value.toFixed(2)}`}
                          layout="vertical"
                          yAxisWidth={200}
                          showLegend={false}
                          customTooltip={({ payload, active }) => {
                            if (!active || !payload?.[0]) return null;
                            const data = payload[0].payload;
                            return (
                              <div className="bg-white p-4 shadow-lg rounded-lg border">
                                <p className="font-bold">{data.key}</p>
                                <p className="text-cyan-500">Spend: ${data.spend.toFixed(2)}</p>
                                <p className="text-gray-600">Total Requests: {data.requests.toLocaleString()}</p>
                                <p className="text-green-600">Successful: {data.successful_requests.toLocaleString()}</p>
                                <p className="text-red-600">Failed: {data.failed_requests.toLocaleString()}</p>
                                <p className="text-gray-600">Tokens: {data.tokens.toLocaleString()}</p>
                              </div>
                            );
                          }}
                        />
                      </Card>
                    </Col>

                    {/* Spend by Provider */}
                    <Col numColSpan={2}>
                      <Card className="h-full">
                        <div className="flex justify-between items-center mb-4">
                          <Title>Spend by Provider</Title>
                        </div>
                        <Grid numItems={2}>
                          <Col numColSpan={1}>
                            <DonutChart
                              className="mt-4 h-40"
                              data={getProviderSpend()}
                              index="provider"
                              category="spend"
                              valueFormatter={(value) => `$${value.toFixed(2)}`}
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
                                  .filter(provider => provider.spend > 0)
                                  .map((provider) => (
                                    <TableRow key={provider.provider}>
                                      <TableCell>{provider.provider}</TableCell>
                                      <TableCell>
                                        ${provider.spend < 0.00001
                                            ? "less than 0.00001" 
                                            : provider.spend.toFixed(2)}
                                    </TableCell>
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
              </TabPanels>
            </TabGroup>
          </TabPanel>

          {/* Team Usage Panel */}
          <TabPanel>
            <EntityUsage 
              accessToken={accessToken}
              entityType="team"
              userID={userID}
              userRole={userRole}
              entityList={teams?.map(team => ({
                label: team.team_alias,
                value: team.team_id
              })) || null}
              premiumUser={premiumUser}
            />
          </TabPanel>

          {/* Tag Usage Panel */}
          <TabPanel>
            <EntityUsage 
              accessToken={accessToken}
              entityType="tag"
              userID={userID}
              userRole={userRole}
              entityList={allTags}
              premiumUser={premiumUser}
            />
          </TabPanel>

        </TabPanels>
      </TabGroup>
    </div>
  );
};

// Add this helper function to process model-specific activity data
const getModelActivityData = (userSpendData: {
  results: DailyData[];
  metadata: any;
}) => {
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
          daily_data: []
        };
      }
      
      modelData[model].total_requests += metrics.metrics.api_requests;
      modelData[model].total_tokens += metrics.metrics.total_tokens;
      modelData[model].daily_data.push({
        date: day.date,
        api_requests: metrics.metrics.api_requests,
        total_tokens: metrics.metrics.total_tokens
      });
    });
  });

  return modelData;
};

// Add this helper function for number formatting
function valueFormatterNumbers(number: number) {
  const formatter = new Intl.NumberFormat('en-US', {
    maximumFractionDigits: 0,
    notation: 'compact',
    compactDisplay: 'short',
  });
  return formatter.format(number);
}

export default NewUsagePage;